import torch
from torch.nn import functional as F


def mild_augmentation(images):
    padded = F.pad(images, (1, 1, 1, 1), value=1.0)
    result = []
    for image in padded:
        dy, dx = torch.randint(0, 3, (2,)).tolist()
        result.append(image[:, dy : dy + images.shape[2], dx : dx + images.shape[3]])
    return torch.stack(result)


def negative_indices(texts, count, device, image_ids=None):
    rows = []
    for i, text in enumerate(texts):
        candidates = [
            j
            for j, other in enumerate(texts)
            if j != i
            and other != text
            and (image_ids is None or image_ids[j] != image_ids[i])
        ]
        if len(candidates) < count:
            raise ValueError(
                f"Need {count} distinct-semantic negatives per sample; batch has only "
                f"{len(candidates)} candidates. Increase batch size or reduce num_negatives."
            )
        order = torch.randperm(len(candidates))[:count].tolist()
        rows.append([candidates[j] for j in order])
    return torch.tensor(rows, device=device)


def compute_objective(model, scheduler, batch, config):
    images, ids, mask = batch["images"], batch["input_ids"], batch["attention_mask"]
    b = len(images)
    t = torch.randint(
        0, scheduler.config.num_train_timesteps, (b,), device=images.device
    )
    count = config["num_negatives"] if config.get("use_contrast", True) else 0
    if count:
        indices = negative_indices(
            batch["texts"], count, images.device, batch.get("image_ids")
        )
        views = [images, mild_augmentation(images)] + [
            images[indices[:, k]] for k in range(count)
        ]
    else:
        views = [images]

    clean = torch.cat(views)
    all_t = t.repeat(len(views))
    noise = torch.randn_like(clean)
    noisy = scheduler.add_noise(clean, noise, all_t)
    result = model(noisy, all_t, ids, mask, clean=images)
    mse = (
        (result["noise"].float() - noise.float())
        .square()
        .flatten(1)
        .mean(1)
        .reshape(len(views), b)
    )
    normal = mse[0].mean()
    zero = normal.new_zeros(())
    positive, negative, mi_loss = zero, zero, zero
    alignment = result["alignment"]
    if count:
        positive = mse[1].mean()

        negative = torch.exp(-2 * mse[2:]).mean()
        alpha = scheduler.alphas_cumprod.to(images.device)[all_t].reshape(-1, 1, 1, 1)
        x0 = (noisy - (1 - alpha).sqrt() * result["noise"]) / alpha.sqrt()
        features = model.visual_encoder(x0.clamp(-1, 1)).mean(1)
        features = F.normalize(features.float(), dim=-1).reshape(len(views), b, -1)
        positive_score = (features[0] * features[1]).sum(-1) / config["temperature"]
        negative_scores = (
            torch.einsum("bd,kbd->bk", features[0], features[2:])
            / config["temperature"]
        )
        mi_loss = (negative_scores.sum(-1) - positive_score).mean()
    total = (
        normal
        + positive
        + config["beta"] * alignment
        + config["lambda"] * negative
        + mi_loss
    )
    terms = {
        "loss": total,
        "normal_mse": normal,
        "positive_mse": positive,
        "negative_exp": negative,
        "alignment": alignment,
        "mi_loss": mi_loss,
    }
    if not all(torch.isfinite(v).all() for v in terms.values()):
        raise FloatingPointError("Non-finite FSA-CDM objective")
    return total, {k: float(v.detach()) for k, v in terms.items()}
