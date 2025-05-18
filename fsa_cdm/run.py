import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, DDIMScheduler

from .data import (
    Collator,
    MarkupDataset,
    prepare_dataset,
    tokenizer_for,
)
from .model import FSACDM
from .objective import compute_objective
from .metrics import evaluate


class TrainingObjective(nn.Module):
    def __init__(self, model, scheduler, config):
        super().__init__()
        self.model, self.scheduler, self.config = model, scheduler, config

    def forward(self, batch):
        return compute_objective(self.model, self.scheduler, batch, self.config)


def load_config(path):
    config = json.loads(Path(path).read_text())
    if len(config["block_out_channels"]) < 4:
        raise ValueError("FSA-CDM needs four attention scales")
    factor = 2 ** (len(config["block_out_channels"]) - 1)
    if any(size % factor for size in config["image_size"]):
        raise ValueError(f"Image size must be divisible by {factor}")
    if (
        config.get("use_contrast", True)
        and config["batch_size"] <= config["num_negatives"]
    ):
        raise ValueError("Per-device batch_size must exceed num_negatives")
    return config


def scheduler_for(config):
    return DDPMScheduler(
        num_train_timesteps=config["diffusion_steps"],
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        variance_type="fixed_small",
        clip_sample=True,
    )


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    accelerator,
    wrapper,
    optimizer,
    lr_scheduler,
    output,
    config,
    step,
    epoch,
    next_batch,
):
    state = rng_state()
    states = [state]
    if accelerator.num_processes > 1:
        states = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(states, state)
    if accelerator.is_main_process:
        checkpoint = {
            "format": "fsa-cdm-v1",
            "config": config,
            "model": accelerator.unwrap_model(wrapper).model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": accelerator.scaler.state_dict() if accelerator.scaler else None,
            "global_step": step,
            "epoch": epoch,
            "next_batch": next_batch,
            "rng": states,
        }
        temporary = output / "latest.pt.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, output / "latest.pt")
    accelerator.wait_for_everyone()


def train(args):
    config = load_config(args.config)
    if args.batch_size:
        config["batch_size"] = args.batch_size
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "no"),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    set_seed(
        config["seed"],
        device_specific=True,
        deterministic=config.get("deterministic", False),
    )
    if config.get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.pt").exists() and not args.resume:
        raise FileExistsError(
            f"{output}/latest.pt already exists; pass --resume or use a new output directory"
        )
    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    if checkpoint:
        if checkpoint["config"] != config:
            old = {k: v for k, v in checkpoint["config"].items() if k != "text_config"}
            if old != config:
                raise ValueError(
                    "Resume config differs from checkpoint; use the same config and batch size"
                )
        config = checkpoint["config"]
    tokenizer = tokenizer_for(
        config, Path(args.resume).parent / "tokenizer" if checkpoint else None
    )
    dataset = MarkupDataset(args.data, "train", config, args.limit)
    if len(dataset) < config["batch_size"]:
        raise ValueError("Training split must contain at least one full batch")
    generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        collate_fn=Collator(tokenizer, config["max_length"]),
        num_workers=0,
        drop_last=True,
    )
    model = FSACDM(config, pretrained=checkpoint is None)
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    scheduler = scheduler_for(config)
    wrapper = TrainingObjective(model, scheduler, config)
    optimizer = torch.optim.Adam(
        (p for p in wrapper.parameters() if p.requires_grad), lr=config["learning_rate"]
    )
    wrapper, optimizer, loader = accelerator.prepare(wrapper, optimizer, loader)
    total_steps = (
        math.ceil(len(loader) / config.get("gradient_accumulation_steps", 1))
        * config["epochs"]
    )
    warmup = config["warmup_steps"]

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(
            0.0,
            0.5
            * (
                1
                + math.cos(
                    math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))
                )
            ),
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    global_step, first_epoch, first_batch = 0, 0, 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if accelerator.scaler and checkpoint["scaler"]:
            accelerator.scaler.load_state_dict(checkpoint["scaler"])
        global_step, first_epoch, first_batch = (
            checkpoint["global_step"],
            checkpoint["epoch"],
            checkpoint["next_batch"],
        )
        if len(checkpoint["rng"]) != accelerator.num_processes:
            raise ValueError("Exact resume requires the same number of processes")
        restore_rng(checkpoint["rng"][accelerator.process_index])
        del checkpoint
    if args.max_steps and global_step >= args.max_steps:
        raise ValueError("--max-steps must exceed the resumed global step")
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output / "tokenizer")
        (output / "config.json").write_text(json.dumps(config, indent=2))
        info = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_samples": len(dataset),
            "device": str(accelerator.device),
            "world_size": accelerator.num_processes,
            "torch": torch.__version__,
            "ccam_count": sum(m.__class__.__name__ == "CCAM" for m in model.modules()),
            "conventional_attention_count": sum(
                m.__class__.__name__ == "BasicTransformerBlock" for m in model.modules()
            ),
            "data": str(Path(args.data).resolve()),
        }
        (output / "run_info.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info), flush=True)
    start = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    wrapper.train()
    for epoch in range(first_epoch, config["epochs"]):
        generator.manual_seed(config["seed"] + epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if epoch == first_epoch and batch_index < first_batch:
                continue
            with accelerator.accumulate(wrapper):
                loss, logs = wrapper(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(wrapper.parameters(), 1.0)
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("Non-finite gradients")
                    logs["grad_norm"] = float(grad_norm)
                optimizer.step()
                if (
                    accelerator.sync_gradients
                    and not accelerator.optimizer_step_was_skipped
                ):
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                logs.update(
                    step=global_step,
                    epoch=epoch,
                    lr=lr_scheduler.get_last_lr()[0],
                    elapsed_seconds=time.monotonic() - start,
                )
                if accelerator.is_main_process:
                    with (output / "train.jsonl").open("a") as file:
                        file.write(json.dumps(logs) + "\n")
                    print(json.dumps(logs), flush=True)
                stop = bool(args.max_steps and global_step >= args.max_steps)
                if global_step % config.get("save_every", 1000) == 0 or stop:
                    save_checkpoint(
                        accelerator,
                        wrapper,
                        optimizer,
                        lr_scheduler,
                        output,
                        config,
                        global_step,
                        epoch,
                        batch_index + 1,
                    )
                if stop:
                    accelerator.end_training()
                    return
    save_checkpoint(
        accelerator,
        wrapper,
        optimizer,
        lr_scheduler,
        output,
        config,
        global_step,
        config["epochs"],
        0,
    )
    accelerator.end_training()


def tensor_image(tensor):
    pixels = (
        ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(pixels[..., 0] if pixels.shape[-1] == 1 else pixels)


@torch.inference_mode()
def generate(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = FSACDM(config, pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    step = checkpoint["global_step"]
    del checkpoint
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device).eval()
    tokenizer = tokenizer_for(config, Path(args.checkpoint).parent / "tokenizer")
    dataset = MarkupDataset(args.data, args.split, config, args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=Collator(tokenizer, config["max_length"]),
    )
    scheduler = scheduler_for(config)
    if args.sampler == "ddim":
        scheduler = DDIMScheduler.from_config(scheduler.config)
    if not 1 <= args.steps <= config["diffusion_steps"]:
        raise ValueError("Inference steps must be within [1, diffusion_steps]")
    scheduler.set_timesteps(args.steps, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = Path(args.output)
    if (output / "generation.json").exists():
        raise FileExistsError(
            f"{output} already contains a generation; choose a new output directory"
        )
    for folder in ["images_pred", "images_gold"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    filenames = []
    for batch in loader:
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        text = model.encode(ids, mask)
        images = torch.randn(
            (len(ids), config["channels"], *config["image_size"]),
            generator=generator,
            device=device,
        )
        for step_index, t in enumerate(scheduler.timesteps):
            noise = model.unet(
                images, t, encoder_hidden_states=text, encoder_attention_mask=mask
            ).sample
            images = scheduler.step(noise, t, images, generator=generator).prev_sample
            if (step_index + 1) % 100 == 0:
                print(
                    f"Sampling batch {len(filenames) // args.batch_size + 1}: {step_index + 1}/{args.steps}",
                    flush=True,
                )
        for i, filename in enumerate(batch["filenames"]):
            if filename in filenames:
                raise ValueError(f"Duplicate output filename: {filename}")
            tensor_image(images[i]).save(output / "images_pred" / filename)
            tensor_image(batch["images"][i]).save(output / "images_gold" / filename)
            filenames.append(filename)
        print(f"Generated {len(filenames)}/{len(dataset)} images", flush=True)
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "training_step": step,
        "split": args.split,
        "data": str(Path(args.data).resolve()),
        "sampler": args.sampler,
        "inference_steps": args.steps,
        "seed": args.seed,
        "filenames": filenames,
    }
    (output / "generation.json").write_text(json.dumps(manifest, indent=2))
    h, w = config["image_size"]
    preview = Image.new("RGB", (w * 2, (h + 20) * min(8, len(filenames))), "white")
    draw = ImageDraw.Draw(preview)
    for i, filename in enumerate(filenames[:8]):
        y = i * (h + 20)
        draw.text((2, y), "Ground truth", fill="black")
        draw.text((w + 2, y), "Generated", fill="black")
        for j, folder in enumerate(["images_gold", "images_pred"]):
            with Image.open(output / folder / filename) as image:
                preview.paste(image.convert("RGB"), (j * w, y + 20))
    preview.save(output / "preview.png")
    evaluate(output)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    parser = argparse.ArgumentParser(
        description="Train and generate markup images with FSA-CDM"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument(
        "--dataset",
        choices=["math", "tables", "music", "molecules"],
        required=True,
    )
    prep.add_argument("--output", required=True)
    prep.add_argument("--parquet-dir")
    prep.add_argument("--limit", type=int)
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--resume")
    g = sub.add_parser("generate")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--data", required=True)
    g.add_argument("--output", required=True)
    g.add_argument("--split", choices=["train", "val", "test"], default="test")
    g.add_argument("--batch-size", type=int, default=2)
    g.add_argument("--limit", type=int)
    g.add_argument("--steps", type=int, default=1000)
    g.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--device")
    e = sub.add_parser("evaluate")
    e.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_dataset(args.dataset, args.output, args.parquet_dir, args.limit)
    elif args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
