import math

import torch
from torch import nn
from torch.nn import functional as F
from diffusers import UNet2DConditionModel
from transformers import AutoConfig, AutoModel


def positions(length, dim, device, dtype):
    p = torch.arange(length, device=device).float()[:, None]
    f = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * (-math.log(10000) / dim)
    )
    return torch.stack(((p * f).sin(), (p * f).cos()), -1).flatten(1)[:, :dim].to(dtype)


class Attention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim or dim, dim, bias=False)
        self.v = nn.Linear(context_dim or dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context

        def split(t):
            return t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(context)), split(self.v(context))
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape_as(x))


class CCAM(nn.Module):
    def __init__(self, dim, text_dim, heads=8):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(dim) for _ in range(3)]
        self.sa = Attention(dim, heads=heads)
        self.character = Attention(dim, text_dim, heads)
        self.text_projection = nn.Linear(text_dim, dim)
        self.relation_projection = nn.Linear(dim, dim)
        self.context = Attention(dim, heads=heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        added_cond_kwargs=None,
    ):
        x = hidden_states
        pos = positions(x.shape[1], x.shape[2], x.device, x.dtype)[None].expand_as(x)
        x = x + self.sa(self.norm1(x) + pos)
        visual = self.norm2(x)
        char = self.character(visual, encoder_hidden_states, encoder_attention_mask)

        relation = F.scaled_dot_product_attention(
            visual[:, None], visual[:, None], pos[:, None]
        )[:, 0]
        query = self.relation_projection(relation)
        memory = torch.cat((visual, self.text_projection(encoder_hidden_states)), dim=1)
        mask = encoder_attention_mask
        if mask is not None:
            visual_mask = torch.zeros(
                (*mask.shape[:-1], visual.shape[1]), device=x.device, dtype=mask.dtype
            )

            mask = torch.cat((visual_mask, mask), dim=-1)
        x = x + char + self.context(query, memory, mask)
        return x + self.ff(self.norm3(x))


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class VisualEncoder(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(channels, dim // 2, 3, stride=2, padding=1),
            Residual(dim // 2),
            nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1),
            Residual(dim),
        )
        self.map_to_sequence = nn.Conv2d(dim, dim, 3, padding=1)
        self.bilstm = nn.LSTM(dim, dim // 2, bidirectional=True, batch_first=True)
        self.refine = nn.Linear(2 * dim, dim)

    def forward(self, images):
        v = self.map_to_sequence(self.resnet(images)).mean(2).transpose(1, 2)
        h, _ = self.bilstm(v)
        return self.refine(torch.cat((v, h), -1))


def alignment_loss(c, text, mask):
    similarity = F.normalize(c.float(), dim=-1) @ F.normalize(
        text.float(), dim=-1
    ).transpose(1, 2)
    valid = mask.bool()
    diagonal = similarity.diagonal(dim1=1, dim2=2)
    off_mask = valid[:, :, None] & valid[:, None, :]
    off_mask = (
        off_mask & ~torch.eye(c.shape[1], device=c.device, dtype=torch.bool)[None]
    )
    count = valid.sum(1)
    negatives = (similarity * off_mask).sum(-1) / (count - 1).clamp_min(1)[:, None]
    return (((1 - diagonal + negatives) * valid).sum(1) / count.clamp_min(1)).mean()


class FSACDM(nn.Module):
    def __init__(self, config, pretrained=True):
        super().__init__()
        self.config = config
        if pretrained:
            self.text_encoder = AutoModel.from_pretrained(config["encoder"])
        else:
            text_config = dict(config["text_config"])
            model_type = text_config.pop("model_type")
            self.text_encoder = AutoModel.from_config(
                AutoConfig.for_model(model_type, **text_config)
            )
        config["text_config"] = self.text_encoder.config.to_dict()
        text_dim = self.text_encoder.config.hidden_size
        self.text_encoder.requires_grad_(config.get("train_text_encoder", True))
        if config.get("gradient_checkpointing") and hasattr(
            self.text_encoder, "gradient_checkpointing_enable"
        ):
            self.text_encoder.gradient_checkpointing_enable()
            self.text_encoder.config.use_cache = False
        widths = config["block_out_channels"]

        n = len(widths)
        down = ["DownBlock2D"] * (n - 4) + ["CrossAttnDownBlock2D"] * 4
        up = ["CrossAttnUpBlock2D"] * 4 + ["UpBlock2D"] * (n - 4)
        self.unet = UNet2DConditionModel(
            sample_size=tuple(config["image_size"]),
            in_channels=config["channels"],
            out_channels=config["channels"],
            block_out_channels=tuple(widths),
            layers_per_block=1,
            down_block_types=tuple(down),
            up_block_types=tuple(up),
            cross_attention_dim=text_dim,
            attention_head_dim=8,
            norm_num_groups=8,
        )
        transformers = [
            (name, module)
            for name, module in self.unet.named_modules()
            if module.__class__.__name__ == "BasicTransformerBlock"
        ]
        assert len(transformers) == 13, len(transformers)
        if config.get("use_ccam", True):
            for index in (0, 3, 6, 9, 12):
                name, old = transformers[index]
                parent_name, key = name.rsplit(".", 1)
                self.unet.get_submodule(parent_name)._modules[key] = CCAM(
                    old.norm1.normalized_shape[0], text_dim
                )
        if config.get("gradient_checkpointing"):
            self.unet.enable_gradient_checkpointing()
        d = config["alignment_dim"]
        self.visual_encoder = VisualEncoder(config["channels"], d)
        self.text_projection = nn.Linear(text_dim, d)
        self.align_attention = Attention(d)

    def encode(self, ids, mask):
        if not self.config.get("train_text_encoder", True):
            self.text_encoder.eval()
        text = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return text * mask[..., None]

    def forward(self, noisy, timesteps, ids, mask, clean=None):
        text = self.encode(ids, mask)
        if len(noisy) % len(ids):
            raise ValueError("Diffusion branches must be whole batches")
        branches = len(noisy) // len(ids)
        pred = self.unet(
            noisy,
            timesteps,
            encoder_hidden_states=text.repeat(branches, 1, 1),
            encoder_attention_mask=mask.repeat(branches, 1),
        ).sample
        result = {"noise": pred}
        if clean is not None and self.config.get("use_fsa", True):
            visual = self.visual_encoder(clean)
            projected = self.text_projection(text)
            aligned = self.align_attention(projected, visual)
            result["alignment"] = alignment_loss(aligned, projected, mask)
        else:
            result["alignment"] = pred.new_zeros(())
        return result
