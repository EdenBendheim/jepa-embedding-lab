"""Compact JEPA model for the patch/mask contract; not an I-JEPA reproduction."""

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    rows: int = 8
    cols: int = 8
    patch_dim: int = 4
    embedding_dim: int = 32
    heads: int = 4
    encoder_depth: int = 2
    predictor_depth: int = 1

    def __post_init__(self):
        if any(not isinstance(v, int) or v < 1 for v in vars(self).values()):
            raise ValueError("Model dimensions must be positive integers")
        if self.embedding_dim % self.heads:
            raise ValueError("embedding_dim must be divisible by heads")


def transformer(config: ModelConfig, depth: int) -> nn.Module:
    layer = nn.TransformerEncoderLayer(config.embedding_dim, config.heads,
                                       dim_feedforward=config.embedding_dim * 2,
                                       dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(config.embedding_dim),
                                 enable_nested_tensor=False)


def index_tensor(indices, *, batch: int, total: int, device) -> torch.Tensor:
    result = torch.as_tensor(indices, device=device)
    if result.dtype not in (torch.int32, torch.int64):
        raise ValueError("Patch indices must be integers")
    result = result.long()
    if result.ndim == 1:
        result = result.unsqueeze(0).expand(batch, -1)
    if result.ndim != 2 or result.shape[0] != batch or result.shape[1] < 1:
        raise ValueError("Indices must be nonempty [count] or [batch, count]")
    if (result < 0).any() or (result >= total).any():
        raise ValueError("Patch index is outside the configured grid")
    ordered = result.sort(dim=1).values
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        raise ValueError("Patch indices must be unique within each sample")
    return result


def gather(sequence: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return sequence.gather(1, indices.unsqueeze(-1).expand(-1, -1, sequence.shape[-1]))


class PatchEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.patch_dim, config.embedding_dim)
        self.positions = nn.Parameter(torch.empty(config.rows * config.cols, config.embedding_dim))
        nn.init.trunc_normal_(self.positions, std=0.02)
        self.blocks = transformer(config, config.encoder_depth)

    def forward(self, patches: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # Gather raw visible patches BEFORE encoding: hidden pixels never reach context attention.
        visible = gather(patches, indices)
        return self.blocks(self.projection(visible) + self.positions[indices])


class JEPA(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.context_encoder = PatchEncoder(config)
        self.target_encoder = deepcopy(self.context_encoder).requires_grad_(False)
        self.predictor_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.predictor_positions = nn.Parameter(torch.empty(config.rows * config.cols, config.embedding_dim))
        self.mask_token = nn.Parameter(torch.empty(1, 1, config.embedding_dim))
        nn.init.trunc_normal_(self.predictor_positions, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.predictor = transformer(config, config.predictor_depth)
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.target_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(self, patches: torch.Tensor, context, target) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        if patches.ndim != 3 or patches.shape[0] < 1 or patches.shape[1:] != (config.rows*config.cols, config.patch_dim):
            raise ValueError("Patches must have shape [batch, grid_size, patch_dim]")
        if not patches.is_floating_point() or not torch.isfinite(patches).all():
            raise ValueError("Patch values must be finite floating-point numbers")
        options = dict(batch=patches.shape[0], total=patches.shape[1], device=patches.device)
        context = index_tensor(context, **options)
        target = index_tensor(target, **options)
        if (context.unsqueeze(2) == target.unsqueeze(1)).any():
            raise ValueError("Context and target indices must be disjoint")
        encoded = self.context_encoder(patches, context)
        visible = self.predictor_projection(encoded) + self.predictor_positions[context]
        tokens = self.mask_token.expand(patches.shape[0], target.shape[1], -1) + self.predictor_positions[target]
        predictions = self.output(self.predictor(torch.cat((visible, tokens), dim=1))[:, -target.shape[1]:])
        with torch.no_grad():
            full = torch.arange(patches.shape[1], device=patches.device).expand(patches.shape[0], -1)
            targets = gather(self.target_encoder(patches, full), target)
            targets = F.layer_norm(targets, (config.embedding_dim,))
        return predictions, targets

    def loss(self, patches: torch.Tensor, context, target) -> torch.Tensor:
        predictions, targets = self(patches, context, target)
        return F.smooth_l1_loss(predictions, targets)

    @torch.no_grad()
    def update_target(self, momentum: float = 0.99):
        if not math.isfinite(momentum) or not 0 <= momentum <= 1:
            raise ValueError("EMA momentum must be in [0, 1]")
        for target, context in zip(self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True):
            target.mul_(momentum).add_(context, alpha=1-momentum)
        for target, context in zip(self.target_encoder.buffers(), self.context_encoder.buffers(), strict=True):
            target.copy_(context)
