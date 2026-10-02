"""Bounded synthetic CPU training, collapse diagnostics, and resumable checkpoints."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from .masking import block_mask
from .model import JEPA, ModelConfig


def synthetic_batch(config: ModelConfig, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    # Smooth spatial patterns with per-image slopes/noise; no real research data.
    rows = torch.linspace(-1, 1, config.rows).repeat_interleave(config.cols)
    cols = torch.linspace(-1, 1, config.cols).repeat(config.rows)
    coefficients = torch.randn(batch_size, 3, generator=generator)
    base = coefficients[:, :1]*rows + coefficients[:, 1:2]*cols + coefficients[:, 2:3]
    noise = torch.randn(batch_size, config.rows*config.cols, config.patch_dim, generator=generator)*0.05
    return base.unsqueeze(-1).expand(-1, -1, config.patch_dim) + noise


def embedding_stats(values: torch.Tensor) -> dict:
    samples = values.detach().reshape(-1, values.shape[-1]).float()
    centered = samples-samples.mean(0)
    covariance = centered.T @ centered/max(1, samples.shape[0]-1)
    off_diagonal = covariance-torch.diag_embed(covariance.diagonal())
    return dict(std_mean=float(samples.std(0, unbiased=False).mean()),
                covariance_off_diagonal_rms=float(off_diagonal.square().mean().sqrt()))


def train_smoke(*, steps: int = 10, batch_size: int = 4, seed: int = 29,
                learning_rate: float = 1e-3, momentum: float = 0.99,
                config: ModelConfig | None = None, resume: str | Path | None = None,
                checkpoint: str | Path | None = None) -> dict:
    if steps < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Steps, batch size, and learning rate must be positive")
    if not math.isfinite(momentum) or not 0 <= momentum <= 1:
        raise ValueError("EMA momentum must be in [0, 1]")
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        saved_config = ModelConfig(**saved["config"])
        if config is not None and config != saved_config:
            raise ValueError("Model config does not match the checkpoint")
        if (batch_size, seed, learning_rate, momentum) != (saved["batch_size"], saved["seed"],
                                                         saved["learning_rate"], saved["momentum"]):
            raise ValueError("Training settings must match the checkpoint for exact resume")
        config = saved_config
    else:
        saved = None
        config = config or ModelConfig()
    torch.manual_seed(seed)
    model = JEPA(config).train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    start = 0
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["data_rng"])
        torch.set_rng_state(saved["torch_rng"])
        start = saved["step"]
    history = []
    for step in range(start, start+steps):
        patches = synthetic_batch(config, batch_size, generator)
        masks = [block_mask(config.rows, config.cols, target_rows=min(2, config.rows),
                            target_cols=min(2, config.cols), seed=seed+step*batch_size+i)
                 for i in range(batch_size)]
        context = torch.tensor([mask.context for mask in masks])
        target = torch.tensor([mask.target for mask in masks])
        optimizer.zero_grad(set_to_none=True)
        predictions, targets = model(patches, context, target)
        loss = torch.nn.functional.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Training produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0,
                                      error_if_nonfinite=True)
        optimizer.step()
        model.update_target(momentum)
        history.append(dict(step=step+1, loss=float(loss.detach()),
                            prediction=embedding_stats(predictions), target=embedding_stats(targets)))
    if checkpoint:
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(format_version=1, config=asdict(config), model=model.state_dict(),
                        optimizer=optimizer.state_dict(), step=start+steps, seed=seed,
                        batch_size=batch_size, learning_rate=learning_rate, momentum=momentum,
                        data_rng=generator.get_state(), torch_rng=torch.get_rng_state()), path)
    return dict(dataset="synthetic spatial patterns", config=asdict(config), seed=seed,
                start_step=start, completed_step=start+steps, history=history,
                limitation="Smoke check only; no public-dataset or downstream-quality evaluation")


def main():
    parser = argparse.ArgumentParser(description="Run a small synthetic JEPA CPU training check")
    parser.add_argument("--steps", type=int, default=10, help="Additional steps, including when resuming")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", help="Optional JSON metrics file (use ignored runs/)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        result = train_smoke(steps=args.steps, batch_size=args.batch_size, seed=args.seed,
                             learning_rate=args.learning_rate, momentum=args.momentum,
                             resume=args.resume, checkpoint=args.checkpoint)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
