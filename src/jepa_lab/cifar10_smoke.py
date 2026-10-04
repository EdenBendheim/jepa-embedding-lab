"""A bounded public-data wiring check, not a downstream representation experiment."""

import argparse
import json
import math
from pathlib import Path
import random

import torch

from .cifar10 import CIFAR10Binary, make_manifest, patchify_rgb
from .masking import block_mask
from .model import JEPA, ModelConfig
from .train import embedding_stats


def train_cifar10_smoke(root: str | Path, manifest_path: str | Path, *, steps: int = 3,
                       batch_size: int = 4, learning_rate: float = 1e-3) -> dict:
    if type(steps) is not int or not 1 <= steps <= 50 or type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("Smoke checks allow 1-50 steps and a batch size of 1-32")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    manifest = json.loads(Path(manifest_path).read_text())
    dataset = CIFAR10Binary(root)
    test = CIFAR10Binary(root, train=False)
    # Rebuild from current batch hashes; stale/wrong-data splits must not silently run.
    if manifest != make_manifest(dataset, test, seed=manifest["seed"]):
        raise ValueError("Manifest does not match the current dataset files or split protocol")
    seed = manifest["seed"]
    indices = manifest["partitions"]["train"]["indices"]
    generator = random.Random(seed)
    torch.manual_seed(seed)
    config = ModelConfig(patch_dim=48)
    model = JEPA(config).train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    history = []
    for step in range(steps):
        chosen = generator.sample(indices, batch_size)
        # Class labels exist only for later probes. They are not used in this objective.
        images = torch.stack([dataset[index][0] for index in chosen])
        patches = patchify_rgb(images)
        masks = [block_mask(8, 8, target_rows=2, target_cols=2, seed=seed+step*batch_size+i)
                 for i in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        predictions, targets = model(patches, [mask.context for mask in masks], [mask.target for mask in masks])
        loss = torch.nn.functional.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Public-data smoke check produced non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0,
                                      error_if_nonfinite=True)
        optimizer.step()
        model.update_target(0.99)
        history.append(dict(step=step+1, train_indices=chosen, loss=float(loss.detach()),
                            prediction=embedding_stats(predictions), target=embedding_stats(targets)))
    return dict(dataset="CIFAR-10", manifest_sha256=manifest["manifest_sha256"], seed=seed,
                steps=steps, batch_size=batch_size, history=history,
                limitation="Bounded wiring check only; no downstream accuracy, long-run checkpoint, or useful-embedding claim")


def main():
    parser = argparse.ArgumentParser(description="Check RGB JEPA training on a few official training images")
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Optional JSON diagnostics; use ignored runs/")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        report = train_cifar10_smoke(args.root, args.manifest, steps=args.steps, batch_size=args.batch_size)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
