"""Bounded, resumable CPU JEPA training tied to a verified CIFAR-10 manifest."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import tempfile

import torch

from .cifar10 import CIFAR10Binary, make_manifest, patchify_rgb
from .masking import block_mask
from .model import JEPA, ModelConfig
from .train import embedding_stats


def atomic_checkpoint(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def train_cifar10(root: str | Path, manifest_path: str | Path, *, steps: int = 20,
                  batch_size: int = 4, learning_rate: float = 1e-3, momentum: float = 0.99,
                  seed: int | None = None, config: ModelConfig | None = None,
                  checkpoint: str | Path | None = None, resume: str | Path | None = None) -> dict:
    if type(steps) is not int or not 1 <= steps <= 1000 or type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("Local runs allow 1-1000 additional steps and a batch size of 1-32")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(momentum) or not 0 <= momentum <= 1:
        raise ValueError("Learning rate must be finite and positive; momentum must be in [0, 1]")
    config = config if config is not None else ModelConfig(patch_dim=48)
    if (config.rows, config.cols, config.patch_dim) != (8, 8, 48):
        raise ValueError("CIFAR-10 protocol requires an 8x8 grid of 4x4 RGB patches (patch_dim=48)")
    manifest = json.loads(Path(manifest_path).read_text())
    dataset, test = CIFAR10Binary(root), CIFAR10Binary(root, train=False)
    if manifest != make_manifest(dataset, test, seed=manifest["seed"]):
        raise ValueError("Manifest does not match the current dataset files or split protocol")
    seed = manifest["seed"] if seed is None else seed
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Seed must be an integer in [0, 2**63)")
    settings = dict(seed=seed, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum)
    runtime = dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads())
    identity = dict(format_version=1, dataset="CIFAR-10", manifest_sha256=manifest["manifest_sha256"],
                    config=asdict(config), settings=settings, runtime=runtime)
    saved = None
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Checkpoint manifest, model, settings, or runtime does not match this run")
        if type(saved.get("step")) is not int or saved["step"] < 1:
            raise ValueError("Checkpoint must contain a positive completed step count")
    generator = random.Random(seed)
    torch.manual_seed(seed)
    model = JEPA(config).train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    start = 0
    if saved is not None:
        try:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            generator.setstate(saved["sample_rng"])
            torch.set_rng_state(saved["torch_rng"])
            start = saved["step"]
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            raise ValueError("Checkpoint training state is incomplete or invalid") from exc
    indices = manifest["partitions"]["train"]["indices"]
    history = []
    for step in range(start, start+steps):
        chosen = generator.sample(indices, batch_size)
        # Labels define the fixed split and future probes, never the JEPA objective.
        images = torch.stack([dataset[index][0] for index in chosen])
        patches = patchify_rgb(images)
        masks = [block_mask(8, 8, target_rows=2, target_cols=2, seed=seed+step*batch_size+i)
                 for i in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        predictions, targets = model(patches, [mask.context for mask in masks], [mask.target for mask in masks])
        loss = torch.nn.functional.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("CIFAR-10 training produced non-finite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0,
                                             error_if_nonfinite=True)
        optimizer.step()
        model.update_target(momentum)
        history.append(dict(step=step+1, train_indices=chosen, loss=float(loss.detach()),
                            gradient_norm=float(norm), prediction=embedding_stats(predictions),
                            target=embedding_stats(targets)))
    if checkpoint:
        atomic_checkpoint(Path(checkpoint), identity | dict(step=start+steps, model=model.state_dict(),
                          optimizer=optimizer.state_dict(), sample_rng=generator.getstate(),
                          torch_rng=torch.get_rng_state()))
    return identity | dict(start_step=start, completed_steps=start+steps, additional_steps=steps, history=history,
                           limitation="Bounded training pilot; downstream accuracy and useful embeddings remain unmeasured")


def main():
    parser = argparse.ArgumentParser(description="Run and resume local CPU JEPA training on fixed CIFAR-10 splits")
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20, help="Additional steps, including when resuming (1-1000)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--seed", type=int, help="Defaults to the fixed manifest seed")
    parser.add_argument("--checkpoint", type=Path, help="Save training state atomically beneath ignored checkpoints/")
    parser.add_argument("--resume", type=Path, help="Restore a matching checkpoint; never unpickle untrusted files")
    parser.add_argument("--output", type=Path, help="Write this invocation's diagnostics beneath ignored runs/")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        report = train_cifar10(args.root, args.manifest, steps=args.steps, batch_size=args.batch_size,
                              learning_rate=args.learning_rate, momentum=args.momentum, seed=args.seed,
                              checkpoint=args.checkpoint, resume=args.resume)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
