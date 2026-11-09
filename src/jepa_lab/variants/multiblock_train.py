"""Bounded synthetic multi-block training with exact, recipe-bound resume."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform

import torch
from torch.nn import functional as F

from ..cifar10_train import atomic_checkpoint
from ..model import JEPA, ModelConfig
from ..train import embedding_stats, synthetic_batch
from .multiblock_masks import multi_block_mask
from .multiblock_model import multi_block_forward


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sources():
    root = Path(__file__).resolve().parent.parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py'))}


def _masks(config, settings, step):
    return [multi_block_mask(config.rows, config.cols, **settings['mask'],
            seed=(settings['seed'] + step*settings['batch_size'] + image) % (2**63))
            for image in range(settings['batch_size'])]


def _mask_digest(masks):
    return _digest([dict(context=mask.context, blocks=mask.blocks, unused=mask.unused) for mask in masks])


def train_multiblock(*, steps=10, batch_size=4, seed=29, blocks=2, target_rows=2, target_cols=2,
                     context_fraction=.75, learning_rate=.001, momentum=.99, config=None,
                     checkpoint=None, resume=None):
    if (type(steps) is not int or not 1 <= steps <= 100 or type(batch_size) is not int
            or not 1 <= batch_size <= 8 or type(seed) is not int or not 0 <= seed < 2**63
            or type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= .1 or type(momentum) not in (int, float)
            or not math.isfinite(momentum) or not 0 <= momentum <= 1 or torch.get_num_threads() != 1):
        raise ValueError('Use 1-100 additional steps, batch 1-8, valid seed/rates and one CPU thread')
    config = ModelConfig() if config is None else config
    if (not isinstance(config, ModelConfig) or config.rows*config.cols > 64 or config.patch_dim > 48
            or not 8 <= config.embedding_dim <= 64 or config.heads > 8
            or config.encoder_depth > 2 or config.predictor_depth > 2):
        raise ValueError('Synthetic variant allows <=64 patches, patch width <=48 and embedding width 8-64, depths <=2')
    settings = dict(seed=seed, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum,
        mask=dict(blocks=blocks, target_rows=target_rows, target_cols=target_cols, context_fraction=context_fraction),
        sampler='disjoint_target_tiles_v1', optimizer='AdamW, weight_decay=.01', gradient_clip_norm=1.,
        loss='mean SmoothL1 over images, equal-sized blocks, tokens and channels')
    _masks(config, settings, 0)  # Validate the recipe before model allocation or checkpoint writes.
    identity = dict(format_version=1, kind='synthetic_multiblock_v1', config=asdict(config), settings=settings,
        runtime=dict(torch=str(torch.__version__), python=platform.python_version(), cpu_threads=1), code_sha256=_sources())
    start, history, saved = 0, [], None
    if checkpoint is not None and Path(checkpoint).exists() and (resume is None or Path(checkpoint).resolve() != Path(resume).resolve()):
        raise ValueError('Checkpoint destination already exists; resume it or choose a new path')
    if resume is not None:
        path = Path(resume)
        if path.stat().st_size > 64*1024*1024:
            raise ValueError('Synthetic checkpoint exceeds 64 MiB')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        fields = set(identity) | {'step', 'model', 'optimizer', 'data_rng', 'torch_rng', 'history'}
        if (not isinstance(saved, dict) or set(saved) != fields
                or any(saved[key] != value for key, value in identity.items())
                or type(saved['step']) is not int or not 1 <= saved['step'] <= 1000
                or not isinstance(saved['history'], list) or len(saved['history']) != saved['step']):
            raise ValueError('Checkpoint variant, model, recipe, runtime, source or journal range differs')
        for step, row in enumerate(saved['history'], 1):
            if (not isinstance(row, dict) or type(row.get('step')) is not int or row['step'] != step
                    or row.get('mask_sha256') != _mask_digest(_masks(config, settings, step-1))):
                raise ValueError('Checkpoint journal differs from its deterministic mask schedule')
        json.dumps(saved['history'], allow_nan=False)
        start, history = saved['step'], list(saved['history'])
    if start + steps > 1000:
        raise ValueError('Synthetic checkpoint trajectory is bounded to 1000 total steps')
    initial_count = len(history)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = JEPA(config).train()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                     lr=learning_rate, weight_decay=.01)
        generator = torch.Generator().manual_seed(seed)
        if saved is not None:
            try:
                if saved['optimizer']['param_groups'] != optimizer.state_dict()['param_groups']:
                    raise ValueError('Saved optimizer settings differ from the declared recipe')
                model.load_state_dict(saved['model'])
                optimizer.load_state_dict(saved['optimizer'])
                generator.set_state(saved['data_rng'])
                torch.set_rng_state(saved['torch_rng'])
            except (ValueError, TypeError, RuntimeError, KeyError) as exc:
                raise ValueError('Invalid multi-block model, optimizer or RNG state') from exc
        for step in range(start, start+steps):
            patches = synthetic_batch(config, batch_size, generator)
            masks = _masks(config, settings, step)
            optimizer.zero_grad(set_to_none=True)
            predictions, targets = multi_block_forward(model, patches, masks)
            loss = F.smooth_l1_loss(predictions, targets)
            if not torch.isfinite(loss):
                raise ValueError('Multi-block loss is not finite')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                                  1., error_if_nonfinite=True)
            optimizer.step()
            model.update_target(momentum)
            block_losses = F.smooth_l1_loss(predictions.detach(), targets, reduction='none').mean(dim=(0, 2, 3))
            row = dict(step=step+1, loss=float(loss.detach()), block_losses=block_losses.tolist(),
                gradient_norm=float(norm), mask_sha256=_mask_digest(masks), context_tokens=len(masks[0].context),
                target_tokens_per_image=sum(len(block) for block in masks[0].blocks),
                predictor_passes=blocks, prediction=embedding_stats(predictions), target=embedding_stats(targets))
            json.dumps(row, allow_nan=False)
            history.append(row)
        if checkpoint is not None:
            atomic_checkpoint(Path(checkpoint), dict(identity, step=start+steps, model=model.state_dict(),
                optimizer=optimizer.state_dict(), data_rng=generator.get_state(),
                torch_rng=torch.get_rng_state(), history=history))
    return dict(dataset='synthetic spatial patterns', config=identity['config'], settings=settings,
        runtime=identity['runtime'], start_step=start, completed_step=start+steps, history=history[initial_count:],
        limitation='Synthetic execution/resume check only; no public-data training or embedding-quality result. '
                   'Multiple predictor passes change compute; equal steps do not imply equal budget.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=10, help='Additional optimizer updates')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--target-rows', type=int, default=2)
    parser.add_argument('--target-cols', type=int, default=2)
    parser.add_argument('--context-fraction', type=float, default=.75)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--momentum', type=float, default=.99)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path, help='Local aggregate metrics JSON, outside checkpoint destinations')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.output and any(args.output.resolve() == path.resolve() for path in (args.resume, args.checkpoint) if path):
            raise ValueError('Metrics output cannot overwrite a checkpoint')
        result = train_multiblock(**{k: v for k, v in vars(args).items() if k != 'output'})
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
