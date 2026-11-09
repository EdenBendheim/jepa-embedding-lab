"""Seeded disjoint target tiles with one context outside the complete target union."""

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class MultiBlockMask:
    rows: int
    cols: int
    context: tuple[int, ...]
    blocks: tuple[tuple[int, ...], ...]
    unused: tuple[int, ...]

    def __post_init__(self):
        if (any(type(v) is not int or v < 1 for v in (self.rows, self.cols))
                or self.rows * self.cols > 256 or not isinstance(self.blocks, tuple)
                or not 1 <= len(self.blocks) <= 4):
            raise ValueError('Use a grid of 1-256 patches and 1-4 target blocks')
        groups = (self.context, *self.blocks, self.unused)
        if (any(not isinstance(group, tuple) for group in groups)
                or not self.context or any(not block for block in self.blocks)
                or len({len(block) for block in self.blocks}) != 1):
            raise ValueError('Use nonempty context and equally sized target-block tuples')
        flattened = [index for group in groups for index in group]
        if (any(type(index) is not int or not 0 <= index < self.rows*self.cols for index in flattened)
                or len(flattened) != len(set(flattened))
                or set(flattened) != set(range(self.rows*self.cols))):
            raise ValueError('Context, target blocks and unused positions must partition the complete grid')
        for block in self.blocks:
            rows, cols = {i // self.cols for i in block}, {i % self.cols for i in block}
            if (rows != set(range(min(rows), max(rows)+1)) or cols != set(range(min(cols), max(cols)+1))
                    or set(block) != {r*self.cols+c for r in rows for c in cols}):
                raise ValueError('Each target block must be a complete contiguous rectangle')


def multi_block_mask(rows, cols, *, blocks=2, target_rows=2, target_cols=2, context_fraction=.75, seed=29):
    """Sample distinct nonoverlapping tiles; do not retry arbitrary placements.

    Grid dimensions must be divisible by tile dimensions. This defined-here
    distribution is deliberately bounded, not the official I-JEPA sampler.
    """
    if (any(type(v) is not int or v < 1 for v in (rows, cols, blocks, target_rows, target_cols))
            or rows*cols > 256 or blocks > 4 or rows % target_rows or cols % target_cols
            or type(context_fraction) not in (int, float) or not math.isfinite(context_fraction)
            or not 0 < context_fraction <= 1 or type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('Use a divisible grid, 1-4 target tiles, context fraction (0,1] and integer seed')
    origins = [(r, c) for r in range(0, rows, target_rows) for c in range(0, cols, target_cols)]
    if blocks >= len(origins):
        raise ValueError('Target tiles must leave at least one tile available for context')
    rng = random.Random(seed)
    selected = sorted(rng.sample(origins, blocks))
    targets = tuple(tuple(r*cols+c for r in range(top, top+target_rows)
                          for c in range(left, left+target_cols)) for top, left in selected)
    target_union = {index for block in targets for index in block}
    available = [index for index in range(rows*cols) if index not in target_union]
    count = max(1, math.floor(len(available)*context_fraction))
    context = tuple(sorted(rng.sample(available, count)))
    return MultiBlockMask(rows, cols, context, targets, tuple(i for i in available if i not in context))
