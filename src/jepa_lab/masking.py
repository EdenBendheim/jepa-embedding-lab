"""Reproducible rectangular target masks with separate visible context."""

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Mask:
    rows: int
    cols: int
    context: tuple[int, ...]
    target: tuple[int, ...]
    unused: tuple[int, ...]

    def render(self) -> str:
        context, target = set(self.context), set(self.target)
        return "\n".join(" ".join("T" if r * self.cols + c in target else
                                  "C" if r * self.cols + c in context else "."
                                  for c in range(self.cols)) for r in range(self.rows))


def block_mask(rows: int, cols: int, *, target_rows: int = 2, target_cols: int = 2,
               context_fraction: float = 0.75, seed: int = 0) -> Mask:
    """Choose one contiguous target block and sample context outside that block.

    Fraction is relative to the remaining non-target patches. This is a simple
    experiment contract, not the official I-JEPA multi-block mask distribution.
    """
    if any(not isinstance(value, int) or value < 1 for value in (rows, cols, target_rows, target_cols)):
        raise ValueError("Grid and target dimensions must be positive integers")
    if target_rows > rows or target_cols > cols or target_rows * target_cols >= rows * cols:
        raise ValueError("Target must fit in the grid and leave visible context")
    if not math.isfinite(context_fraction) or not 0 < context_fraction <= 1:
        raise ValueError("context_fraction must be in (0, 1]")
    rng = random.Random(seed)
    top, left = rng.randrange(rows - target_rows + 1), rng.randrange(cols - target_cols + 1)
    target = tuple(r * cols + c for r in range(top, top + target_rows) for c in range(left, left + target_cols))
    target_set = set(target)
    available = [i for i in range(rows * cols) if i not in target_set]
    count = max(1, math.floor(len(available) * context_fraction))
    context = tuple(sorted(rng.sample(available, count)))
    context_set = set(context)
    unused = tuple(i for i in available if i not in context_set)
    return Mask(rows, cols, context, target, unused)
