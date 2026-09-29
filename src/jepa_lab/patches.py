from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class Patch:
    row: int
    col: int
    values: tuple[float, ...]


def patchify(image: Sequence[Sequence[float]], patch_size: int) -> tuple[Patch, ...]:
    """Flatten a rectangular, single-channel image into non-overlapping patches.

    Incomplete edge patches are rejected instead of silently dropping observations.
    Multi-channel/temporal tensor support belongs to the later data adapter.
    """
    if patch_size < 1 or not isinstance(patch_size, int):
        raise ValueError("patch_size must be a positive integer")
    if not image or not image[0]:
        raise ValueError("Image must be nonempty")
    height, width = len(image), len(image[0])
    if any(len(row) != width for row in image):
        raise ValueError("Image must be rectangular")
    if height % patch_size or width % patch_size:
        raise ValueError("Image dimensions must be divisible by patch_size")
    if any(not math.isfinite(float(value)) for row in image for value in row):
        raise ValueError("Image values must be finite")
    patches = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            values = tuple(float(image[r][c]) for r in range(y, y + patch_size) for c in range(x, x + patch_size))
            patches.append(Patch(y // patch_size, x // patch_size, values))
    return tuple(patches)
