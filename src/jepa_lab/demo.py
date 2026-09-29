import argparse
import json

from .masking import block_mask
from .patches import patchify


def main():
    parser = argparse.ArgumentParser(description="Inspect reproducible JEPA context/target masks")
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--target-rows", type=int, default=2)
    parser.add_argument("--target-cols", type=int, default=2)
    parser.add_argument("--context-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    # Public demo has synthetic pixel intensities, no private research data.
    image = [[float((r * 3 + c * 5) % 17) / 16 for c in range(args.image_size)] for r in range(args.image_size)]
    try:
        patches = patchify(image, args.patch_size)
        grid = args.image_size // args.patch_size
        mask = block_mask(grid, grid, target_rows=args.target_rows, target_cols=args.target_cols,
                          context_fraction=args.context_fraction, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    print("C = visible context; T = prediction target; . = unused\n")
    print(mask.render())
    print(json.dumps(dict(seed=args.seed, patch_count=len(patches), patch_width=len(patches[0].values),
                          context_count=len(mask.context), target_count=len(mask.target),
                          context_indices=mask.context, target_indices=mask.target), indent=2))


if __name__ == "__main__":
    main()
