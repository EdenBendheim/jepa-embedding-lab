"""Fixed full-image token pooling rules, without fitting on held-out examples."""

import torch


POOLING = {
    "mean": "mean of all 64 patch tokens",
    "max": "maximum of all 64 patch tokens",
    "mean_std": "concatenated mean and population std of all 64 patch tokens",
}


def pooling_dimension(embedding_dim: int, mode: str) -> int:
    if type(mode) is not str or mode not in POOLING:
        raise ValueError("Pooling must be mean, max, or mean_std")
    if type(embedding_dim) is not int or not 1 <= embedding_dim <= 2048:
        raise ValueError("Embedding width must be an integer in [1, 2048]")
    return embedding_dim * (2 if mode == "mean_std" else 1)


def pooling_mode(description: str) -> str:
    for mode, value in POOLING.items():
        if description == value:
            return mode
    raise ValueError("Unrecognized full-image pooling contract")


@torch.inference_mode()
def pool_tokens(tokens: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Pool spatial tokens per image; mean_std doubles the classifier dimension."""
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[0] < 1
            or tokens.shape[1] != 64 or not tokens.is_floating_point()):
        raise ValueError("Pooling requires finite floating [batch, 64, width] tokens")
    if not torch.isfinite(tokens).all():
        raise ValueError("Pooling received non-finite encoder embeddings")
    pooling_dimension(tokens.shape[2], mode)
    if mode == "max":
        return tokens.amax(dim=1)
    average = tokens.mean(dim=1)
    if mode == "mean_std":
        return torch.cat((average, tokens.std(dim=1, unbiased=False)), dim=1)
    return average
