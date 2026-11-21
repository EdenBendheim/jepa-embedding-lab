"""Read global and ordered quadrant context means under matched image views."""

from jepa_lab.analysis.spatial_embeddings import paired_embeddings

from experiments.multiblock_patch_control import shuffle_patches


def encoder_views(model, images, indices, *, patch_seed):
    """Extract both readouts from a single context pass per view."""
    shuffled = shuffle_patches(images, indices, seed=patch_seed)
    return {
        view: {name: features[name] for name in ('mean', 'quadrants')}
        for view, features in (
            ('original', paired_embeddings(model, images, indices)),
            ('patch_shuffle', paired_embeddings(model, shuffled, indices)),
        )
    }
