"""Controlled checkpoint-age comparisons with a single shared baseline pair."""

import re

import torch

from .compare import validate_comparison_inputs
from .features import _sha


def validate_progression(arms, candidates):
    """Validate every arm before fitting; matching settings do not prove ancestry."""
    if (not isinstance(arms, dict) or not 2 <= len(arms) <= 4
            or any(not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9-]{0,47}", name) is None for name in arms)):
        raise ValueError("Use 2-4 ordered checkpoint arms with distinct short lowercase names")
    first = next(iter(arms.values()))
    reference = validate_comparison_inputs(first, candidates)
    contract = {key: reference[0].metadata.get(key) for key in
        ("config", "training_settings", "encoder", "pooling", "normalization")}
    previous_step, hashes, identity = 0, set(), []
    for name, sets in arms.items():
        pair = validate_comparison_inputs(sets, candidates)
        metadata = pair[0].metadata
        step, fingerprint = metadata.get("completed_steps"), metadata.get("checkpoint_sha256")
        if (type(step) is not int or step <= previous_step or not _sha(fingerprint) or fingerprint in hashes
                or not isinstance(metadata.get("training_settings"), dict)
                or {key: metadata.get(key) for key in contract} != contract):
            raise ValueError("Checkpoint ages must increase with distinct weights and identical architecture/training settings")
        for item, expected in zip(pair, reference):
            if (item.indices != expected.indices or not torch.equal(item.labels, expected.labels)
                    or item.manifest_sha256 != expected.manifest_sha256
                    or item.selection_sha256 != expected.selection_sha256):
                raise ValueError("Checkpoint arms require identical ordered examples and labels")
        for baseline in ("random", "pixels"):
            for item, expected in zip(sets[baseline], first[baseline]):
                if (item.artifact_sha256 != expected.artifact_sha256 or item.metadata != expected.metadata
                        or not torch.equal(item.features, expected.features)):
                    raise ValueError("All checkpoint arms must reuse the exact same random/pixel artifacts")
        identity.append(dict(name=name, completed_steps=step, checkpoint_sha256=fingerprint))
        previous_step = step
        hashes.add(fingerprint)
    return identity
