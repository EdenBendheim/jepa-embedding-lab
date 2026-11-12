"""Verified CIFAR-10 training reads and deterministic sample-journal replay."""

import json
from pathlib import Path
import random

import torch

from ..cifar10 import CIFAR10Binary, make_manifest


class CifarTrainingData:
    def __init__(self, root, manifest_path):
        manifest = json.loads(Path(manifest_path).read_text())
        if not isinstance(manifest, dict) or type(manifest.get('seed')) is not int:
            raise ValueError('Invalid CIFAR-10 manifest')
        self._dataset = CIFAR10Binary(root)
        test_metadata = CIFAR10Binary(root, train=False)
        if manifest != make_manifest(self._dataset, test_metadata, seed=manifest['seed']):
            raise ValueError('CIFAR-10 manifest differs from the local dataset or split protocol')
        self.manifest = manifest
        self.labels = tuple(self._dataset.labels)
        self.indices = tuple(manifest['partitions']['train']['indices'])
        self._training_indices = frozenset(self.indices)

    def read(self, indices):
        indices = tuple(indices)
        if (not indices or any(type(index) is not int or index not in self._training_indices for index in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('Choose distinct indices from the fixed CIFAR-10 training split')
        rows = [self._dataset[index] for index in indices]
        if any(label != self.labels[index] for index, (_, label) in zip(indices, rows, strict=True)):
            raise ValueError('CIFAR-10 training labels changed after manifest verification')
        return torch.stack([image for image, _ in rows]), torch.tensor([label for _, label in rows])

    def replay_sampling(self, history, *, seed, batch_size):
        """Rebuild the historical Python sampler from observed rows without image reads."""
        generator = random.Random(seed)
        for row in history:
            observed = row.get('train_indices') if isinstance(row, dict) else None
            expected = generator.sample(self.indices, batch_size)
            if (not isinstance(observed, list) or any(type(i) is not int for i in observed)
                    or observed != expected):
                raise ValueError('Checkpoint training image order differs from its deterministic schedule')
        return generator
