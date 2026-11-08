"""Persist each cross-fitting head against its exact disjoint feature subsets."""

import os
from pathlib import Path
import tempfile

import torch

from ..features import FeatureSet
from ..probe_artifacts import load_probe, save_probe
from ..ridge import fit_ridge
from .color_audit import digest
from .spatial_cache import tensor_sha, validate_features
from .spatial_evaluation import stratified_folds


class FoldArtifacts:
    """Load or fit one fixed head; score replay never silently fits a missing head."""

    def __init__(self, directory, inputs, features, *, reuse_only=False):
        if type(reuse_only) is not bool:
            raise ValueError('Reuse-only must be boolean')
        validate_features(features, inputs)
        self.directory = Path(directory)
        self.inputs, self.features, self.reuse_only = inputs, features, reuse_only
        self.assignments = stratified_folds(inputs.labels)
        self.identity_sha = digest(inputs.identity)
        self.records = {}
        self.fitted = 0
        self.reused = 0

    def pair(self, key, fold):
        if key not in self.features or type(fold) is not int or not 0 <= fold < 5:
            raise ValueError('Use a declared spatial source and fold 0-4')
        values = self.features[key]
        metadata = dict(kind='crossfit subset of original encoder-training rows',
            source_identity_sha256=self.identity_sha, source=key, fold=fold,
            fold_assignments_sha256=digest(self.assignments.tolist()),
            full_features_sha256=tensor_sha(values), full_labels_sha256=tensor_sha(self.inputs.labels))
        selection_sha = digest(metadata)
        result = []
        for partition, selected in (('train', self.assignments != fold), ('validation', self.assignments == fold)):
            indices = tuple(index for index, keep in zip(self.inputs.indices, selected.tolist(), strict=True) if keep)
            features, labels = values[selected], self.inputs.labels[selected]
            artifact_sha = digest(dict(features=tensor_sha(features), labels=tensor_sha(labels), indices=indices))
            result.append(FeatureSet(features, labels, indices, partition,
                self.inputs.identity['source_identity']['manifest_sha256'], selection_sha,
                metadata, artifact_sha))
        return tuple(result)

    def path(self, key, fold):
        return self.directory / (digest(dict(identity=self.identity_sha, source=key, fold=fold)) + '.pt')

    def __call__(self, key, fold, features, labels, config):
        train, scoring = self.pair(key, fold)
        if not torch.equal(features, train.features) or not torch.equal(labels, train.labels):
            raise ValueError('Fold fitter rows differ from the bound feature subsets')
        path = self.path(key, fold)
        if not path.exists():
            if self.reuse_only:
                raise ValueError('Required saved spatial head is missing; replay cannot refit')
            probe = fit_ridge(features, labels, config)
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as handle:
                    temporary = Path(handle.name)
                save_probe(temporary, probe, train, scoring)
                try:
                    os.link(temporary, path)
                    self.fitted += 1
                except FileExistsError:
                    pass
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        else:
            self.reused += 1
        probe, report = load_probe(path, train, scoring)
        if probe.config != config:
            raise ValueError('Saved spatial head configuration differs from the fixed recipe')
        self.records[f'{key}/fold{fold}'] = dict(artifact_sha256=report['artifact_sha256'],
            state_sha256=report['state_sha256'], fitting_count=len(train.labels), scoring_count=len(scoring.labels))
        return probe
