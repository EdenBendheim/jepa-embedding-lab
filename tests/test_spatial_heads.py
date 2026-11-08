import tempfile
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from jepa_lab.analysis.spatial_evaluation import crossfit_ridge, stratified_folds
from jepa_lab.analysis.spatial_heads import FoldArtifacts
from jepa_lab.ridge import RidgeConfig


class SpatialHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.labels = torch.arange(10).repeat(5)
        self.values = torch.nn.functional.one_hot(self.labels, 10).float()
        self.inputs = SimpleNamespace(labels=self.labels, indices=tuple(range(50)),
            identity={'source_identity': {'manifest_sha256': 'a'*64}})

    def artifacts(self, *, reuse_only=False):
        with patch('jepa_lab.analysis.spatial_heads.validate_features'):
            return FoldArtifacts(Path(self.folder.name), self.inputs, {'example': self.values}, reuse_only=reuse_only)

    def test_five_saved_heads_reproduce_scores_without_refitting_and_bind_row_order(self):
        store = self.artifacts()
        provider = lambda fold, x, y, c: store('example', fold, x, y, c)
        assignments = stratified_folds(self.labels)
        first = crossfit_ridge(self.values, self.labels, assignments, fold_provider=provider)
        self.assertEqual(5, store.fitted)
        before = {p: p.read_bytes() for p in Path(self.folder.name).iterdir()}
        replay = self.artifacts(reuse_only=True)
        with patch('jepa_lab.analysis.spatial_heads.fit_ridge', side_effect=AssertionError('refit')):
            second = crossfit_ridge(self.values, self.labels, assignments,
                fold_provider=lambda fold, x, y, c: replay('example', fold, x, y, c))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertEqual(first[1], second[1])
        self.assertEqual(store.records, replay.records)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        train, scoring = store.pair('example', 0)
        self.assertFalse(set(train.indices) & set(scoring.indices))
        self.inputs.indices = tuple(reversed(self.inputs.indices))
        wrong = self.artifacts(reuse_only=True)
        with self.assertRaisesRegex(ValueError, 'sources|identity'):
            wrong('example', 0, train.features, train.labels, RidgeConfig(.1))

    def test_missing_head_never_refits_and_failed_save_leaves_no_artifact(self):
        store = self.artifacts(reuse_only=True)
        train, _ = store.pair('example', 0)
        with patch('jepa_lab.analysis.spatial_heads.fit_ridge', side_effect=AssertionError('refit')), \
                self.assertRaisesRegex(ValueError, 'missing'):
            store('example', 0, train.features, train.labels, RidgeConfig(.1))
        with patch('jepa_lab.analysis.spatial_heads.save_probe', side_effect=OSError('disk error')), \
                self.assertRaisesRegex(OSError, 'disk error'):
            self.artifacts()('example', 0, train.features, train.labels, RidgeConfig(.1))
        self.assertEqual([], list(Path(self.folder.name).iterdir()))
