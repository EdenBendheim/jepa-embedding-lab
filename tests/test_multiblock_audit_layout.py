import unittest
from unittest.mock import patch

import torch

from experiments import multiblock_audit_layout as audit
from jepa_lab.analysis.spatial_evaluation import stratified_folds


class LayoutTransferTests(unittest.TestCase):
    def test_shuffled_scores_reuse_heads_fit_on_original_rows(self):
        labels = torch.arange(30) % 10
        original = torch.nn.functional.one_hot(labels, 10).float()
        shuffled = torch.nn.functional.one_hot((labels+1) % 10, 10).float()
        folds = stratified_folds(labels, folds=3)
        with patch.object(audit, 'fit_ridge', wraps=audit.fit_ridge) as fit:
            report = audit.layout_transfer({'29/model/original': original,
                '29/model/shuffled': shuffled}, labels, folds, alpha=.1)
        self.assertEqual(3, fit.call_count)
        for fold, call in enumerate(fit.call_args_list):
            self.assertTrue(torch.equal(call.args[0], original[folds != fold]))
        counts = report['comparisons']['29/model']
        self.assertEqual((30, 0, 30),
                         (counts['baseline_correct'], counts['candidate_correct'], counts['regressed']))
