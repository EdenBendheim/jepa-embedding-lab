import unittest

import torch

from jepa_lab.diagnostics import paired_diagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_paired_recoveries_regressions_and_confusion_changes(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        before = torch.tensor([0, 1, 1, 2, 0, 0])
        after = torch.tensor([0, 0, 2, 1, 1, 0])
        result = paired_diagnostics(labels, before, after)
        self.assertEqual((1, 2, 1, 2), tuple(result[key] for key in
            ("both_correct", "recovered", "regressed", "both_wrong")))
        self.assertEqual(4, result["predictions_changed"])
        self.assertAlmostEqual(1/6, result["accuracy_delta"])
        self.assertEqual(1, result["per_class"][0]["recovered"])
        self.assertEqual(1, result["per_class"][1]["regressed"])
        self.assertIsNone(result["per_class"][9]["accuracy_delta"])
        self.assertEqual([1, -1, 0], result["confusion_delta"][0][:3])
        self.assertEqual(0, sum(sum(row) for row in result["confusion_delta"]))
        reverse = paired_diagnostics(labels, after, before)
        self.assertEqual(result["recovered"], reverse["regressed"])
        self.assertEqual(result["regressed"], reverse["recovered"])
        self.assertEqual(-result["accuracy_delta"], reverse["accuracy_delta"])

    def test_identical_predictions_and_invalid_vectors(self):
        labels = torch.arange(10)
        result = paired_diagnostics(labels, labels, labels)
        self.assertEqual(10, result["both_correct"])
        self.assertEqual(0, result["predictions_changed"])
        for invalid in (torch.tensor([]), labels.float(), labels[:3], torch.tensor([-1]*10),
                        torch.tensor([10]*10), labels.reshape(2, 5), torch.zeros(1001, dtype=torch.long)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                paired_diagnostics(labels, labels, invalid)
        with self.assertRaises(ValueError):
            paired_diagnostics(torch.empty(0, dtype=torch.long), labels, labels)
