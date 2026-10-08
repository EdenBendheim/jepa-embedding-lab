import unittest

import torch

from jepa_lab.probes import ProbeConfig, fit_probe, select_probe


def synthetic(repeats=4):
    labels = torch.arange(10).repeat(repeats)
    values = torch.nn.functional.one_hot(labels,10).float()
    return torch.cat([values,torch.full((len(labels),1),3.0)],dim=1), labels


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fit_is_repeatable_frozen_and_preserves_caller_rng(self):
        features, labels = synthetic(); features.requires_grad_()
        before, state = features.detach().clone(), torch.get_rng_state().clone()
        config = ProbeConfig(steps=60, learning_rate=0.1)
        first, second = fit_probe(features, labels, config), fit_probe(features, labels, config)
        self.assertTrue(torch.equal(first.weight,second.weight))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertIsNone(features.grad)
        self.assertTrue(torch.equal(before,features))
        self.assertTrue(first.constant_columns[-1])
        self.assertEqual(1.0,float(first.scale[-1]))
        self.assertEqual(1.0,first.score(*synthetic(2))["accuracy"])
        self.assertFalse(first.weight.requires_grad)

    def test_normalization_and_weights_do_not_use_validation_values(self):
        train, labels = synthetic(); validation, held = synthetic(2)
        candidates = (ProbeConfig(steps=30),)
        a, _ = select_probe(train,labels,validation,held,candidates)
        b, _ = select_probe(train,labels,validation+10,held,candidates)
        self.assertTrue(torch.equal(a.mean,train.mean(0)))
        self.assertTrue(torch.equal(a.mean,b.mean))
        self.assertTrue(torch.equal(a.scale,b.scale))
        self.assertTrue(torch.equal(a.weight,b.weight))

    def test_candidate_selection_metrics_and_missing_class_reporting(self):
        values, labels = synthetic()
        configs = (ProbeConfig(steps=25,learning_rate=0.01),ProbeConfig(steps=25,learning_rate=0.1))
        _, report = select_probe(values,labels,*synthetic(2),configs)
        metrics = [item["validation"] for item in report["candidates"]]
        expected = min(range(2),key=lambda i:(-metrics[i]["accuracy"],metrics[i]["cross_entropy"],i))
        self.assertEqual(expected,report["selected_candidate"])
        self.assertEqual(20,sum(sum(row) for row in metrics[0]["confusion_matrix"]))
        self.assertEqual([2]*10,[item["count"] for item in metrics[0]["per_class"]])
        probe = fit_probe(values,labels,configs[0])
        partial = probe.score(values[:1],labels[:1])
        self.assertIsNone(partial["per_class"][1]["accuracy"])

    def test_invalid_inputs_and_inconsistent_candidate_budgets_are_rejected(self):
        values, labels = synthetic()
        for settings in (dict(steps=True),dict(steps=1001),dict(learning_rate=float("nan")),
                         dict(weight_decay=-1),dict(seed=True)):
            with self.subTest(settings=settings),self.assertRaises(ValueError): ProbeConfig(**settings)
        for data,targets in ((values.double(),labels),(values,labels.float()),(values,labels[:-1]),
                             (values[:5],labels[:5]),(torch.full_like(values,float("inf")),labels)):
            with self.assertRaises(ValueError): fit_probe(data,targets)
        for configs in ((),(ProbeConfig(),ProbeConfig()),(ProbeConfig(steps=2),ProbeConfig(steps=3)),
                        (ProbeConfig(seed=1),ProbeConfig(seed=2))):
            with self.assertRaises(ValueError): select_probe(values,labels,values,labels,configs)
