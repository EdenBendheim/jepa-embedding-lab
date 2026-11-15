import unittest

from jepa_lab.variants.multiblock_timing import estimate_step_budget


class StepBudgetTests(unittest.TestCase):
    def test_step_budget_uses_slower_order_and_keeps_equal_updates_at_slowest_arm(self):
        report = dict(trials=[
            dict(arms=dict(separate=dict(mean_step_ms=16), joint=dict(mean_step_ms=8))),
            dict(arms=dict(separate=dict(mean_step_ms=20), joint=dict(mean_step_ms=10))),
        ])
        budget = estimate_step_budget(report, seconds_per_arm=5)
        self.assertEqual(dict(separate=200, joint=400), budget['equal_time_steps'])
        self.assertEqual(200, budget['equal_update_steps'])
        self.assertEqual(dict(separate=1000, joint=1000),
                         estimate_step_budget(report, seconds_per_arm=60)['equal_time_steps'])
        self.assertEqual(0, estimate_step_budget(report, seconds_per_arm=.001)['equal_update_steps'])

    def test_incomplete_or_nonfinite_timing_cannot_drive_a_budget(self):
        for report in (dict(trials=[]), dict(trials=[dict(arms={})]),
                       dict(trials=[dict(arms=dict(separate=dict(mean_step_ms=0), joint=dict(mean_step_ms=float('nan'))))])):
            with self.assertRaises(ValueError):
                estimate_step_budget(report, seconds_per_arm=5)
