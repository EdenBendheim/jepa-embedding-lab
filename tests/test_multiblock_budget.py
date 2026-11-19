import unittest

from experiments.multiblock_budget import budget_schedule


def timing(separate=20, joint=10):
    return dict(trials=[dict(arms=dict(separate=dict(mean_step_ms=separate),
                                      joint=dict(mean_step_ms=joint)))])


class BudgetScheduleTests(unittest.TestCase):
    def test_per_arm_budget_retains_the_shared_checkpoint_and_bounded_chunks(self):
        schedule = budget_schedule(timing(), seconds_per_arm=5, chunk_steps=75)
        self.assertEqual({
            'separate': [75, 150, 200],
            'joint': [75, 150, 200, 275, 350, 400],
        }, schedule['checkpoints'])
        self.assertEqual(200, schedule['planning']['equal_update_steps'])

    def test_same_budget_has_one_checkpoint_at_each_boundary(self):
        schedule = budget_schedule(timing(joint=20), seconds_per_arm=5, chunk_steps=100)
        self.assertEqual([100, 200], schedule['checkpoints']['joint'])
        self.assertEqual(schedule['checkpoints']['joint'], schedule['checkpoints']['separate'])

    def test_invalid_or_clipped_schedules_are_rejected(self):
        for seconds, chunk in ((60, 100), (.001, 100), (5, 101), (5, True)):
            with self.subTest(seconds=seconds, chunk=chunk), self.assertRaises(ValueError):
                budget_schedule(timing(), seconds_per_arm=seconds, chunk_steps=chunk)
