from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab import objective_selection as selection
from jepa_lab.probes import ProbeConfig
from test_position_compare import position_arms


class ObjectiveSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / 'source.json'
        self.pair = position_arms()['positionsincos2d']['checkpoint']
        self.candidates = (ProbeConfig(steps=2, learning_rate=.001), ProbeConfig(steps=2, learning_rate=.01))

    def record(self, **options):
        return selection.source_record(self.path, self.pair, self.candidates, [29, 31], [.1, 1.], **options)

    def test_all_candidate_states_replay_and_tampered_losing_scores_fail(self):
        rng = torch.get_rng_state().clone()
        result = self.record()
        self.assertEqual(6, result['verified_candidate_states'])
        self.assertEqual(3, result['verified_selected_states'])
        before = {p: p.read_bytes() for p in self.path.parent.rglob('*') if p.is_file()}
        with patch.object(selection, 'fit_probe', side_effect=AssertionError('refit')), \
                patch.object(selection, 'fit_ridge', side_effect=AssertionError('ridge refit')):
            self.assertEqual(result, self.record(reuse_only=True))
            changed = deepcopy(result['report'])
            grid = changed['grids'][0]
            loser = 1-grid['selected_candidate']
            grid['candidates'][loser]['scores']['validation']['accuracy'] += .01
            self.path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'candidate scores'):
                self.record(reuse_only=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual({p: data for p, data in before.items() if p != self.path},
                         {p: p.read_bytes() for p in before if p != self.path})

    def test_partial_report_recovery_reuses_states_and_rejects_changed_grids(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.record(reuse_only=True)
        result = self.record()
        self.path.unlink()
        with patch.object(selection, 'fit_probe', side_effect=AssertionError('partial refit')), \
                patch.object(selection, 'fit_ridge', side_effect=AssertionError('partial refit')):
            self.assertEqual(result, self.record())
        before = self.path.read_bytes()
        self.candidates = (ProbeConfig(steps=3),)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.record()
        self.assertEqual(before, self.path.read_bytes())
