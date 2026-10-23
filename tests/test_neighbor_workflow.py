from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.neighbors import NeighborConfig
from jepa_lab.neighbor_workflow import neighbor_record
from test_compare import sets


class NeighborWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cached_grid_replays_every_candidate_and_rejects_metadata_or_path_edits(self):
        pair, grid = sets()['checkpoint'], [NeighborConfig(1), NeighborConfig(3)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'selected.json'
            row = neighbor_record(path, pair, grid)
            before = {p: p.read_bytes() for p in path.parent.iterdir()}
            with patch('jepa_lab.neighbor_workflow.select_neighbors', side_effect=AssertionError('retuning')):
                self.assertEqual(row, neighbor_record(path, pair, grid, reuse_only=True))
            for edit in ('context', 'path', 'losing_score'):
                broken = deepcopy(row)
                if edit == 'context':
                    broken['selection']['context']['selection_sha256'] = '0'*64
                elif edit == 'path':
                    alias = path.parent/'alias.pt'
                    alias.write_bytes(path.with_suffix('.pt').read_bytes())
                    broken['path'] = str(alias)
                else:
                    broken['selection']['candidates'][1]['validation']['confusion_matrix'][0][0] += 1
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    neighbor_record(path, pair, grid, reuse_only=True)
                self.assertEqual(before[path.with_suffix('.pt')], path.with_suffix('.pt').read_bytes())
            path.write_bytes(before[path])
            self.assertEqual(row, neighbor_record(path, pair, grid, reuse_only=True))

    def test_partial_state_is_preserved_and_missing_restore_or_confirmation_cannot_fit(self):
        from dataclasses import replace
        pair, grid = sets()['checkpoint'], [NeighborConfig(1), NeighborConfig(3)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'selected.json'
            with patch('jepa_lab.neighbor_workflow.select_neighbors', side_effect=AssertionError('fit')):
                with self.assertRaisesRegex(ValueError, 'missing'):
                    neighbor_record(path, pair, grid, reuse_only=True)
                with self.assertRaises(ValueError):
                    neighbor_record(path, (pair[0], replace(pair[1], selection_kind='confirmation')), grid)
            row = neighbor_record(path, pair, grid)
            index = path.with_suffix('.pt')
            before = index.read_bytes()
            path.unlink()
            self.assertEqual(row, neighbor_record(path, pair, grid))
            self.assertEqual(before, index.read_bytes())
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'Partial'):
                neighbor_record(path, pair, [NeighborConfig(3)])
            self.assertFalse(path.exists())
            self.assertEqual(before, index.read_bytes())
