"""Cache and replay a complete tuning-only kNN selection without retuning."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from .compare import write_report
from .neighbor_artifacts import FIELDS, index_context, load_neighbors, save_neighbors, select_neighbors
from .neighbors import NeighborConfig


def neighbor_record(path, pair, candidates, reuse_only=False):
    path = Path(path)
    index_path = path.with_suffix('.pt')
    if (not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= 8
            or any(not isinstance(c, NeighborConfig) for c in candidates)
            or len(set(candidates)) != len(candidates)):
        raise ValueError('Use 1-8 distinct neighbor configurations')
    context = index_context(*pair)  # Refuse confirmation or mutated sources before writing.
    if not path.exists():
        if reuse_only:
            raise ValueError('Selected neighbor report missing')
        index, selected = select_neighbors(*pair, candidates)
        if index_path.exists():
            restored, artifact = load_neighbors(index_path, *pair)
            if (restored.config != index.config
                    or any(not torch.equal(getattr(restored, field), getattr(index, field)) for field in FIELDS)):
                raise ValueError('Partial neighbor state differs; preserve the previous artifact')
        else:
            artifact = save_neighbors(index_path, index, *pair)
        write_report(path, dict(selection=selected, index=artifact, path=str(index_path)))
    row = json.loads(path.read_text())
    if Path(row['path']).resolve() != index_path.resolve():
        raise ValueError('Neighbor report must retain its adjacent saved index')
    index, artifact = load_neighbors(index_path, *pair)
    selection = row['selection']
    options, chosen = selection['candidates'], selection['selected_candidate']
    if (selection['context'] != context or selection['test_evaluated'] is not False
            or selection['selection_rule'] != 'Highest tuning accuracy, then declared order'
            or [r['config'] for r in options] != [asdict(c) for c in candidates]
            or type(chosen) is not int or not 0 <= chosen < len(options)
            or chosen != min(range(len(options)), key=lambda i: (-options[i]['validation']['accuracy'], i))
            or row['index'] != artifact or options[chosen]['config'] != artifact['config']
            or options[chosen]['validation'] != artifact['scores']['validation']):
        raise ValueError('Selected neighbor grid/state/result differs')
    for option, config in zip(options, candidates, strict=True):
        candidate = replace(index, config=config)
        if option['validation'] != candidate.score(pair[1].features, pair[1].labels):
            raise ValueError('Reported neighbor candidate scores do not reproduce')
    return row
