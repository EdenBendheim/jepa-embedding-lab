"""Measure distribution and aligned layout changes in the frozen feature panel."""

from jepa_lab.feature_geometry import feature_geometry
from jepa_lab.stability import paired_features


MILESTONE_PAIRS = (
    ('separate', 'joint_matched'),
    ('joint_matched', 'joint_extended'),
    ('random', 'joint_extended'),
)


def measure_geometry(panel, seeds):
    geometry = {}
    for key, values in panel.items():
        if key == 'rgb':
            continue
        source, view = key.rsplit('/', 1)
        geometry.setdefault(source, {})[view] = feature_geometry(values)

    layout = {source: paired_features(panel[f'{source}/original'], panel[f'{source}/shuffled'])
              for source in geometry}
    milestones = {}
    for seed in seeds:
        milestones[str(seed)] = {
            f'{after}-vs-{before}': paired_features(
                panel[f'{seed}/{before}/original'], panel[f'{seed}/{after}/original'])
            for before, after in MILESTONE_PAIRS
        }
    return dict(views=geometry, paired_layout=layout, paired_milestones=milestones)
