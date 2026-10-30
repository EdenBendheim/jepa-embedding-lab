"""Source-pinned preparation for a matched context-variance training contrast.

Preparation reads dataset hashes and labels, never image tensors. A reservation
is not an observation and must not enter the exposure ledger until scoring.
"""

import hashlib
import json
import math
from pathlib import Path

import torch

from .cifar10 import CIFAR10Binary, make_manifest
from .compare import write_report
from .exposure import exposure_summary, load_exposure_bundle, make_ledger_confirmation
from .masking import block_mask
from .model import ModelConfig, config_dict
from .model_size import model_size, validate_local_config
from .objective import ObjectiveConfig
from .selection import validate_pilot


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_paths():
    """Pin the complete installed experiment implementation, not a handpicked subset."""
    package = Path(__file__).resolve().parent
    return sorted([*package.glob('*.py'), package.parents[1] / 'pyproject.toml'])


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def validate_recipe(d):
    """Bound the two-arm experiment before any output or raw-image access."""
    if (type(d.get('version')) is not int or d['version'] != 1
            or d.get('kind') != 'context_variance_comparison'):
        raise ValueError('Expected a version-one context-variance comparison protocol')
    config = validate_local_config(ModelConfig(**d['config']))
    if d['config'] != config_dict(config):
        raise ValueError('Objective protocol needs canonical model configuration')
    arms = d.get('arms')
    if (not isinstance(arms, list) or len(arms) != 2 or any(not isinstance(a, dict)
            or set(a) != {'name', 'context_variance_weight', 'context_std_floor'} for a in arms)
            or [a['name'] for a in arms] != ['prediction', 'context_variance']):
        raise ValueError('Declare prediction then context_variance arms')
    objectives = [ObjectiveConfig(a['context_variance_weight'], a['context_std_floor']) for a in arms]
    if (objectives[0].variance_weight != 0 or objectives[0].std_floor != 1
            or objectives[1].variance_weight <= 0):
        raise ValueError('Compare the historical objective with one positive variance penalty')
    seeds = d.get('training_seeds')
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3
            or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Declare one to three distinct integer training seeds')
    for key, low, high in (('total_steps', 1, 10000), ('chunk_steps', 1, 1000),
                            ('checkpoint_every', 1, 1000), ('batch_size', 2, 32),
                            ('confirmation_per_class', 1, 100)):
        if type(d.get(key)) is not int or not low <= d[key] <= high:
            raise ValueError(f'Invalid bounded objective recipe: {key}')
    for key in ('learning_rate', 'momentum'):
        value = d.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'Invalid objective recipe: {key}')
    if d['learning_rate'] <= 0 or not 0 <= d['momentum'] <= 1:
        raise ValueError('Learning rate must be positive and EMA momentum in [0, 1]')
    mask = d.get('mask')
    if not isinstance(mask, dict) or set(mask) != {'target_rows', 'target_cols', 'context_fraction'}:
        raise ValueError('Declare the full shared target/context mask')
    block_mask(config.rows, config.cols, **mask, seed=0)
    if d.get('runtime') != dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads()):
        raise ValueError('Objective runtime differs from its predeclared identity')
    return config


class ObjectiveRun:
    def __init__(self, root, directory, *, protocol, expected_sha):
        self.root, self.directory, self.protocol = Path(root), Path(directory), Path(protocol)
        self.expected_sha = expected_sha

    def inputs(self):
        if not _sha(self.expected_sha) or file_sha(self.protocol) != self.expected_sha:
            raise ValueError('Objective protocol bytes changed')
        d = json.loads(self.protocol.read_text())
        validate_recipe(d)
        sources = d.get('sources')
        if not isinstance(sources, list) or not sources:
            raise ValueError('Objective inputs require original source-file pins')
        pins = {}
        for row in sources:
            if (not isinstance(row, dict) or set(row) != {'path', 'sha256'}
                    or not isinstance(row['path'], str) or not row['path'] or not _sha(row['sha256'])):
                raise ValueError('Invalid objective source pin')
            path = Path(row['path']).resolve()
            if path in pins:
                raise ValueError('Duplicate objective source path')
            pins[path] = row['sha256']
        required = [Path(d[k]).resolve() for k in ('manifest_path', 'selection_path', 'exposure_bundle_path')]
        if not set([*required, *code_paths()]) <= set(pins):
            raise ValueError('Pin the dataset, selection, exposure bundle and complete experiment code')
        for path, expected in pins.items():
            if file_sha(path) != expected:
                raise ValueError('Declared objective source bytes changed')
        manifest = json.loads(Path(d['manifest_path']).read_text())
        train, test = CIFAR10Binary(self.root), CIFAR10Binary(self.root, train=False)
        if (manifest != make_manifest(train, test, seed=manifest['seed'])
                or manifest['manifest_sha256'] != d['manifest_sha256']):
            raise ValueError('Objective dataset or fixed split changed')
        reference = validate_pilot(json.loads(Path(d['selection_path']).read_text()), manifest, train.labels)
        if reference['selection_sha256'] != d['selection_sha256']:
            raise ValueError('Original objective fitting selection changed')
        bundle = load_exposure_bundle(d['exposure_bundle_path'], manifest, train.labels)
        if bundle['ledger']['ledger_sha256'] != d['ledger_sha256']:
            raise ValueError('Objective exposure history changed')
        fresh = make_ledger_confirmation(manifest, train.labels, reference, bundle['ledger'],
            validation_per_class=d['confirmation_per_class'], seed=d['confirmation_seed'])
        if fresh['selection_sha256'] != d['confirmation_selection_sha256']:
            raise ValueError('Objective confirmation reservation changed')
        protected = set(pins) | {self.protocol.resolve()}
        protected.update(Path(row['path']).resolve() for row in bundle['source_files'])
        # All outputs live below this directory, so also reject evidence descendants.
        if any(self.directory.resolve() == p or self.directory.resolve() in p.parents for p in protected):
            raise ValueError('Objective output directory contains protected experiment evidence')
        selected = self.directory / 'confirmation-selection.json'
        if selected.exists() and json.loads(selected.read_text()) != fresh:
            raise ValueError('Existing objective reservation differs; preserve the prior file')
        return d, manifest, reference, fresh, train.labels, bundle

    def prepare(self, *, reuse_only=False):
        d, manifest, reference, fresh, labels, bundle = self.inputs()
        path = self.directory / 'confirmation-selection.json'
        if not path.exists():
            if reuse_only:
                raise ValueError('Reserved objective selection is missing')
            write_report(path, fresh)
        return dict(protocol_sha256=self.expected_sha, manifest_sha256=manifest['manifest_sha256'],
            reference_selection_sha256=reference['selection_sha256'],
            confirmation_selection_sha256=fresh['selection_sha256'],
            exposure=exposure_summary(bundle['ledger'], manifest, labels),
            reserved_count=len(fresh['partitions']['validation']['indices']),
            model_size=model_size(ModelConfig(**d['config'])), raw_images_sampled=False,
            models_trained=False, confirmation_scored=False, test_evaluated=False,
            limitation='Reservation only; source pins cannot detect omitted observations or authenticate a producer')
