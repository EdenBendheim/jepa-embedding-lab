"""Execute the committed 24-state, no-refit confirmation protocol locally."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import torch

from jepa_lab.cifar10 import CIFAR10Binary, make_manifest
from jepa_lab.compare import write_report
from jepa_lab.confirmation import score_confirmation
from jepa_lab.embeddings import extract_cifar10, load_encoder
from jepa_lab.features import load_features
from jepa_lab.probe_artifacts import load_probe
from jepa_lab.selection import make_confirmation, validate_pilot


ROOT = Path('data/cifar-10-batches-bin')
MANIFEST = Path('runs/cifar10-manifest.json')
PROTOCOL = Path('experiments/confirmation-sources-20261012.json')
PROTOCOL_SHA = '35d72dd90702f8fe0e3e5f5fd3010c35c395185be4398a619cd904d01fc3a024'
ARMS = ('learned_mean', 'learned_mean_std', 'learned_mask4x4', 'random_mean', 'random_mean_std', 'pixels')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(directory=Path('runs/confirmation-20261012'), *, reuse_only=False):
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('Committed confirmation source protocol changed')
    declared = json.loads(PROTOCOL.read_text())
    train, test = CIFAR10Binary(ROOT), CIFAR10Binary(ROOT, train=False)
    manifest = json.loads(MANIFEST.read_text())
    if manifest != make_manifest(train, test, seed=manifest['seed']) or manifest['manifest_sha256'] != declared['manifest_sha256']:
        raise ValueError('Declared dataset or split changed')
    reference = validate_pilot(json.loads(Path(declared['reference_path']).read_text()), manifest, train.labels)
    if reference['selection_sha256'] != declared['reference_selection_sha256']:
        raise ValueError('Original fitted selection changed')
    prior = [validate_pilot(json.loads(Path(p).read_text()), manifest, train.labels) for p in declared['excluded_paths']]
    selected = make_confirmation(manifest, train.labels, reference, exclusions=prior,
        seed=declared['confirmation_seed'], validation_per_class=declared['confirmation_per_class'])
    directory = Path(directory)
    selection_path = directory/'selection.json'
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selected:
            raise ValueError('Cached confirmation selection differs from the declared examples')
    elif reuse_only:
        raise ValueError('Cached selection is missing; reuse-only must not extract new data')
    else:
        write_report(selection_path, selected)
    for source in [*declared['source_reports'].values(), *declared['checkpoints'].values()]:
        if sha(source['path']) != source['sha256']:
            raise ValueError('Declared historical report or checkpoint bytes changed')

    # Verify all historical states before extracting or observing any new score.
    sources = {}
    for row in declared['classifiers']:
        for field, fingerprint in (('probe','probe_sha256'),('train_features','train_sha256'),('tuning_features','tuning_sha256')):
            if sha(row[field]) != row[fingerprint]:
                raise ValueError('Declared original classifier/source bytes changed')
        arm = row['arm']
        if arm not in sources:
            sources[arm] = tuple(load_features(row[field], manifest=manifest, selection=reference,
                training_labels=train.labels, partition=partition)
                for field, partition in (('train_features','train'),('tuning_features','validation')))
        restored, original = load_probe(row['probe'], *sources[arm])
        if (original['state_sha256'] != row['state_sha256'] or original['config'] != row['selected_config']
                or original['scores']['validation']['accuracy'] != row['original_tuning_accuracy']):
            raise ValueError('Fixed classifier no longer reproduces its declared tuning result')
    if set(sources) != set(ARMS) or len(declared['classifiers']) != 24:
        raise ValueError('Exactly the six declared arms and 24 states are required')
    _, baseline = load_encoder(declared['checkpoints']['default']['path'], manifest['manifest_sha256'])
    from jepa_lab.model import ModelConfig
    config = ModelConfig(**baseline['config'])

    fresh = {}
    for arm in ARMS:
        path = directory/'features'/f'{arm}-validation.pt'
        if not path.exists():
            if reuse_only: raise ValueError('A cached feature arm is missing')
            representation = 'pixels' if arm == 'pixels' else 'random' if arm.startswith('random_') else 'checkpoint'
            checkpoint = declared['checkpoints']['mask4x4' if arm == 'learned_mask4x4' else 'default']['path'] if representation == 'checkpoint' else None
            extract_cifar10(ROOT, MANIFEST, partition='validation', selection_path=selection_path,
                representation=representation, checkpoint=checkpoint, config=config if representation == 'random' else None,
                pooling='mean_std' if arm.endswith('mean_std') else 'mean', seed=29, batch_size=16, output=path)
        fresh[arm] = load_features(path, manifest=manifest, selection=selected, training_labels=train.labels, partition='validation')
        if fresh[arm].metadata != sources[arm][0].metadata:
            raise ValueError('Cached confirmation representation differs from its original frozen source')
    reference_fresh = fresh[ARMS[0]]
    if any(item.indices != reference_fresh.indices or not torch.equal(item.labels, reference_fresh.labels) for item in fresh.values()):
        raise ValueError('All arms must score the identical ordered confirmation examples')

    rows = []
    for row in declared['classifiers']:
        result = score_confirmation(row['probe'], *sources[row['arm']], fresh[row['arm']], selected)
        rows.append(dict(arm=row['arm'], head=row['head'], classifier_seed=row['classifier_seed'],
            selected_config=row['selected_config'], state_sha256=row['state_sha256'], probe_sha256=row['probe_sha256'],
            original_tuning_accuracy=row['original_tuning_accuracy'], confirmation=result['confirmation'],
            source_sha256=dict(train=row['train_sha256'], tuning=row['tuning_sha256'], confirmation=fresh[row['arm']].artifact_sha256),
            original_state_and_scores_reproduced=True, classifier_refitted=False, normalization_refitted=False))
    summary = {}
    for arm in ARMS:
        adamw = [row for row in rows if row['arm'] == arm and row['head'] == 'adamw']
        ridge = [row for row in rows if row['arm'] == arm and row['head'] == 'ridge']
        if [row['classifier_seed'] for row in adamw] != [29,31,37] or len(ridge) != 1:
            raise ValueError('Each arm requires the three fixed seeds and one fixed ridge state')
        values = [row['confirmation']['accuracy'] for row in adamw]
        summary[arm] = dict(feature_dim=fresh[arm].features.shape[1],
            adamw=dict(accuracies=values, mean_accuracy=mean(values), sample_standard_deviation=stdev(values),
                       original_tuning_mean=mean(row['original_tuning_accuracy'] for row in adamw)),
            ridge=dict(accuracy=ridge[0]['confirmation']['accuracy'], original_tuning_accuracy=ridge[0]['original_tuning_accuracy']))
    excluded = {i for parent in [reference,*prior] for i in parent['partitions']['validation']['indices']}
    return dict(version=1, date='2026-10-12', protocol_sha256=PROTOCOL_SHA, manifest_sha256=manifest['manifest_sha256'],
        reference_selection_sha256=reference['selection_sha256'], confirmation_selection_sha256=selected['selection_sha256'],
        counts=dict(fitted_train=len(sources[ARMS[0]][0].labels), tuning=len(sources[ARMS[0]][1].labels),
                    confirmation=len(reference_fresh.labels), excluded_prior_validation=len(excluded)),
        summary=summary, classifiers=rows, verified_classifiers=len(rows), classifier_refitted=False,
        normalization_refitted=False, settings_selected=False, test_evaluated=False,
        limitation='Fresh examples only relative to declared pilots; one learned encoder seed per mask, shared original validation partition, unequal feature dimensions/token budgets, and descriptive classifier-seed variation. Scores must not be reused to claim untuned confirmation of later model choices.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('runs/confirmation-20261012'))
    parser.add_argument('--reuse-only', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('runs/confirmation-20261012/summary.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        result = run(args.directory, reuse_only=args.reuse_only)
        write_report(args.output, result)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(summary=result['summary'], counts=result['counts'],
                         verified_classifiers=result['verified_classifiers'], test_evaluated=False), indent=2))


if __name__ == '__main__': main()
