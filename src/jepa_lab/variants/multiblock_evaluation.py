"""Score the matched pair on fixed classifier folds within CIFAR-10 training rows."""

import torch

from ..analysis.color_statistics import FEATURE_NAMES, rgb_statistics
from ..analysis.spatial_evaluation import crossfit_ridge, stratified_folds
from ..diagnostics import paired_diagnostics
from ..embeddings import image_embeddings, random_encoder
from ..feature_geometry import feature_geometry
from ..selection import validate_pilot
from .multiblock_pair import verify_pair
from .multiblock_train import _digest


def evaluate_pair(data, directory, spec, selection):
    if selection.get('kind') == 'confirmation':
        raise ValueError('Use a fixed fitting pilot for the multi-block diagnostic')
    validate_pilot(selection, data.manifest, data.labels)
    indices = selection['partitions']['train']['indices']
    labels = torch.tensor([data.labels[index] for index in indices], dtype=torch.long)
    assignments = stratified_folds(labels)
    arms, training = verify_pair(data, directory, spec)
    reference, variant = arms
    models = {name: model for name, (model, _, _) in arms.items()}
    config = models[reference].config
    models['random'] = random_encoder(config, spec['settings']['seed'])
    feature_parts = {name: [] for name in (*models, 'rgb_statistics')}
    for start in range(0, len(indices), 16):
        batch = indices[start:start+16]
        images, _ = data.read(batch)
        for name, model in models.items():
            feature_parts[name].append(image_embeddings(model, images))
        feature_parts['rgb_statistics'].append(rgb_statistics(images))

    scores, predictions = {}, {}
    for name, parts in feature_parts.items():
        features = torch.cat(parts)
        predictions[name], score = crossfit_ridge(features, labels, assignments)
        scores[name] = dict(score=score, geometry=feature_geometry(features))
    paired = {}
    for baseline, candidate in ((reference, variant), ('random', reference), ('random', variant)):
        errors = paired_diagnostics(labels, predictions[baseline], predictions[candidate])
        errors['limitation'] = 'Descriptive differences on training rows with fixed classifier folds.'
        paired[f'{candidate}-minus-{baseline}'] = errors

    summaries = {}
    for name, (_, _, journal) in arms.items():
        first, last = journal[0], journal[-1]
        passes = first['predictor_passes']
        tokens_per_pass = first['context_tokens'] + first['target_tokens_per_image']//passes
        summaries[name] = dict(first_batch_loss=first['loss'], last_batch_loss=last['loss'],
            predictor_passes=passes, predictor_sequence_tokens=tokens_per_pass,
            predictor_tokens_per_image=passes*tokens_per_pass,
            predictor_attention_pairs_per_head_per_layer=passes*tokens_per_pass**2)

    first_journal = arms[reference][2]
    sampled = {index for row in first_journal for index in row['train_indices']}
    return dict(version=1, kind='matched_multiblock_training_row_diagnostic',
        manifest_sha256=data.manifest['manifest_sha256'], selection_sha256=selection['selection_sha256'],
        code_sha256=_digest(arms[reference][1]['code_sha256']), training=training, arms=summaries,
        scoring_rows=len(indices), scoring_indices_sha256=_digest(indices),
        scoring_rows_seen_in_encoder_training=len(sampled.intersection(indices)),
        folds=5, fold_seed=29, fold_assignments_sha256=_digest(assignments.tolist()), alpha=.1,
        encoder='context', pooling='mean', rgb_statistics=list(FEATURE_NAMES), scores=scores, paired=paired,
        partition='train', validation_images_indexed=False, test_images_indexed=False,
        limitation='Exploratory classifier cross-fitting on fixed training rows. A single encoder seed '
                   'and unequal predictor computation limit comparison. Independent confirmation is pending.')
