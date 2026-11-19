"""Verify shared training prefixes and report actual costs at fixed milestones."""

from jepa_lab.variants.multiblock_features import load_cifar_encoder
from jepa_lab.variants.multiblock_train import _digest

from experiments.multiblock_trajectory import checkpoint_path


def _shared_inputs(history):
    return [(row['train_indices'], row['mask_sha256']) for row in history]


def verify_milestones(data, directory, receipts):
    """Load frozen encoders and check that compared trajectories share their inputs."""
    final_steps = {arm: chunks[-1]['end_step'] for arm, chunks in receipts.items()}
    shared = min(final_steps.values())
    if set(final_steps) != {'separate', 'joint'} or final_steps['joint'] < final_steps['separate']:
        raise ValueError('The joint trajectory must cover the separate update budget')
    milestones = {'separate': ('separate', shared), 'joint_matched': ('joint', shared)}
    if final_steps['joint'] > shared:
        milestones['joint_extended'] = ('joint', final_steps['joint'])

    models, histories, costs, recipes = {}, {}, {}, {}
    for name, (arm, step) in milestones.items():
        chunks = [row for row in receipts[arm] if row['end_step'] <= step]
        endpoint = chunks[-1]
        if endpoint['end_step'] != step:
            raise ValueError('The shared milestone must have a saved checkpoint in both arms')
        path = checkpoint_path(directory / arm, step)
        model, metadata, history = load_cifar_encoder(path, data, expected_sha256=endpoint['checkpoint_sha256'])
        settings = metadata['settings'].copy()
        grouping = settings.pop('predictor_grouping', 'separate')
        if (grouping != arm or metadata['completed_steps'] != step
                or any((row['context_tokens'], row['target_tokens_per_image'], row['predictor_passes'])
                       != (36, 16, 1 if arm == 'joint' else 4) for row in history)):
            raise ValueError('Checkpoint differs from the declared predictor grouping or geometry')
        recipes[name] = metadata['config'], settings, metadata['initial_state_sha256']
        histories[name] = history
        models[name] = model
        costs[name] = dict(updates=step, chunks=len(chunks), checkpoint_sha256=endpoint['checkpoint_sha256'],
            loop_seconds=sum(row['loop_ns'] for row in chunks) / 1e9,
            trainer_wall_seconds=sum(row['wall_ns'] for row in chunks) / 1e9,
            image_order_sha256=_digest([row['train_indices'] for row in history]),
            mask_order_sha256=_digest([row['mask_sha256'] for row in history]),
            first_batch_loss=history[0]['loss'], last_batch_loss=history[-1]['loss'])
    matched = histories['joint_matched']
    if (any(recipe != recipes['separate'] for recipe in recipes.values())
            or _shared_inputs(histories['separate']) != _shared_inputs(matched)):
        raise ValueError('Matched milestones must share initialization, images and masks')
    if 'joint_extended' in histories and histories['joint_extended'][:shared] != matched:
        raise ValueError('Extended joint trajectory must retain its complete shared journal')
    return models, dict(shared_updates=shared, initial_state_sha256=recipes['separate'][2],
                        milestones=costs)
