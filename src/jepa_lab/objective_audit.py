"""Audit completed objective-training journals; no fitting or quality scoring."""

import json
import math
from pathlib import Path
from statistics import mean

from .compare import write_report
from .objective import objective_from_settings
from .training_diagnostics import summarize_training


def objective_diagnostics(report, *, window=100):
    diagnostics = summarize_training(report, window=window)
    config = objective_from_settings(report['settings'])
    if config.variance_weight:
        for row in report['history']:
            terms = [row.get(k) for k in ('prediction_loss', 'context_variance_penalty', 'context_std_mean')]
            if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in terms)
                    or not math.isclose(row['loss'], terms[0] + config.variance_weight*terms[1],
                                        rel_tol=1e-5, abs_tol=1e-6)):
                raise ValueError('Objective journal has invalid components or loss decomposition')
    for summary in diagnostics['windows']:
        rows = report['history'][summary['first_step']-1:summary['last_step']]
        summary['prediction_loss_mean'] = mean(r['prediction_loss'] if config.variance_weight else r['loss'] for r in rows)
        if config.variance_weight:
            summary['context_variance_penalty_mean'] = mean(r['context_variance_penalty'] for r in rows)
            summary['visible_context_image_std_mean'] = mean(r['context_std_mean'] for r in rows)
    diagnostics['limitation'] = ('The objectives have different total-loss definitions. Compare the separate '
        'prediction/penalty terms as training diagnostics only; neither establishes downstream quality.')
    return diagnostics


def audit_training(run, *, output=None):
    reports = run.training(reuse_only=True)
    arms = {}
    for name, report in reports.items():
        arms[name] = {k: report[k] for k in ('config', 'settings', 'checkpoint_sha256', 'model_size',
            'train_image_order_sha256', 'reconstructed_initial_state_sha256', 'derived_mask_order_sha256')}
        arms[name]['diagnostics'] = objective_diagnostics(report)
    result = dict(version=1, protocol_sha256=run.expected_sha, training=arms,
        actual_image_order_matched=True, trainable_capacity_matched=True,
        reconstructed_initialization_matched=True, derived_mask_schedule_matched=True,
        raw_images_sampled=False, models_trained=False, classifiers_fitted=False,
        confirmation_scored=False, test_evaluated=False,
        limitation='Source-verified training-journal replay, not an independent ancestry attestation. '
                   'The shared mask schedule is reconstructed, not separately recorded in these journals. '
                   'Higher variance or lower prediction loss does not establish useful embeddings.')
    if output is not None:
        output = Path(output)
        d, manifest, ref, fresh, labels, bundle = run.inputs()
        protected = {Path(row['path']).resolve() for row in d['sources']} | {run.protocol.resolve()}
        protected.update(Path(row['path']).resolve() for row in bundle['source_files'])
        if (output.resolve() in protected or output.resolve().is_relative_to(run.root.resolve())
                or output.resolve().is_relative_to(run.directory.resolve())):
            raise ValueError('Objective audit cannot replace data or experiment evidence; use a separate output')
        if output.exists():
            if json.loads(output.read_text()) != result:
                raise ValueError('Existing objective audit differs; preserve the prior file')
        else:
            write_report(output, result)
    return result
