"""Descriptive, paired classification errors on the same ordered examples."""

import torch


def _classes(values, count=None):
    if (not isinstance(values, torch.Tensor) or values.device.type != "cpu"
            or values.dtype != torch.long or values.ndim != 1
            or not 1 <= len(values) <= 1000 or (count is not None and len(values) != count)
            or (values < 0).any() or (values >= 10).any()):
        raise ValueError("Use matching, nonempty CPU int64 class vectors in 0-9 (at most 1000)")


def paired_diagnostics(labels, baseline, candidate):
    """Count recoveries/regressions; callers must bind all vectors to identical indices.

    These are descriptive validation counts, not a significance test. No example
    images, features, identifiers, or probability scores are returned.
    """
    _classes(labels)
    _classes(baseline, len(labels))
    _classes(candidate, len(labels))
    before, after = baseline == labels, candidate == labels
    recovered, regressed = ~before & after, before & ~after
    before_matrix = torch.bincount(labels * 10 + baseline, minlength=100).reshape(10, 10)
    after_matrix = torch.bincount(labels * 10 + candidate, minlength=100).reshape(10, 10)
    per_class = []
    for label in range(10):
        selected = labels == label
        count = int(selected.sum())
        per_class.append(dict(label=label, count=count,
            baseline_correct=int((selected & before).sum()), candidate_correct=int((selected & after).sum()),
            recovered=int((selected & recovered).sum()), regressed=int((selected & regressed).sum()),
            accuracy_delta=float((after[selected].sum()-before[selected].sum()) / count) if count else None))
    return dict(count=len(labels), baseline_correct=int(before.sum()), candidate_correct=int(after.sum()),
        both_correct=int((before & after).sum()), both_wrong=int((~before & ~after).sum()),
        recovered=int(recovered.sum()), regressed=int(regressed.sum()),
        predictions_changed=int((baseline != candidate).sum()),
        accuracy_delta=(int(after.sum())-int(before.sum()))/len(labels), per_class=per_class,
        baseline_prediction_counts=torch.bincount(baseline, minlength=10).tolist(),
        candidate_prediction_counts=torch.bincount(candidate, minlength=10).tolist(),
        confusion_delta=(after_matrix-before_matrix).tolist(),
        limitation="Descriptive paired validation errors only; no significance or independent confirmation")
