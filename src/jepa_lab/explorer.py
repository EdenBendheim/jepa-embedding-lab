"""Inspect saved frozen neighborhoods; no fitting, image export, or test access."""
import argparse
import json
from pathlib import Path

import torch

from .cifar10 import CIFAR10Binary, CLASSES, make_manifest
from .compare import write_report
from .features import load_features
from .neighbor_artifacts import index_context, load_neighbors
from .selection import validate_pilot


def explore_neighbors(index_path, train, validation, *, query_indices=None):
    """Restore an exact pilot-bound index and describe up to 100 validation rows."""
    index_context(train, validation)
    ids = tuple(validation.indices[:20] if query_indices is None else query_indices)
    if (not 1 <= len(ids) <= 100 or len(set(ids)) != len(ids)
            or any(type(i) is not int or i not in validation.indices for i in ids)):
        raise ValueError('Choose 1-100 distinct official indices from the validation selection')
    index, metadata = load_neighbors(index_path, train, validation)
    rows = [validation.indices.index(i) for i in ids]
    positions, similarities = index.nearest(validation.features[rows])
    queries = []
    for row, neighbors, scores in zip(rows, positions.tolist(), similarities.tolist()):
        label = int(validation.labels[row])
        queries.append(dict(index=validation.indices[row], label=label, class_name=CLASSES[label],
            neighbors=[dict(rank=rank+1, index=train.indices[pos], label=int(train.labels[pos]),
                class_name=CLASSES[int(train.labels[pos])], cosine=score)
                for rank, (pos, score) in enumerate(zip(neighbors, scores))]))
    return dict(version=1, task='Frozen embedding neighborhood explorer',
        training=train.summary(), validation=validation.summary(), saved_index=metadata,
        queries=queries, test_evaluated=False, restored_without_fitting=True,
        limitation='Descriptive tuning-row inspection, not new quality evidence. Cosines are not probabilities; '
                   'zero vectors tie at zero and stable ties follow fitting-row order. No raw images or feature vectors included.')


def load_exploration(root, manifest_path, selection_path, train_path, validation_path, index_path, *, query_indices=None):
    manifest=json.loads(Path(manifest_path).read_text())
    train,test=CIFAR10Binary(root),CIFAR10Binary(root,train=False)
    if manifest != make_manifest(train,test,seed=manifest['seed']):
        raise ValueError('Dataset differs from the fixed split')
    selection=validate_pilot(json.loads(Path(selection_path).read_text()),manifest,train.labels)
    if selection.get('kind') == 'confirmation':
        raise ValueError('Explorer uses the original observed tuning pilot; do not expose fresh confirmation rows')
    pair=tuple(load_features(path,manifest=manifest,selection=selection,training_labels=train.labels,partition=part)
        for path,part in ((train_path,'train'),(validation_path,'validation')))
    return explore_neighbors(index_path,*pair,query_indices=query_indices)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    for name in ('manifest','selection','train-features','validation-features','index','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--query-index',type=int,nargs='+',help='Official training-file indices inside the validation pilot (default first 20)')
    args=p.parse_args();torch.set_num_threads(1)
    try:
        report=load_exploration(args.root,args.manifest,args.selection,args.train_features,args.validation_features,
            args.index,query_indices=args.query_index)
        write_report(args.output,report)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:p.error(str(exc))
    print(json.dumps(dict(output=str(args.output),query_count=len(report['queries']),test_evaluated=False)))


if __name__ == '__main__': main()
