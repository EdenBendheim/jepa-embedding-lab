"""Select or restore a kNN index from verified saved features, without image reads."""
import argparse
import json
from pathlib import Path

import torch

from .cifar10 import CIFAR10Binary,make_manifest
from .compare import write_report
from .features import load_features
from .neighbor_artifacts import select_neighbors,save_neighbors,load_neighbors
from .neighbors import NeighborConfig
from .selection import validate_pilot


def run_neighbors(root,manifest_path,selection_path,train_path,validation_path,*,ks=None,save_index=None,restore_index=None):
    if (save_index is None)==(restore_index is None):raise ValueError('Choose exactly one save-index or restore-index mode')
    if restore_index is not None and ks is not None:raise ValueError('Restore mode cannot select new neighbor counts')
    if save_index is not None and Path(save_index).exists():raise ValueError('Keep earlier neighbor states; choose a new index path')
    manifest=json.loads(Path(manifest_path).read_text());train,test=CIFAR10Binary(root),CIFAR10Binary(root,train=False)
    if manifest!=make_manifest(train,test,seed=manifest['seed']):raise ValueError('Dataset differs from the fixed split')
    selection=validate_pilot(json.loads(Path(selection_path).read_text()),manifest,train.labels)
    if selection.get('kind')=='confirmation':raise ValueError('Neighbor selection/restore requires the original tuning pilot')
    pair=tuple(load_features(path,manifest=manifest,selection=selection,training_labels=train.labels,partition=part)
        for path,part in ((train_path,'train'),(validation_path,'validation')))
    if restore_index is not None:
        index,metadata=load_neighbors(restore_index,*pair);selected=None
    else:
        configs=tuple(NeighborConfig(k) for k in ([1,5,15] if ks is None else ks))
        index,selected=select_neighbors(*pair,configs);metadata=save_neighbors(save_index,index,*pair)
    return dict(version=1,task='Frozen cosine-neighbor tuning audit',selection=selected,
        saved_index=metadata,validation=index.score(pair[1].features,pair[1].labels),
        restored_without_fitting=restore_index is not None,test_evaluated=False,
        limitation='Tuning-selected neighbor count; uniform votes are not calibrated probabilities or official test evidence')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    for name in ('manifest','selection','train-features','validation-features'):p.add_argument('--'+name,type=Path,required=True)
    modes=p.add_mutually_exclusive_group(required=True);modes.add_argument('--save-index',type=Path);modes.add_argument('--restore-index',type=Path)
    p.add_argument('--k',type=int,nargs='+');p.add_argument('--output',type=Path)
    args=p.parse_args();torch.set_num_threads(1)
    try:
        result=run_neighbors(args.root,args.manifest,args.selection,args.train_features,args.validation_features,
            ks=args.k,save_index=args.save_index,restore_index=args.restore_index)
        if args.output:write_report(args.output,result)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:p.error(str(exc))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
