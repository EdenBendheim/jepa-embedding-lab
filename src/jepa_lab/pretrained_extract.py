"""Bounded, offline pretrained feature extraction on exact selected examples."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from .cifar10 import CIFAR10Binary, make_manifest
from .cifar10_train import atomic_checkpoint
from .pretrained import load_pretrained, pretrained_embeddings
from .selection import validate_pilot


def extract_pretrained(root, manifest_path, selection_path, weights_path, *,
                       partition="train", batch_size=8, output=None):
    if partition not in ("train", "validation"):
        raise ValueError("Pretrained pilot extraction reserves the official test set")
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("Pretrained extraction requires batch size 1-16")
    train, test = CIFAR10Binary(root), CIFAR10Binary(root, train=False)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest != make_manifest(train,test,seed=manifest["seed"]):
        raise ValueError("Dataset differs from the fixed manifest")
    selection = validate_pilot(json.loads(Path(selection_path).read_text()),manifest,train.labels)
    if selection.get("kind") == "confirmation" and partition != "validation":
        raise ValueError("Confirmation extraction is validation-only")
    indices = selection["partitions"][partition]["indices"]
    if not 1 <= len(indices) <= 1000:
        raise ValueError("Pretrained extraction is bounded at 1,000 selected images")
    model, metadata = load_pretrained(weights_path)
    start = perf_counter()
    features, labels = [], []
    for offset in range(0,len(indices),batch_size):
        batch = [train[index] for index in indices[offset:offset+batch_size]]
        images = torch.stack([image for image,_ in batch])
        features.append(pretrained_embeddings(model,images))
        labels.extend(label for _,label in batch)
    values = torch.cat(features)
    identity = dict(format_version=1,dataset="CIFAR-10",
        manifest_sha256=manifest["manifest_sha256"],selection_sha256=selection["selection_sha256"],
        selection_kind=selection.get("kind","pilot"),partition=partition,official_split="train",
        indices=indices,feature_metadata=metadata,
        features_sha256=hashlib.sha256(values.contiguous().numpy().tobytes()).hexdigest(),
        extraction_settings=dict(limit=len(indices),batch_size=batch_size),
        runtime=dict(torch=str(torch.__version__),cpu_threads=torch.get_num_threads()))
    if output is not None:
        atomic_checkpoint(Path(output),identity | dict(features=values,labels=torch.tensor(labels,dtype=torch.long)))
    return identity | dict(count=len(indices),feature_dim=512,finite=True,
        extraction_seconds=perf_counter()-start,test_evaluated=False,
        limitation="Frozen transfer features only; no classifier was fitted or scored")


def main():
    parser = argparse.ArgumentParser(description="Extract an offline frozen ResNet-18 transfer reference")
    parser.add_argument("root",type=Path)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--weights",type=Path,required=True,help="Already downloaded official local weights; no network")
    parser.add_argument("--partition",choices=("train","validation"),required=True)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    try:
        report=extract_pretrained(args.root,args.manifest,args.selection,args.weights,
            partition=args.partition,batch_size=args.batch_size,output=args.output)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
