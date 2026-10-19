"""Paired frozen-classifier errors on exact reserved confirmation sources."""
import argparse
import json
from pathlib import Path
import torch

from .cifar10 import CIFAR10Binary,make_manifest
from .compare import write_report
from .confirmation import verify_confirmation_sources
from .diagnostics import paired_diagnostics
from .features import load_features
from .probe_artifacts import load_probe
from .selection import validate_pilot,validate_confirmation


def compare_confirmed_probes(baseline,candidate,baseline_path,candidate_path,selection):
    for triple in (baseline,candidate):
        if not isinstance(triple,(list,tuple)) or len(triple)!=3:raise ValueError('Each classifier needs original train/tuning and fresh feature sets')
        verify_confirmation_sources(*triple,selection)
    for left,right in zip(baseline,candidate):
        if (left.indices!=right.indices or not torch.equal(left.labels,right.labels)
                or left.manifest_sha256!=right.manifest_sha256 or left.selection_sha256!=right.selection_sha256):
            raise ValueError('Paired confirmation requires identical ordered rows and labels')
    before,before_summary=load_probe(baseline_path,*baseline[:2])
    after,after_summary=load_probe(candidate_path,*candidate[:2])
    with torch.inference_mode():
        a=before.logits(baseline[2].features).argmax(1);b=after.logits(candidate[2].features).argmax(1)
    return dict(version=1,task='Paired frozen-classifier confirmation',baseline_probe=before_summary,
        candidate_probe=after_summary,baseline_fresh=baseline[2].summary(),candidate_fresh=candidate[2].summary(),
        paired_confirmation=paired_diagnostics(baseline[2].labels,a,b),
        baseline_accuracy=float((a==baseline[2].labels).float().mean()),
        candidate_accuracy=float((b==candidate[2].labels).float().mean()),
        refitted=False,settings_selected=False,test_evaluated=False,
        limitation='Descriptive paired errors on the same declared confirmation rows. No independent training-seed uncertainty or official test claim; using these results to tune makes them observed.')


def run_paired_confirmation(root,manifest_path,reference_path,confirmation_path,baseline_paths,candidate_paths):
    if any(not isinstance(paths,(list,tuple)) or len(paths)!=4 for paths in (baseline_paths,candidate_paths)):
        raise ValueError('Each arm needs TRAIN TUNING FRESH PROBE paths')
    manifest=json.loads(Path(manifest_path).read_text());train,test=CIFAR10Binary(root),CIFAR10Binary(root,train=False)
    if manifest!=make_manifest(train,test,seed=manifest['seed']):raise ValueError('Dataset differs from fixed manifest')
    reference=validate_pilot(json.loads(Path(reference_path).read_text()),manifest,train.labels)
    if reference.get('kind')=='confirmation':raise ValueError('Use original tuning pilot as reference')
    selected=validate_confirmation(json.loads(Path(confirmation_path).read_text()),manifest,train.labels)
    triples=[tuple(load_features(path,manifest=manifest,selection=reference if i<2 else selected,
        training_labels=train.labels,partition='train' if i==0 else 'validation') for i,path in enumerate(paths[:3]))
        for paths in (baseline_paths,candidate_paths)]
    return compare_confirmed_probes(*triples,baseline_paths[3],candidate_paths[3],selected)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    for name in ('manifest','reference-selection','confirmation-selection','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('baseline','candidate'):p.add_argument('--'+name,type=Path,nargs=4,required=True,metavar=('TRAIN','TUNING','FRESH','PROBE'))
    args=p.parse_args();torch.set_num_threads(1)
    try:
        report=run_paired_confirmation(args.root,args.manifest,args.reference_selection,args.confirmation_selection,args.baseline,args.candidate)
        write_report(args.output,report)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:p.error(str(exc))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
