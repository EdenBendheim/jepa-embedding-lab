"""Compare frozen neighborhoods on identical observed rows, without model fitting."""
import argparse
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import re
from statistics import mean

import torch

from .compare import write_report
from .explorer import load_exploration
from .explorer_html import validate_report


def compare_explorations(reports, *, depth=None):
    if (not isinstance(reports,dict) or not 2<=len(reports)<=4
            or any(not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',name) for name in reports)):
        raise ValueError('Use 2-4 distinctly named representations (letters, digits, underscore, hyphen)')
    for report in reports.values():validate_report(report)
    reference=next(iter(reports.values()))
    for report in reports.values():
        for partition in ('training','validation'):
            for key in ('manifest_sha256','selection_sha256','selection_kind','partition','count'):
                if key not in reference[partition] or report[partition].get(key)!=reference[partition][key]:
                    raise ValueError('Neighborhood comparisons require matching split identities/counts')
        identities=lambda r:[(q['index'],q['label']) for q in r['queries']]
        if identities(report)!=identities(reference):raise ValueError('Representations must inspect the same ordered queries and labels')
    available=min(len(q['neighbors']) for r in reports.values() for q in r['queries'])
    depth=available if depth is None else depth
    if type(depth) is not int or not 1<=depth<=available:raise ValueError('Comparison depth exceeds a restored neighbor count')
    pairs=[]
    for left,right in combinations(reports,2):
        rows=[]
        for a,b in zip(reports[left]['queries'],reports[right]['queries']):
            aa=[n['index'] for n in a['neighbors'][:depth]];bb=[n['index'] for n in b['neighbors'][:depth]]
            shared=set(aa)&set(bb)
            # A shared official training row must have the same immutable class.
            labels={n['index']:n['label'] for n in a['neighbors']}
            if any(n['index'] in labels and labels[n['index']]!=n['label'] for n in b['neighbors']):
                raise ValueError('Shared training-row labels differ')
            rows.append(dict(query_index=a['index'],shared_count=len(shared),overlap_fraction=len(shared)/depth,
                shared_indices=[i for i in aa if i in shared],left_only=[i for i in aa if i not in shared],
                right_only=[i for i in bb if i not in shared]))
        pairs.append(dict(left=left,right=right,mean_overlap_fraction=mean(q['overlap_fraction'] for q in rows),queries=rows))
    return dict(version=1,task='Frozen neighborhood comparison',depth=depth,representations=deepcopy(reports),
        pairs=pairs,test_evaluated=False,limitation='Descriptive overlap on already observed tuning rows, not accuracy or new quality evidence. '
            'Common depth is bounded by each restored k; scores across representations need not be comparable. No retuning or official test evaluation.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    for name in ('manifest','selection','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',nargs=4,action='append',required=True,metavar=('NAME','TRAIN_PT','VALIDATION_PT','INDEX_PT'))
    p.add_argument('--query-index',type=int,nargs='+');p.add_argument('--depth',type=int);p.add_argument('--html',type=Path)
    args=p.parse_args();torch.set_num_threads(1)
    try:
        if not 2<=len(args.arm)<=4 or len({arm[0] for arm in args.arm})!=len(args.arm):raise ValueError('Use 2-4 distinct arm names')
        reports={name:load_exploration(args.root,args.manifest,args.selection,train,validation,index,query_indices=args.query_index)
            for name,train,validation,index in args.arm}
        report=compare_explorations(reports,depth=args.depth)
        if args.html:
            from .explorer_html import write_html
            write_html(args.html,report)
        write_report(args.output,report)
    except (ValueError,OSError,TypeError,KeyError,RuntimeError) as exc:p.error(str(exc))
    print(json.dumps(dict(output=str(args.output),representations=list(reports),query_count=len(next(iter(reports.values()))['queries']),depth=report['depth'])))


if __name__=='__main__':main()
