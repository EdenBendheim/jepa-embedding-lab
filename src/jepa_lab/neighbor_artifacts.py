"""Select and persist frozen kNN state bound to exact verified feature sources."""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .cifar10_train import atomic_checkpoint
from .neighbors import NeighborConfig,NeighborIndex,fit_neighbors
from .probe_artifacts import probe_context
from .probes import validate_probe_data

FIELDS=('mean','scale','vectors','labels')


def select_neighbors(train,validation,candidates):
    context=index_context(train,validation)
    if (not isinstance(candidates,(list,tuple)) or not 1<=len(candidates)<=8
            or any(not isinstance(c,NeighborConfig) for c in candidates) or len(set(candidates))!=len(candidates)):
        raise ValueError('Use 1-8 distinct neighbor configurations')
    indices=[];rows=[]
    for config in candidates:
        index=fit_neighbors(train.features,train.labels,config);indices.append(index)
        rows.append(dict(config=asdict(config),validation=index.score(validation.features,validation.labels)))
    chosen=min(range(len(rows)),key=lambda i:(-rows[i]['validation']['accuracy'],i))
    return indices[chosen],dict(candidates=rows,selected_candidate=chosen,
        selection_rule='Highest tuning accuracy, then declared order',context=context,test_evaluated=False)


def content_hash(tensor):return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def index_context(train,validation):
    context=probe_context(train,validation)
    for source in (train,validation):
        for field in ('features','labels'):
            expected=getattr(source,field+'_sha256')
            if expected is not None and content_hash(getattr(source,field))!=expected:
                raise ValueError('Verified neighbor source changed after loading')
    return context|dict(fitting_content_sha256=content_hash(train.features),tuning_content_sha256=content_hash(validation.features),
        fitting_labels_sha256=content_hash(train.labels),tuning_labels_sha256=content_hash(validation.labels))


def fingerprint(state,config,context):
    h=hashlib.sha256(json.dumps(dict(config=config,context=context),sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    for field in FIELDS:h.update(field.encode());h.update(state[field].contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_state(state,train,config):
    n,d=train.features.shape
    if not isinstance(state,dict) or set(state)!=set(FIELDS):raise ValueError('Neighbor state is incomplete')
    for field,shape in (('mean',(d,)),('scale',(d,)),('vectors',(n,d)),('labels',(n,))):
        t=state[field]
        if (not isinstance(t,torch.Tensor) or t.device.type!='cpu' or t.requires_grad or tuple(t.shape)!=shape
                or t.dtype!=(torch.long if field=='labels' else torch.float32) or not torch.isfinite(t).all()):
            raise ValueError('Invalid saved neighbor tensor')
    if config.k>n or (state['scale']<=0).any() or not torch.equal(state['labels'],train.labels):
        raise ValueError('Neighbor counts, scales or fitting labels differ')
    expected=F.normalize((train.features-state['mean'])/state['scale'],dim=1,eps=1e-12)
    if not torch.equal(expected,state['vectors']):raise ValueError('Saved index vectors differ from their exact fitting source')


def save_neighbors(path,index,train,validation):
    if not isinstance(index,NeighborIndex):raise ValueError('Save a fitted neighbor index')
    context=index_context(train,validation);config=asdict(index.config)
    state={field:getattr(index,field).detach().clone() for field in FIELDS};validate_state(state,train,index.config)
    scores=dict(validation=index.score(validation.features,validation.labels))
    saved=dict(format_version=1,kind='frozen cosine neighbor index',context=context,config=config,state=state,
        state_sha256=fingerprint(state,config,context),scores=scores,runtime=dict(torch=str(torch.__version__),cpu_threads=torch.get_num_threads()),test_evaluated=False)
    atomic_checkpoint(Path(path),saved)
    return load_neighbors(path,train,validation)[1]


def load_neighbors(path,train,validation):
    context=index_context(train,validation);path=Path(path)
    if path.stat().st_size>20*1024*1024:raise ValueError('Neighbor artifacts are bounded to 20 MiB')
    payload=path.read_bytes();saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    if (not isinstance(saved,dict) or type(saved.get('format_version')) is not int or saved['format_version']!=1
            or saved.get('kind')!='frozen cosine neighbor index' or saved.get('context')!=context
            or saved.get('runtime')!=dict(torch=str(torch.__version__),cpu_threads=torch.get_num_threads())
            or saved.get('test_evaluated') is not False):raise ValueError('Neighbor artifact sources, runtime or identity differ')
    config=NeighborConfig(**saved['config']);state=saved['state'];validate_state(state,train,config)
    if saved['state_sha256']!=fingerprint(state,asdict(config),context):raise ValueError('Neighbor state fingerprint differs')
    index=NeighborIndex(*(state[field].clone() for field in FIELDS),config)
    scores=dict(validation=index.score(validation.features,validation.labels))
    if scores!=saved['scores']:raise ValueError('Saved neighbor tuning scores do not reproduce')
    return index,dict(context=context,config=asdict(config),scores=scores,state_sha256=saved['state_sha256'],
        artifact_sha256=hashlib.sha256(payload).hexdigest(),test_evaluated=False)
