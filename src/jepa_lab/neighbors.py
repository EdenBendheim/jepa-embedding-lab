"""Bounded frozen-feature cosine kNN with training-only standardization."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .probes import validate_probe_data,_values,_labels


@dataclass(frozen=True)
class NeighborConfig:
    k:int=5

    def __post_init__(self):
        if type(self.k) is not int or not 1<=self.k<=100:
            raise ValueError('Neighbor count must be an integer in 1-100')


@dataclass(frozen=True)
class NeighborIndex:
    mean:torch.Tensor
    scale:torch.Tensor
    vectors:torch.Tensor
    labels:torch.Tensor
    config:NeighborConfig

    @torch.inference_mode()
    def nearest(self,features):
        """Return fitting-row positions and cosine similarities, in stable rank order.

        Queries never change training normalization. Zero vectors have zero
        similarity, so ties resolve to fitting-row order.
        """
        _values(features,self.vectors.shape[1])
        normalized=F.normalize((features.detach()-self.mean)/self.scale,dim=1,eps=1e-12)
        if not torch.isfinite(normalized).all():raise ValueError('Query normalization is non-finite')
        positions=[];scores=[]
        for start in range(0,len(normalized),64):
            similarity=normalized[start:start+64]@self.vectors.T
            # Stable ties use fitting-row order; vote ties use the smallest class ID.
            nearest=similarity.argsort(dim=1,descending=True,stable=True)[:,:self.config.k]
            positions.append(nearest);scores.append(similarity.gather(1,nearest))
        return torch.cat(positions),torch.cat(scores)

    @torch.inference_mode()
    def predict(self,features):
        nearest,_=self.nearest(features)
        votes=F.one_hot(self.labels[nearest],num_classes=10).sum(1)
        return votes.argmax(1)

    def score(self,features,labels):
        _labels(labels,len(features));predictions=self.predict(features)
        counts=torch.bincount(labels*10+predictions,minlength=100).reshape(10,10)
        correct=int((predictions==labels).sum())
        return dict(count=len(labels),correct=correct,accuracy=correct/len(labels),confusion_matrix=counts.tolist())


def fit_neighbors(features,labels,config=NeighborConfig()):
    validate_probe_data(features,labels,fitting=True)
    if not isinstance(config,NeighborConfig) or config.k>len(features):
        raise ValueError('Use a validated neighbor count no larger than the fitting set')
    values=features.detach().clone()
    mean=values.double().mean(0).float();std=values.double().std(0,unbiased=False).float()
    scale=torch.where(std<1e-6,torch.ones_like(std),std)
    vectors=F.normalize((values-mean)/scale,dim=1,eps=1e-12)
    if not torch.isfinite(vectors).all():raise ValueError('Fitting normalization is non-finite')
    return NeighborIndex(mean,scale,vectors,labels.clone(),config)
