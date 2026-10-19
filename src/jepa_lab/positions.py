"""Defined-here row/column sinusoidal positions for controlled model comparisons."""
import torch


def sincos_2d(rows:int,cols:int,dimension:int) -> torch.Tensor:
    """Row-major [sin(row*w), cos(row*w), sin(col*w), cos(col*w)] float32 grid.

    Frequencies are 10000**(-j/(dimension/4)), j=0..dimension/4-1.
    This is a fixed experimental encoding, not a claim to reproduce I-JEPA.
    """
    if (any(type(v) is not int or v<1 for v in (rows,cols,dimension))
            or rows*cols>4096 or dimension>4096 or dimension%4):
        raise ValueError('Positions require positive integer grid/dimension, at most 4096 tokens/dimensions and width divisible by four')
    frequencies=10000.**(-torch.arange(dimension//4,dtype=torch.float64)/(dimension//4))
    r=torch.arange(rows,dtype=torch.float64).repeat_interleave(cols)[:,None]*frequencies
    c=torch.arange(cols,dtype=torch.float64).repeat(rows)[:,None]*frequencies
    return torch.cat((r.sin(),r.cos(),c.sin(),c.cos()),dim=1).float()
