import torch
import torch.nn as nn

from model import Cell as InferCell
from genotypes import *
from operations import FactorizedReduce


PRIMITIVES_2 = [
  #'skip_connect',
  'best_cell'
  'cell1',
  'cell2',
  'cell3',
  'cell4',
  'cell5',
]

OPS_2 = {
  #'skip_connect' : lambda C, stride, affine: Identity_dual(stride),
  'best_cell': lambda C, stride, affine: InferCell_standardized(C, stride, affine, best_cell),
  'cell1': lambda C, stride, affine: InferCell_standardized(C, stride, affine, cell1),
  'cell2': lambda C, stride, affine: InferCell_standardized(C, stride, affine, cell2),
  'cell3': lambda C, stride, affine: InferCell_standardized(C, stride, affine, cell3),
  'cell4': lambda C, stride, affine: InferCell_standardized(C, stride, affine, cell4),
  'cell5': lambda C, stride, affine: InferCell_standardized(C, stride, affine, cell5),
}

def InferCell_standardized(C, stride, affine, cell):
  reduction = (stride==2)
  if reduction:
    return InferCell(cell.reduce, C, C, C, reduction = (stride==2), reduction_prev=False, scnd=True)
  else:
    return InferCell(cell.normal, C, C, C, reduction = (stride==2), reduction_prev=False, scnd=True)

class Identity_dual(nn.Module):

  def __init__(self, stride):
    super(Identity_dual, self).__init__()
    self.stride = stride

  def forward(self, s0, s1, drop_prob):
    if self.stride==2:
      s0 = FactorizedReduce(C, C)(s0)
      s1 = FactorizedReduce(C, C)(s1)
    return torch.cat([s0, s1], axis=1)
  def show_state(self):
    print("Identity cell, no state to show")