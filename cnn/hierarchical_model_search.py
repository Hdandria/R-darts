import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from hierarchical_operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES_2
from genotypes import Genotype
import numpy as np

class MixedOp(nn.Module):

  def __init__(self, C, stride, primitive=PRIMITIVES_2, op_list = OPS):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.primitive=primitive
    self.op_list = OPS
    for primitive in self.primitive:
      op = op_list[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)


  def forward(self, s0, s1, weights, drop_prob=0.2):
    s=0
    l = []
    #print("\nWOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO\n")
    for w, op in zip(weights, self._ops):

        a = w * op(s0, s1, drop_prob)
        # print("w:", w)
        # print("op: ", type(op))
        # print("result: ", a.shape)
        l.append(a)
    # print("\nWOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO\n")

    return sum(l)

class Cell2(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell2, self).__init__()
    self.C_prev_prev=C_prev_prev
    self.C_prev = C_prev
    self.C = C

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

    self.reduction = reduction
    if reduction:
      self._steps=1
    else:
      self._steps = steps
    self._multiplier = multiplier
    
    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range((2+i)*(1+i)//2):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, PRIMITIVES_2, OPS_2)
        self._ops.append(op)

  def choose_all_pairs_from_list(self, l):
    result = []
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            result.append((l[i],l[j]))
    return result
  def show_state(self):

    print("C_prev_prev: ", self.C_prev_prev)
    print("C_prev: ", self.C_prev)
    print("C : ", self.C)
    print("reduction: ", self.reduction)
    print("B: ", self._steps)


  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](t_1, t_2, weights[offset+j]) for j, (t_1, t_2) in enumerate(self.choose_all_pairs_from_list(states)))
      offset += (i+2)*(i+1)//2
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

class Network2(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network2, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    #print("C_curr when building stem: ", C_curr)
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        #C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell2(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def show_state(self):
    pass
    # print("Now showing the list of cells")
    # for i, cell in enumerate(self.cells):
    #   # print(f"Now showing cell {i} of the network:")
    #   # print(type(cell))
    #   # cell.show_state()
    #   # print("_______________________________________\n")

  def new(self):
    model_new = Network2(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      # print(f'we are forwarding cell {i}')
      # print(f"s0 shape: {s0.shape}")
      # print(f"s1 shape: {s1.shape}")
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range((2+i)*(i+1)//2))
    num_ops = len(PRIMITIVES_2)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters


  def _from_pair_to_index(self, i,j, size):
    return (size-2)*i-(i)*(i-1)//2+j-1

  def _from_index_to_pair(self, idx, size):
    s=0
    i=0
    while(s+(size-1)-i < idx and i<size):
        s+=size-1-i
        i+=1
    return i, idx-s+i+1

  def genotype(self):
    def _parse(weights):
      gene = []
      start = 0
      for i in range(self._steps):
        n = (i+2)*(i+1)//2
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(n), key=lambda x: -max(W[x]))[0]
        t1, t2 = self._from_index_to_pair(edges, i+2)
        op_idx = np.argmax(W[edges])
        op_name = PRIMITIVES_2[op_idx]
        gene.append(((t1,t2), op_name))
        start = end
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def genotype_random(self):
    def _parse_random(weights):
      gene = []
      start = 0
      for i in range(self._steps):
        n = (i+2)*(i+1)//2
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(n), key=lambda x: -(np.random.choice(W[x], p=W[x])))[0]
        t1, t2 = self._from_index_to_pair(edges, i+2)
        op_idx = np.random.choice(range(len(W[edges])), p=W[edges])
        op_name = PRIMITIVES_2[op_idx]
        gene.append(((t1, t2), op_name))
        start = end
      return gene

    gene_normal = _parse_random(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse_random(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype