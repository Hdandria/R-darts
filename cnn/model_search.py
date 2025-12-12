import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Callable, Any, Tuple, Optional

from .genotypes import Genotype
from .operations import FactorizedReduce, ReLUConvBN, OPS as DEFAULT_OPS
from .genotypes import PRIMITIVES as DEFAULT_PRIMITIVES


class MixedOp(nn.Module):
    def __init__(
        self,
        C: int,
        stride: int,
        primitive: List[str] = DEFAULT_PRIMITIVES,
        op_list: Dict[str, Callable] = DEFAULT_OPS,
    ):
        super().__init__()
        self._ops = nn.ModuleList()
        self.primitive = primitive
        self.op_list = op_list
        for primitive_name in self.primitive:
            op = op_list[primitive_name](C, stride, False)
            if "pool" in primitive_name:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return sum(w * op(x) for w, op in zip(weights, self._ops, strict=True)) # type: ignore


class Cell(nn.Module):
    def __init__(
        self,
        steps: int,
        multiplier: int,
        C_prev_prev: int,
        C_prev: int,
        C: int,
        reduction: bool,
        reduction_prev: bool,
        primitives: List[str] = DEFAULT_PRIMITIVES,
        ops: Dict[str, Callable] = DEFAULT_OPS,
    ):
        super().__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, primitives, ops)
                self._ops.append(op)

    def forward(self, s0: torch.Tensor, s1: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for _i in range(self._steps):
            s = sum(
                self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s) # type: ignore

        return torch.cat(states[-self._multiplier :], dim=1)


class Network(nn.Module):
    def __init__(
        self,
        C: int,
        num_classes: int,
        layers: int,
        criterion: nn.Module,
        steps: int = 4,
        multiplier: int = 4,
        stem_multiplier: int = 3,
        primitives: List[str] = DEFAULT_PRIMITIVES,
        ops: Dict[str, Callable] = DEFAULT_OPS,
    ):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.primitives = primitives
        self.ops = ops

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                steps,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
                primitives,
                ops,
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self) -> "Network":
        model_new = Network(
            self._C,
            self._num_classes,
            self._layers,
            self._criterion,
            self._steps,
            self._multiplier,
            primitives=self.primitives,
            ops=self.ops,
        )
        if torch.cuda.is_available():
            model_new = model_new.cuda()
            
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters(), strict=True):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(input)
        for _i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self) -> None:
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(self.primitives)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops), requires_grad=True)
        if torch.cuda.is_available():
            self.alphas_normal = self.alphas_normal.cuda()
            self.alphas_reduce = self.alphas_reduce.cuda()
            
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self) -> List[torch.Tensor]:
        return self._arch_parameters

    def genotype(self) -> Genotype:
        def _parse(weights: np.ndarray) -> List[Tuple[str, int]]:
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(
                        W[x][k]
                        for k in range(len(W[x]))
                        if k != self.primitives.index("none")
                    ),
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != self.primitives.index("none"):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    assert k_best is not None
                    gene.append((self.primitives[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype

    def genotype_random(self) -> Genotype:
        def _parse_random(weights: np.ndarray) -> List[Tuple[str, int]]:
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -np.random.choice(W[x], p=W[x]))[:2]
                for edge in edges:
                    k_best = 0
                    while k_best == 0:
                        k_best = np.random.choice(range(len(W[edge])), p=W[edge])
                    gene.append((self.primitives[k_best], edge))
                start = end
                n += 1
            return gene

        gene_normal = _parse_random(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse_random(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype
