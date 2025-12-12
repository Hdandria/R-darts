from collections.abc import Callable

from genotypes import PRIMITIVES as DEFAULT_PRIMITIVES
from genotypes import Genotype
import numpy as np
from operations import OPS as DEFAULT_OPS
from operations import FactorizedReduce, ReLUConvBN
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedOp(nn.Module):
    def __init__(
        self,
        C: int,
        stride: int,
        primitive: list[str] = DEFAULT_PRIMITIVES,
        op_list: dict[str, Callable] = DEFAULT_OPS,
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
        primitives: list[str] = DEFAULT_PRIMITIVES,
        ops: dict[str, Callable] = DEFAULT_OPS,
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
        primitives: list[str] = DEFAULT_PRIMITIVES,
        ops: dict[str, Callable] = DEFAULT_OPS,
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

        self.alphas_normal = (1e-3 * torch.randn(k, num_ops)).requires_grad_(True)
        self.alphas_reduce = (1e-3 * torch.randn(k, num_ops)).requires_grad_(True)
        if torch.cuda.is_available():
            self.alphas_normal = self.alphas_normal.cuda().requires_grad_(True)
            self.alphas_reduce = self.alphas_reduce.cuda().requires_grad_(True)

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self) -> list[torch.Tensor]:
        return self._arch_parameters

    def genotype(self) -> Genotype:
        def _parse(weights: np.ndarray) -> list[tuple[str, int]]:
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

    def sample_genotypes(self, k: int = 1) -> list[Genotype]:
        """
        Samples k distinct genotypes from the architecture distribution.
        This provides diversity for the recursive search step.
        """
        genotypes = []
        for _ in range(k):
            gene_normal = self._sample_cell(self.alphas_normal)
            gene_reduce = self._sample_cell(self.alphas_reduce)

            concat = range(2 + self._steps - self._multiplier, self._steps + 2)
            genotype = Genotype(
                normal=gene_normal,
                normal_concat=concat,
                reduce=gene_reduce,
                reduce_concat=concat,
            )
            genotypes.append(genotype)
        return genotypes

    def _sample_cell(self, alphas) -> list[tuple[str, int]]:
        weights = F.softmax(alphas, dim=-1).data.cpu().numpy()
        gene = []
        n = 2
        start = 0
        none_idx = self.primitives.index("none")

        for _i in range(self._steps):
            end = start + n
            W = weights[start:end].copy()

            # 1. Edge Selection (Probabilistic)
            # We calculate an "existence probability" for each edge based on sum of non-none ops
            edge_scores = []
            for j in range(len(W)):
                prob_edge_active = np.sum(W[j]) - W[j][none_idx]
                edge_scores.append(prob_edge_active)

            # Normalize to valid probabilities
            edge_probs = np.array(edge_scores)
            sum_probs = np.sum(edge_probs)
            if sum_probs > 0:
                edge_probs = edge_probs / sum_probs
            else:
                # Fallback to uniform if all are none (unlikely)
                edge_probs = np.ones_like(edge_probs) / len(edge_probs)

            # Sample 2 edges without replacement
            selected_edges = np.random.choice(range(n), size=2, replace=False, p=edge_probs)

            # 2. Op Selection (Probabilistic) for selected edges
            for j in selected_edges:
                # Sample op from non-none ops
                op_probs = W[j].copy()
                op_probs[none_idx] = 0 # Mask 'none'
                sum_op_probs = np.sum(op_probs)

                if sum_op_probs > 0:
                    op_probs = op_probs / sum_op_probs
                    k_sampled = np.random.choice(range(len(op_probs)), p=op_probs)
                else:
                    # Fallback: pick any non-none op (should not happen if edge was selected)
                    k_sampled = np.random.choice([x for x in range(len(op_probs)) if x != none_idx])

                gene.append((self.primitives[k_sampled], int(j)))

            start = end
            n += 1

        # Sort by node index to match standard format
        gene.sort(key=lambda x: x[1])
        return gene
