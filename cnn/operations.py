from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

# Forward declaration or imports if needed
# from .genotypes import Genotype # Avoid circular import if genotypes imports operations

class ReLUConvBN(nn.Module):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int, affine: bool = True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DilConv(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        affine: bool = True,
    ):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SepConv(nn.Module):
    def __init__(
        self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int, affine: bool = True
    ):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Zero(nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in: int, C_out: int, affine: bool = True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


OPS: dict[str, Callable[[int, int, bool], nn.Module]] = {
    "none": lambda C, stride, affine: Zero(stride),
    "avg_pool_3x3": lambda C, stride, affine: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False
    ),
    "max_pool_3x3": lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    "skip_connect": lambda C, stride, affine: Identity()
    if stride == 1
    else FactorizedReduce(C, C, affine=affine),
    "sep_conv_3x3": lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    "sep_conv_5x5": lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    "sep_conv_7x7": lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    "dil_conv_3x3": lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    "dil_conv_5x5": lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    "conv_7x1_1x7": lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine),
    ),
}

class CellOp(nn.Module):
    def __init__(self, genotype: Any, C: int, stride: int, affine: bool):
        super().__init__()
        self.stride = stride
        # Logic adapted from model.Cell but simplified for an operation context

        if stride == 2:
            op_names, indices = zip(*genotype.reduce, strict=True)
            concat = genotype.reduce_concat
            reduction = True
        else:
            op_names, indices = zip(*genotype.normal, strict=True)
            concat = genotype.normal_concat
            reduction = False

        self._compile(C, list(op_names), list(indices), concat, reduction, affine)

    def _compile(
        self,
        C: int,
        op_names: list[str],
        indices: list[int],
        concat: list[int],
        reduction: bool,
        affine: bool,
    ):
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices, strict=True):
            # Internal stride logic of DARTS cell:
            # stride=2 if reduction and index < 2 else 1
            inp_stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, inp_stride, affine)
            self._ops.append(op)
        self._indices = indices

        # Final projection to ensure output channels = C
        # A standard DARTS cell outputs (multiplier * C) channels.
        # But a primitive operation is expected to output C (or C * stride_factor?
        # No, usually C_out=C_in for stride 1 ops in DARTS search space, except factorized reduce)
        # FactorizedReduce changes spatial dim but keeps C_out = C_in (in argument logic).
        # Wait, OPS signatures are (C, stride, affine). C is input channels.
        # Usually these ops output C channels as well (SepConv(C, C...)).

        self.project_out = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C * self.multiplier, C, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C, affine=affine)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We treat x as both s0 and s1
        # If stride=2 (reduction), s0 and s1 should probably be preprocessed?
        # But OPS inside will handle the stride.
        s0 = s1 = x
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states.append(s)
        out = torch.cat([states[i] for i in self._concat], dim=1)
        return self.project_out(out)
