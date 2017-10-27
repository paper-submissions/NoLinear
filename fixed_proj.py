import torch.nn as nn
import math
import torch
from torch.autograd import Variable
from scipy.linalg import hadamard


class HadamardProj(nn.Module):

    def __init__(self, input_size, output_size, bias=True, init_scale=10):

        super(HadamardProj, self).__init__()
        if init_scale is not None:
            self.weight = nn.Parameter(torch.Tensor(1).fill_(init_scale))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size).fill_(0))
        self.sz = 2 ** int(math.ceil(math.log(max(input_size, output_size), 2)))
        mat = torch.from_numpy(hadamard(self.sz))
        self.proj = Variable(mat, requires_grad=False)
        self.output_size = output_size
        self.input_size = input_size
        self.coeff = self.input_size ** 0.5

    def forward(self, x):
        x = x[:, :self.proj.size(1)]
        x = x / x.norm(2, -1, keepdim=True)
        w = self.proj.type_as(x)
        dist = nn.functional.linear(x, w)[:, :self.output_size]
        out = dist / self.coeff
        if hasattr(self, 'weight'):
            out = out * self.weight
        if hasattr(self, 'bias'):
            out = out + self.bias.view(1, -1)
        return out


class Proj(nn.Module):

    def __init__(self, input_size, output_size, bias=True, init_scale=10):
        super(Proj, self).__init__()
        if init_scale is not None:
            self.weight = nn.Parameter(torch.Tensor(1).fill_(init_scale))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size).fill_(0))
        self.proj = Variable(torch.Tensor(
            output_size, input_size), requires_grad=False)
        torch.manual_seed(123)
        nn.init.orthogonal(self.proj)

    def forward(self, x):
        w = self.proj.type_as(x)
        x = x / x.norm(2, -1, keepdim=True)
        out = nn.functional.linear(x, w)
        if hasattr(self, 'weight'):
            out = out * self.weight
        if hasattr(self, 'bias'):
            out = out + self.bias.view(1, -1)
        return out
