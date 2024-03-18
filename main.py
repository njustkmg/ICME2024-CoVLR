import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
# from utils import calc_coeff
# from torch_ops import grl_hook

class LinearML(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearML, self).__init__(in_features, out_features, bias=bias)
        self.weight.fast = None
        if bias :
            self.bias.fast = None
        self.bias_flag = bias

    def forward(self, x):
        # print('----------------------------------self.weight', self.weight)
        # print('----------------------------------self.weight.fast',self.weight.fast)
        # print(self.named_parameters())
        if self.weight.fast is not None :
        # if True:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        # elif self.weight.fast is not None:
        #     out = F.linear(x, self.weight.fast)
        else:
            out = super(LinearML, self).forward(x)
        return out
decoder = LinearML(10, 20, bias=False)
# bias = nn.Parameter(torch.zeros(20))
# decoder.bias = bias
print(decoder)
print(decoder.weight.fast)
print(decoder.bias)
print(decoder.bias_flag)