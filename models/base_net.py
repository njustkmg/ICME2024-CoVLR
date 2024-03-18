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
        # if self.weight.fast is not None :
        # if True:
        #     out = F.linear(x, self.weight.fast, self.bias.fast)
        # elif self.weight.fast is not None:
        #     out = F.linear(x, self.weight.fast)
        # else:
        out = super(LinearML, self).forward(x)
        return out


class LayerNormML(nn.LayerNorm):
    def __init__(self,normalized_shape, eps=1e-05, elementwise_affine=True):
        super(LayerNormML, self).__init__(normalized_shape,eps = eps,elementwise_affine = elementwise_affine)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.layer_norm(x,weight=self.weight.fast,bias=self.bias.fast)
        else:
            out = super(LayerNormML,self).forward(x)
        return out



