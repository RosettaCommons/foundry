import torch
import torch.nn as nn
from icecream import ic
import inspect

import sys, os
script_dir = os.path.dirname(os.path.realpath(__file__))+'/'
sys.path.insert(0,script_dir+'SE3Transformer')
#sys.path.insert(0, '/home/ahern/projects/rf_diffusion/RF2-allatom/rf2aa/SE3Transformer') # jue commented this -- might need to uncomment

from rf2aa.util_module import init_lecun_normal_param

from se3_transformer.model import SE3Transformer
from se3_transformer.model.fiber import Fiber
se3_transformer_path = inspect.getfile(SE3Transformer)
se3_fiber_path = inspect.getfile(Fiber)
#ic(se3_transformer_path, se3_fiber_path)
assert 'rf2aa' in se3_transformer_path

class SE3TransformerWrapper(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers=2, num_channels=32, num_degrees=3, n_heads=4, div=4,
                 l0_in_features=32, l0_out_features=32,
                 l1_in_features=3, l1_out_features=2,
                 num_edge_features=32):
        super().__init__()
        # Build the network
        self.l1_in = l1_in_features
        self.l1_out = l1_out_features
        #
        fiber_edge = Fiber({0: num_edge_features})
        if l1_out_features > 0:
            if l1_in_features > 0:
                fiber_in = Fiber({0: l0_in_features, 1: l1_in_features})
                fiber_hidden = Fiber.create(num_degrees, num_channels)
                fiber_out = Fiber({0: l0_out_features, 1: l1_out_features})
            else:
                fiber_in = Fiber({0: l0_in_features})
                fiber_hidden = Fiber.create(num_degrees, num_channels)
                fiber_out = Fiber({0: l0_out_features, 1: l1_out_features})
        else:
            if l1_in_features > 0:
                fiber_in = Fiber({0: l0_in_features, 1: l1_in_features})
                fiber_hidden = Fiber.create(num_degrees, num_channels)
                fiber_out = Fiber({0: l0_out_features})
            else:
                fiber_in = Fiber({0: l0_in_features})
                fiber_hidden = Fiber.create(num_degrees, num_channels)
                fiber_out = Fiber({0: l0_out_features})
        
        self.se3 = SE3Transformer(num_layers=num_layers,
                                  fiber_in=fiber_in,
                                  fiber_hidden=fiber_hidden,
                                  fiber_out = fiber_out,
                                  num_heads=n_heads,
                                  channels_div=div,
                                  fiber_edge=fiber_edge,
                                  populate_edge="arcsin",
                                  final_layer="lin",
                                  use_layer_norm=True)

        self.reset_parameter()

    def reset_parameter(self):

        # make sure linear layer before ReLu are initialized with kaiming_normal_
        for n, p in self.se3.named_parameters():
            if "bias" in n:
                nn.init.zeros_(p)
            elif len(p.shape) == 1:
                continue
            else:
                if "radial_func" not in n:
                    p = init_lecun_normal_param(p) 
                else:
                    if "net.6" in n:
                        nn.init.zeros_(p)
                    else:
                        nn.init.kaiming_normal_(p, nonlinearity='relu')
        
        # make last layers to be zero-initialized
        #self.se3.graph_modules[-1].to_kernel_self['0'] = init_lecun_normal_param(self.se3.graph_modules[-1].to_kernel_self['0'])
        #self.se3.graph_modules[-1].to_kernel_self['1'] = init_lecun_normal_param(self.se3.graph_modules[-1].to_kernel_self['1'])
        #nn.init.zeros_(self.se3.graph_modules[-1].to_kernel_self['0'])
        #nn.init.zeros_(self.se3.graph_modules[-1].to_kernel_self['1'])
        nn.init.zeros_(self.se3.graph_modules[-1].weights['0'])
        if self.l1_out > 0:
            nn.init.zeros_(self.se3.graph_modules[-1].weights['1'])

    def forward(self, G, type_0_features, type_1_features=None, edge_features=None):
        if self.l1_in > 0:
            node_features = {'0': type_0_features, '1': type_1_features}
        else:
            node_features = {'0': type_0_features}
        edge_features = {'0': edge_features}
        return self.se3(G, node_features, edge_features)
