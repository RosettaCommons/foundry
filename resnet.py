import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

# pre-activation bottleneck resblock
class ResBlock2D_bottleneck(nn.Module):
    def __init__(self, n_c, kernel=3, dilation=1, p_drop=0.15):
        super(ResBlock2D_bottleneck, self).__init__()
        padding = self._get_same_padding(kernel, dilation)

        n_b = n_c // 2 # bottleneck channel
        
        layer_s = list()
        # pre-activation
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        layer_s.append(nn.ELU(inplace=True))
        # project down to n_b
        layer_s.append(nn.Conv2d(n_c, n_b, 1, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_b, affine=True, eps=1e-6))
        layer_s.append(nn.ELU(inplace=True))
        # convolution
        layer_s.append(nn.Conv2d(n_b, n_b, kernel, dilation=dilation, padding=padding, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_b, affine=True, eps=1e-6))
        layer_s.append(nn.ELU(inplace=True))
        # dropout
        layer_s.append(nn.Dropout(p_drop))
        # project up
        layer_s.append(nn.Conv2d(n_b, n_c, 1, bias=False))

        # make final layer initialize with zeros
        #nn.init.zeros_(layer_s[-1].weight)

        self.layer = nn.Sequential(*layer_s)

        self.reset_parameter()

    def reset_parameter(self):
        # zero-initialize final layer right before residual connection 
        nn.init.zeros_(self.layer[-1].weight)

    def _get_same_padding(self, kernel, dilation):
        return (kernel + (kernel - 1) * (dilation - 1) - 1) // 2

    def forward(self, x):
        out = self.layer(x)
        return x + out

class ResidualNetwork(nn.Module):
    def __init__(self, n_block, n_feat_in, n_feat_block, n_feat_out, 
                 dilation=[1,2,4,8], p_drop=0.15):
        super(ResidualNetwork, self).__init__()


        layer_s = list()
        # project to n_feat_block
        if n_feat_in != n_feat_block:
            layer_s.append(nn.Conv2d(n_feat_in, n_feat_block, 1, bias=False))

        # add resblocks
        for i_block in range(n_block):
            d = dilation[i_block%len(dilation)]
            res_block = ResBlock2D_bottleneck(n_feat_block, kernel=3, dilation=d, p_drop=p_drop)
            layer_s.append(res_block)

        if n_feat_out != n_feat_block:
            # project to n_feat_out
            layer_s.append(nn.Conv2d(n_feat_block, n_feat_out, 1))
        
        self.layer = nn.Sequential(*layer_s)
    
    def forward(self, x):
        return self.layer(x)
