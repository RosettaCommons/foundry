import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from functools import partial
import numpy as np
from torch import relu

from rf2aa.debug import debug_nans
from rf2aa.model.layers.SE3_network import FullyConnectedSE3, FullyConnectedSE3_noR
from rf2aa.model.layers.structure_bias import structure_bias_factory
from rf2aa.model.layers.Attention_module import BiasedAxialAttention, FeedForwardLayer, MSAColAttention, \
    MSARowAttentionWithBias, TriangleMultiplication, MSAColGlobalAttention, \
    OldMSAColAttention, OldMSAColGlobalAttention, BiasedUntiedAxialAttention, TriangleAttention
from rf2aa.model.layers.outer_product import OuterProductMean # need to code this correctly
from rf2aa.training.checkpoint import create_custom_forward
from rf2aa.util_module import Dropout
from rf2aa.model.AF3_structure import AtomAttentionEncoder, AtomAttentionDecoder


class NonEquivariantAtomEncoder(nn.Module):

    def __init__(self, block_params):
        super().__init__()
        # c_atom, c_atompair, c_token = block_params.c_atom_pair, block_params.c_atom, block_params.c_token
        self.model = AtomAttentionEncoder(**block_params)


class NonEquivariantAtomDecoder(nn.Module):

    def __init__(self, block_params):
        super().__init__()
        # c_atom, c_atompair, c_token = block_params.c_atom_pair, block_params.c_atom, block_params.c_token
        self.model = AtomAttentionDecoder(**block_params)

