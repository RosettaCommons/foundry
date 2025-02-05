
import torch.nn as nn

from rf2aa.model.AF3_structure import AtomAttentionDecoder, AtomAttentionEncoder


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
