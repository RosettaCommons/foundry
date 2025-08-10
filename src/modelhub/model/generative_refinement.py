import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum

from modelhub.chemical import ChemicalData as ChemData
from modelhub.flow_matching import data_utils as du
from modelhub.model.RF3_structure import FourierEmbedding
from modelhub.model.layers.Attention_module import FeedForwardLayer
from modelhub.model.layers.SE3_network import (
    SE3TransformerWrapper,
)
from modelhub.util import is_atom, is_nucleic, is_protein
from modelhub.util_module import init_lecun_normal


def get_bondgraph(bonds, num_bonds, dist_matrix, idx, is_prot, is_na, is_atom):
    """Combine different sources of info to get bond-distance graph for all i,j pairs"""
    L = bonds.shape[1]

    # intra-prot/intra-na bonds
    bonds[:, torch.arange(L), :, torch.arange(L), :] = num_bonds.transpose(0, 1)

    # ligand bonds
    ia = is_atom.nonzero()[:, 0]
    bonds[:, ia[:, None], 1, ia[None, :], 1] = dist_matrix[ia[:, None], ia[None, :]].to(
        dtype=bonds.dtype
    )

    # we need to handle covalent bonds between residues
    # to reduce computational load only consider +/- 1 residue
    # prot-prot
    ii, jj = ChemData().protein_connect
    resmask = (
        is_prot[0, :-1] * is_prot[0, 1:] * ((idx[1:] - idx[:-1]) == 1)
    ).nonzero()[..., 0]
    bonds[:, resmask + 1, :, resmask, :] = (
        num_bonds[:, resmask + 1, ii : (ii + 1)]
        + num_bonds[:, resmask, :, jj : jj + 1]
        + 1
    ).transpose(0, 1)
    bonds[:, resmask, :, resmask + 1, :] = (
        num_bonds[:, resmask + 1, ii : (ii + 1)]
        + num_bonds[:, resmask, :, jj : jj + 1]
        + 1
    ).transpose(0, 1)
    # na-na
    ii, jj = ChemData().na_connect
    resmask = (is_na[0, :-1] * is_na[0, 1:] * ((idx[1:] - idx[:-1]) == 1)).nonzero()[
        ..., 0
    ]
    bonds[:, resmask + 1, :, resmask, :] = (
        num_bonds[:, resmask + 1, ii : (ii + 1)]
        + num_bonds[:, resmask, :, jj : jj + 1]
        + 1
    ).transpose(0, 1)
    bonds[:, resmask, :, resmask + 1, :] = (
        num_bonds[:, resmask + 1, ii : (ii + 1)]
        + num_bonds[:, resmask, :, jj : jj + 1]
        + 1
    ).transpose(0, 1)

    return bonds


#
def make_atom_graph(
    xyz,
    mask,
    is_prot,
    is_na,
    is_atom,
    idx,
    num_bonds,
    dist_matrix,
    top_k=24,
    max_nbonds_encode=8,
    max_nbonds_connect=3,
):
    """
    Build an atom level graph from a mixed residue/ligand pose
    Parameters of interest:
      max_nbonds_encode - edge features encode # bonds, max this number
      max_nbonds_connect - force connections between atoms this # bonds or fewer
    Ensure top_k is large enough for max_nbonds_connect:
      with max_nbonds_connect=3, ~15 atoms are brought in by bonds alone
      with max_nbonds_connect=2, ~11 atoms are brought in by bonds alone
      with max_nbonds_connect=1, ~4 atoms are brought in by bonds alone
    """
    import dgl

    B, L, A = xyz.shape[:3]
    device = xyz.device
    D = torch.norm(xyz[:, None, None, :, :] - xyz[:, :, :, None, None], dim=-1)
    mask2d = mask[:, :, :, None, None] * mask[:, None, None, :, :]

    bonds = torch.full_like(D, ChemData().MAX_BOND_DIST, dtype=num_bonds.dtype)
    bonds = get_bondgraph(bonds, num_bonds, dist_matrix, idx, is_prot, is_na, is_atom)

    # set D to _negative_ for close bonded
    # all missing-atom pairs will have D==0 so need to prefer these
    D[bonds <= max_nbonds_connect] = -1.0
    D[bonds == 0] = np.inf  # set D large for self
    D[~mask2d] = np.inf  # set D large for non-atoms

    # select top K neighbors for each atom
    # keep indices as batch/res/atm indices
    nmaxedge = torch.sum(mask) - 1  # most edges = num atoms - 1

    if top_k > nmaxedge:
        top_k = nmaxedge

    D_neigh, E_idx = torch.topk(
        D.reshape(B, L, A, -1), top_k, largest=False
    )  # shape of E_idx: (B, L, A, top_k)
    Eres, Eatm = torch.div(E_idx, A, rounding_mode="trunc"), E_idx % A
    bi, ri, ai = mask.nonzero(as_tuple=True)
    bi = bi[:, None].repeat(1, top_k).reshape(-1)
    ri = ri[:, None].repeat(1, top_k).reshape(-1)
    ai = ai[:, None].repeat(1, top_k).reshape(-1)
    rj, aj = Eres[mask].reshape(-1), Eatm[mask].reshape(-1)

    # on each edge, encode:
    #    a) 1-hot encode the number of bonds (up to maxbonds) separating each atom
    #    b) 1/D
    bonds = bonds[bi, ri, ai, rj, aj]
    bonds[bonds >= max_nbonds_encode] = max_nbonds_encode

    natm = torch.sum(mask)
    index = torch.zeros_like(mask, dtype=torch.long, device=device)
    index[mask] = torch.arange(natm, device=device)
    src = index[bi, ri, ai]
    tgt = index[bi, rj, aj]

    G = dgl.graph((src, tgt), num_nodes=natm).to(device)
    G.edata["rel_pos"] = xyz[bi, ri, ai] - xyz[bi, rj, aj]

    edge = torch.cat(
        [
            F.one_hot(bonds - 1),
            1 / (torch.norm(G.edata["rel_pos"], dim=-1, keepdim=True) + 1),
        ],
        dim=-1,
    )

    return G, edge


class BiasedSequenceAttention(nn.Module):
    def __init__(self, global_params, block_params):
        super(BiasedSequenceAttention, self).__init__()
        self.norm_state = nn.LayerNorm(global_params.d_state, bias=False)
        self.norm_pair = nn.LayerNorm(global_params.d_pair, bias=False)
        #
        self.to_q = nn.Linear(
            global_params.d_state,
            block_params.n_channels * block_params.n_heads,
            bias=False,
        )
        self.to_k = nn.Linear(
            global_params.d_state,
            block_params.n_channels * block_params.n_heads,
            bias=False,
        )
        self.to_v = nn.Linear(
            global_params.d_state,
            block_params.n_channels * block_params.n_heads,
            bias=False,
        )
        self.to_b = nn.Linear(global_params.d_pair, block_params.n_heads, bias=False)
        self.to_g = nn.Linear(
            global_params.d_state, block_params.n_channels * block_params.n_heads
        )
        self.to_out = nn.Linear(
            block_params.n_channels * block_params.n_heads,
            global_params.d_state,
            bias=False,
        )

        self.scaling = 1 / np.sqrt(block_params.n_channels)
        self.h = block_params.n_heads
        self.dim = block_params.n_channels
        self.transition = FeedForwardLayer(
            global_params.d_state, 4, p_drop=block_params.msa_transition_drop
        )

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        # nn.init.zeros_(self.to_out.weight)

    def forward(self, state, pair):  # TODO: make this as tied-attention
        B, L = state.shape[:2]
        #
        state = self.norm_state(state)
        pair = self.norm_pair(pair)

        query = self.to_q(state).reshape(B, L, self.h, self.dim)
        key = self.scaling * self.to_k(state).reshape(B, L, self.h, self.dim)
        value = self.to_v(state).reshape(B, L, self.h, self.dim)
        bias = self.to_b(pair)  # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(state))

        attn = einsum("bqhd,bkhd->bqkh", query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)

        out = einsum("bqkh,bkhd->bqhd", attn, value).reshape(B, L, -1)
        out = gate * out

        out = self.to_out(out)
        out = state + self.transition(out)

        return out


class GenerativeRefinement(nn.Module):
    def __init__(self, global_params, block_params) -> None:
        super(GenerativeRefinement, self).__init__()
        self.proj_atoms = nn.Linear(ChemData().NELTTYPES, global_params.d_state)

        self.proj_edge = nn.Linear(
            9, block_params.num_edge_features
        )  # max_bonds_encode+1, need to refactor

        self.norm_state_0 = nn.LayerNorm(global_params.d_state + global_params.d_msa)
        self.proj_state_0 = nn.Linear(
            global_params.d_state + global_params.d_msa, global_params.d_state
        )
        self.ff_state_0 = FeedForwardLayer(global_params.d_state, 2, zero_init=False)

        self.norm_pair_0 = nn.LayerNorm(global_params.d_pair)
        self.proj_pair_0 = nn.Linear(global_params.d_pair, global_params.d_pair)
        self.ff_pair_0 = FeedForwardLayer(global_params.d_pair, 2, zero_init=False)

        self.timestep_embedding_dim = 256
        self.fourier_embedding = FourierEmbedding(self.timestep_embedding_dim)
        self.norm_timstep_emb = nn.LayerNorm(self.timestep_embedding_dim)
        self.emb_timestep = nn.Linear(
            self.timestep_embedding_dim, global_params.d_state, bias=False
        )

        self.proj_state_1 = nn.Linear(
            global_params.d_state, global_params.d_state, bias=False
        )
        self.proj_state_2 = nn.Linear(
            global_params.d_state, global_params.d_state, bias=False
        )
        self.norm_state_1 = nn.LayerNorm(global_params.d_state)
        self.norm_state_2 = nn.LayerNorm(global_params.d_state)

        self.sigma_data = 16  # expose as parameter

        self.atom_encoder = nn.ModuleList(
            [
                SE3TransformerWrapper(
                    num_layers=block_params.num_layers,
                    num_channels=block_params.num_channels,
                    num_degrees=block_params.num_degrees,
                    n_heads=block_params.n_heads,
                    div=block_params.div,
                    l0_in_features=block_params.l0_in_features,
                    l0_out_features=block_params.l0_out_features,
                    l1_in_features=1,
                    l1_out_features=1,
                    num_edge_features=block_params.num_edge_features,
                    compute_gradients=True,
                )
            ]
        )

        self.token_processing = nn.ModuleList(
            [
                BiasedSequenceAttention(global_params, block_params)
                for i in range(block_params.num_attention_layers)
            ]
        )

        self.atom_decoder = nn.ModuleList(
            [
                SE3TransformerWrapper(
                    num_layers=block_params.num_layers,
                    num_channels=block_params.num_channels,
                    num_degrees=block_params.num_degrees,
                    n_heads=block_params.n_heads,
                    div=block_params.div,
                    l0_in_features=block_params.l0_in_features,
                    l0_out_features=0,
                    l1_in_features=1,
                    l1_out_features=1,
                    num_edge_features=block_params.num_edge_features,
                    compute_gradients=True,
                )
            ]
        )

    def _unpack_latents(self, latent_feats):
        msa, pair, state = (
            latent_feats["msa"],
            latent_feats["pair"],
            latent_feats["state"],
        )

        seq_unmasked = latent_feats["seq_unmasked"]
        allatom_mask = ChemData().allatom_mask.to(state.device)
        is_valid_atom = allatom_mask[seq_unmasked]
        num_bonds = ChemData().num_bonds.to(state.device)
        num_bonds_sequence = num_bonds[seq_unmasked]
        xyz = latent_feats["trans_t"][0]
        t = latent_feats["t"]

        is_atomized = is_atom(seq_unmasked)
        is_prot = is_protein(seq_unmasked)
        is_na = is_nucleic(seq_unmasked)
        dist_matrix = latent_feats["dist_matrix"][0]
        idx = latent_feats["idx"][0]

        return (
            msa,
            pair,
            state,
            seq_unmasked,
            is_valid_atom,
            num_bonds_sequence,
            xyz,
            t,
            is_atomized,
            is_prot,
            is_na,
            dist_matrix,
            idx,
        )

    def _embed_1d(self, latent_feats):
        seq_unmasked = latent_feats["seq_unmasked"]

        # feature set 1: element
        elts = ChemData().aa2eltidx.to(seq_unmasked.device)
        elts = elts[seq_unmasked]
        elts = torch.nn.functional.one_hot(elts, ChemData().NELTTYPES).float()

        return self.proj_atoms(elts)

    def forward(self, latent_feats):
        # get outputs + noised xyz
        (
            msa,
            pair,
            state,
            seq_unmasked,
            is_valid_atom,
            num_bonds_sequence,
            xyz,
            t,
            is_atomized,
            is_prot,
            is_na,
            dist_matrix,
            idx,
        ) = self._unpack_latents(latent_feats)

        t_hat = (1 - t) * du.NM_TO_ANG_SCALE  ## ?

        # initial state embedding from msa (single seq) + state
        msa = msa.squeeze(1)
        state = torch.cat([msa, state], dim=-1)
        state = self.proj_state_0(self.norm_state_0(state))

        # add timestep embedding
        timestep_emb_T = self.fourier_embedding(
            1 / 4 * torch.log(t_hat / self.sigma_data)
        )
        timestep_emb_T = self.emb_timestep(self.norm_timstep_emb(timestep_emb_T))
        state = state + timestep_emb_T

        state = self.ff_state_0(state)

        # initial pair embedding
        pair = self.proj_pair_0(self.norm_pair_0(pair))
        pair = self.ff_pair_0(pair)

        # add atom embeddings
        # to do: a lot more...
        atomstate = state[..., None, :] + self._embed_1d(latent_feats)

        # encode atom level
        atomstate, xyz = self.atom_update(
            t_hat,
            xyz,
            atomstate,
            is_valid_atom,
            is_prot,
            is_na,
            is_atomized,
            idx,
            num_bonds_sequence,
            dist_matrix,
            encoder=True,
        )

        # to residue level
        atomstate = F.relu_(self.proj_state_1(F.relu_(atomstate)))
        state = (
            self.proj_state_1(self.norm_state_1(state))
            + (atomstate * is_valid_atom[..., None]).sum(dim=2)
            / is_valid_atom.sum(dim=2)[..., None]  # project down atomstate
        )

        # res level updates
        for layer in self.token_processing:
            state = layer(state, pair)

        # combine atom&res level state, decode atom level
        atomstate = (
            atomstate + self.proj_state_2(self.norm_state_2(state))[..., None, :]
        )

        atomstate, xyz = self.atom_update(
            t_hat,
            xyz,
            atomstate,
            is_valid_atom,
            is_prot,
            is_na,
            is_atomized,
            idx,
            num_bonds_sequence,
            dist_matrix,
            encoder=False,
        )

        # to residue level
        state = (atomstate * is_valid_atom[..., None]).sum(dim=2) / is_valid_atom.sum(
            dim=2
        )[..., None]

        return {
            "state": state,
            "xyz": xyz,
        }

    def atom_update(
        self,
        t_hat,
        xyz,
        state,
        is_valid_atom,
        is_prot,
        is_na,
        is_atomized,
        idx,
        num_bonds_sequence,
        dist_matrix,
        encoder=True,
    ):
        SE3_SCALE = 10.0  # to do: make this a parameter

        if encoder:
            layers = self.atom_encoder
        else:
            layers = self.atom_decoder

        for layer in layers:
            G, edge = make_atom_graph(
                xyz,
                is_valid_atom,
                is_prot,
                is_na,
                is_atomized,
                idx,
                num_bonds_sequence,
                dist_matrix,
            )
            node = state[is_valid_atom]
            node_l1 = xyz[
                is_valid_atom
            ].unsqueeze(
                -2
            )  # torch.ones((node.shape[0], 3,3), device=state.device, dtype=state.dtype)
            edge = self.proj_edge(edge).unsqueeze(-1)
            shift = layer(G, node[..., None], node_l1, edge)

            xyz[is_valid_atom] = xyz[is_valid_atom] + shift["1"].squeeze(1) / SE3_SCALE
            if encoder:
                state[is_valid_atom] = (
                    state[is_valid_atom] + shift["0"][..., 0] / SE3_SCALE
                )
            else:
                xyz = (
                    xyz + shift["0"].sum()
                )  # fd hack to avoid unused grads (even though shift['0'] is of dim 0)

        return state, xyz
