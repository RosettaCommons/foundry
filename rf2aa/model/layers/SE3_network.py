import inspect

import torch
import torch.nn as nn

from rf2aa.loss.loss import calc_chiral_grads
from rf2aa.model.layers.Attention_module import FeedForwardLayer
from rf2aa.model.layers.resnet import SCPred
from rf2aa.SE3Transformer.se3_transformer.model import SE3Transformer
from rf2aa.SE3Transformer.se3_transformer.model.fiber import Fiber

# script_dir = os.path.dirname(os.path.realpath(__file__))+'/'
# sys.path.insert(0,script_dir+'SE3Transformer')
from rf2aa.util import xyz_frame_from_rotation_mask
from rf2aa.util_module import (
    get_seqsep_protein_sm,
    init_lecun_normal,
    make_full_graph,
    rbf,
)

se3_transformer_path = inspect.getfile(SE3Transformer)
se3_fiber_path = inspect.getfile(Fiber)
assert "rf2aa" in se3_transformer_path


class SE3TransformerWrapper(nn.Module):
    """SE(3) equivariant GCN with attention"""

    def __init__(
        self,
        num_layers=2,
        num_channels=32,
        num_degrees=3,
        n_heads=4,
        div=4,
        l0_in_features=32,
        l0_out_features=32,
        l1_in_features=3,
        l1_out_features=2,
        num_edge_features=32,
        compute_gradients=False,
        **kwargs,
    ):
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

        self.se3 = SE3Transformer(
            num_layers=num_layers,
            fiber_in=fiber_in,
            fiber_hidden=fiber_hidden,
            fiber_out=fiber_out,
            num_heads=n_heads,
            channels_div=div,
            fiber_edge=fiber_edge,
            populate_edge="arcsin",
            final_layer="lin",
            use_layer_norm=True,
            compute_gradients=compute_gradients,
        )

        self.reset_parameter()

    def reset_parameter(self):
        pass
        # nn.init.zeros_(self.se3.graph_modules[-1].weights['0'])
        # if self.l1_out > 0:
        #    nn.init.zeros_(self.se3.graph_modules[-1].weights['1'])

    def forward(self, G, type_0_features, type_1_features=None, edge_features=None):
        if self.l1_in > 0:
            node_features = {"0": type_0_features, "1": type_1_features}
        else:
            node_features = {"0": type_0_features}
        edge_features = {"0": edge_features}
        return self.se3(G, node_features, edge_features)


class FullyConnectedSE3_noR(nn.Module):
    def __init__(
        self,
        d_msa,
        d_pair,
        d_rbf,
        num_layers,
        num_channels,
        num_degrees,
        n_heads,
        div,
        l0_in_features,
        l0_out_features,
        l1_in_features,
        l1_out_features,
        num_edge_features,
        residual_state,
        compute_gradients,
    ):
        super(FullyConnectedSE3_noR, self).__init__()
        # initial node & pair feature process
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.embed_node = nn.Linear(d_msa, l0_in_features)
        self.ff_node = FeedForwardLayer(l0_in_features, 2)  # HACK: hardcoded value
        self.norm_node = nn.LayerNorm(l0_in_features)

        self.embed_edge = nn.Linear(d_pair + d_rbf + 1, num_edge_features)
        self.ff_edge = FeedForwardLayer(num_edge_features, 2)
        self.norm_edge = nn.LayerNorm(num_edge_features)

        self.residual_state = residual_state

        self.se3 = SE3TransformerWrapper(
            num_layers=num_layers,
            num_channels=num_channels,
            num_degrees=num_degrees,
            n_heads=n_heads,
            div=div,
            l0_in_features=l0_in_features,
            l0_out_features=l0_out_features,
            l1_in_features=l1_in_features,
            l1_out_features=l1_out_features,
            num_edge_features=num_edge_features,
            compute_gradients=compute_gradients,
        )
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.embed_node = init_lecun_normal(self.embed_node)
        self.embed_edge = init_lecun_normal(self.embed_edge)

        # initialize bias to zeros
        nn.init.zeros_(self.embed_node.bias)
        nn.init.zeros_(self.embed_edge.bias)

    def embed_node_feats(self, msa, state):
        seq = self.norm_msa(msa[:, 0])
        node = self.embed_node(seq)
        node = node + self.ff_node(node)
        node = self.norm_node(node)
        return node

    def embed_edge_feats(self, pair, xyz, idx, bond_feats, dist_matrix, rotation_mask):
        neighbor = get_seqsep_protein_sm(idx, bond_feats, dist_matrix, rotation_mask)
        pair = self.norm_pair(pair)
        cas = xyz[:, :, 1].contiguous()
        rbf_feat = rbf(torch.cdist(cas, cas))
        edge = torch.cat((pair, rbf_feat, neighbor), dim=-1)
        edge = self.embed_edge(edge)
        edge = edge + self.ff_edge(edge)
        edge = self.norm_edge(edge)
        return edge

    def construct_graph(self, xyz, edge):
        B, L = xyz.shape[:2]
        idx = torch.arange(L, device=edge.device).reshape(
            B, L
        )  # NOTE: only works in B==1
        G, edge_feats = make_full_graph(xyz[:, :, 1, :], edge, idx)
        return G, edge_feats

    def construct_l1_feats(self, xyz, is_atom, atom_frames, chirals):
        l1_feats = get_chiral_vectors(xyz[..., :3, :], chirals)[
            ..., 1:2, :
        ]  # only pass features from Calpha
        return l1_feats

    def compute_structure_update(
        self,
        G,
        node,
        l1_feats,
        edge_feats,
        xyz,
        state,
        is_atom,
        drop_layer=False,
        is_motif=None,
    ):
        weight = 0.0 if drop_layer else 1.0
        B, L = xyz.shape[:2]
        shift = self.se3(G, node.reshape(B * L, -1, 1), l1_feats, edge_feats)

        if self.residual_state:
            state = state + shift["0"].reshape(B, L, -1)
        else:
            state = shift["0"].reshape(B, L, -1)

        offset = shift["1"].reshape(B, L, 3)
        if is_motif is not None:
            offset[is_motif, ...] = 0  # Frozen motif
        T = offset / 10.0
        xyz_update = xyz.clone()
        xyz_update[..., 1:2, :] = xyz[..., 1:2, :] + weight * T[..., None, :]
        return state, xyz_update, None

    def forward(
        self,
        msa,
        pair,
        state,
        xyz,
        is_atom,
        atom_frames,
        chirals,
        idx,
        bond_feats,
        dist_matrix,
        drop_layer=False,
        is_motif=None,
    ):
        # TODO: allow these functions to accept kwargs so we can pass
        # different inputs when iterating
        B, N, L = msa.shape[:3]
        node = self.embed_node_feats(msa, state)
        edge = self.embed_edge_feats(pair, xyz, idx, bond_feats, dist_matrix, is_atom)
        G, edge_feats = self.construct_graph(xyz, edge)
        # TODO: get extra l1 feats automatically and populate the extra l1 dimension
        l1_feats = self.construct_l1_feats(xyz, is_atom, atom_frames, chirals)
        state, xyz_update, quat_update = self.compute_structure_update(
            G,
            node,
            l1_feats,
            edge_feats,
            xyz,
            state,
            is_atom,
            drop_layer,
            is_motif=is_motif,
        )
        return {
            "state": state,
            "xyz": xyz_update,
            "quat_update": quat_update,
        }


class FullyConnectedSE3(FullyConnectedSE3_noR):
    def __init__(
        self,
        d_msa,
        d_pair,
        d_rbf,
        num_layers,
        num_channels,
        num_degrees,
        n_heads,
        div,
        l0_in_features,
        l0_out_features,
        l1_in_features,
        l1_out_features,
        num_edge_features,
        sc_pred_d_hidden,
        sc_pred_p_drop,
        residual_state,
        compute_gradients,
    ):
        """
        Params:
            sc_pred_d_hidden: Hidden dimension of the sidechain predictor.
                Set to 0 to omit sidechain prediction.
        """
        super().__init__(
            d_msa,
            d_pair,
            d_rbf,
            num_layers,
            num_channels,
            num_degrees,
            n_heads,
            div,
            l0_in_features,
            l0_out_features,
            l1_in_features,
            l1_out_features,
            num_edge_features,
            residual_state,
            compute_gradients,
        )
        self.embed_node = nn.Linear(d_msa + l0_out_features, l0_in_features)
        self.norm_state = nn.LayerNorm(l0_out_features)
        self.sc_predictor = None
        if sc_pred_d_hidden:
            self.sc_predictor = SCPred(
                d_msa=d_msa,
                d_state=l0_out_features,
                d_hidden=sc_pred_d_hidden,
                p_drop=sc_pred_p_drop,
            )
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.embed_node = init_lecun_normal(self.embed_node)
        self.embed_edge = init_lecun_normal(self.embed_edge)

        # initialize bias to zeros
        nn.init.zeros_(self.embed_node.bias)
        nn.init.zeros_(self.embed_edge.bias)

    def embed_node_feats(self, msa, state):
        seq = self.norm_msa(msa[:, 0])
        state = self.norm_state(state)
        node = self.embed_node(torch.cat((seq, state), dim=-1))
        node = node + self.ff_node(node)
        node = self.norm_node(node)
        return node

    def construct_l1_feats(self, xyz, is_atom, atom_frames, chirals):
        l1_feats = torch.cat(
            [
                get_backbone_offset_vectors(xyz, is_atom, atom_frames),
                get_chiral_vectors(xyz, chirals),
            ],
            dim=1,
        )
        return l1_feats

    def compute_structure_update(
        self,
        G,
        node,
        l1_feats,
        edge_feats,
        xyz,
        state,
        is_atom,
        drop_layer=False,
        is_motif=None,
    ):
        weight = 0.0 if drop_layer else 1.0
        B, L = node.shape[:2]
        shift = self.se3(G, node.reshape(B * L, -1, 1), l1_feats, edge_feats)

        if self.residual_state:
            state = state + shift["0"].reshape(B, L, -1)  # fd change
        else:
            state = shift["0"].reshape(B, L, -1)  # fd change

        offset = shift["1"].reshape(B, L, 2, 3)
        if is_motif is not None:
            offset[is_motif, ...] = 0  # Frozen motif
        T = offset[:, :, 0, :] / 10
        R = offset[:, :, 1, :] / 100.0

        Qnorm = torch.sqrt(1 + torch.sum(R * R, dim=-1))
        qA, qB, qC, qD = (
            1 / Qnorm,
            R[:, :, 0] / Qnorm,
            R[:, :, 1] / Qnorm,
            R[:, :, 2] / Qnorm,
        )

        v = xyz - xyz[:, :, 1:2, :]
        Rout = torch.zeros((B, L, 3, 3), device=xyz.device)
        Rout[:, :, 0, 0] = qA * qA + qB * qB - qC * qC - qD * qD
        Rout[:, :, 0, 1] = 2 * qB * qC - 2 * qA * qD
        Rout[:, :, 0, 2] = 2 * qB * qD + 2 * qA * qC
        Rout[:, :, 1, 0] = 2 * qB * qC + 2 * qA * qD
        Rout[:, :, 1, 1] = qA * qA - qB * qB + qC * qC - qD * qD
        Rout[:, :, 1, 2] = 2 * qC * qD - 2 * qA * qB
        Rout[:, :, 2, 0] = 2 * qB * qD - 2 * qA * qC
        Rout[:, :, 2, 1] = 2 * qC * qD + 2 * qA * qB
        Rout[:, :, 2, 2] = qA * qA - qB * qB - qC * qC + qD * qD
        I = torch.eye(3, device=Rout.device).expand(B, L, 3, 3)
        Rout = torch.where(is_atom.reshape(B, L, 1, 1), I, Rout)
        if weight != 1:
            Rout = (1 - weight) * I + weight * Rout

        xyz = (
            torch.einsum("blij,blaj->blai", Rout, v)
            + xyz[:, :, 1:2, :]
            + weight * T[:, :, None, :]
        )
        quat_update = torch.stack([qA, qB, qC, qD], dim=2)

        return state, xyz, quat_update

    def forward(
        self,
        msa,
        pair,
        state,
        xyz,
        is_atom,
        atom_frames,
        chirals,
        idx,
        bond_feats,
        dist_matrix,
        drop_layer=False,
        is_motif=None,
    ):
        block_outputs = super().forward(
            msa,
            pair,
            state,
            xyz,
            is_atom,
            atom_frames,
            chirals,
            idx,
            bond_feats,
            dist_matrix,
            is_motif=is_motif,
        )
        state, xyz = block_outputs["state"], block_outputs["xyz"]

        alpha = None
        if self.sc_predictor:
            alpha = self.sc_predictor(msa[:, 0], state)
        return {
            "state": state,
            "xyz": xyz,
            "alpha": alpha,
            "quat_update": block_outputs["quat_update"],
        }


def get_backbone_offset_vectors(xyz, is_atom, atom_frames):
    xyz_frame = xyz_frame_from_rotation_mask(xyz, is_atom, atom_frames)
    l1_feats = xyz_frame - xyz_frame[:, :, 1, :].unsqueeze(2)
    return l1_feats[0][..., :3, :]


def get_chiral_vectors(xyz, chirals):
    (dchiraldxyz,) = calc_chiral_grads(xyz, chirals)
    extra_l1 = dchiraldxyz[0]
    extra_l1_slice = extra_l1.clone()
    return extra_l1_slice.detach()
