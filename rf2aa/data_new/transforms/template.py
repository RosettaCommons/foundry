import logging
from dataclasses import dataclass
from typing import Any

import biotite.structure as struc
import numpy as np
import pandas as pd
import torch
from assertpy import assert_that
from biotite.structure import AtomArray
from cifutils.cifutils_biotite import cifutils_biotite

from data.data_constants import ChainType
from rf2aa.chemical import ChemicalData, initialize_chemdata
from rf2aa.data_new.dataframe_parsers import QueryMoleculeDFParser
from rf2aa.data_new.loader import load_from_row
from rf2aa.data_new.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from rf2aa.data_new.transforms.atom_array import AddChainTypeAnnotation, FilterAndAnnotateMolecules
from rf2aa.data_new.transforms.base import Compose, Transform
from rf2aa.data_new.transforms.polymer_encoding import atom_array_from_encoding, atom_array_to_rf2aa_atom36
from rf2aa.data_new.utils import get_template_msa_lookup_id

initialize_chemdata()
chemdata = ChemicalData()

logger = logging.getLogger(__name__)


@dataclass
class RF2AATemplate:
    """
    Data class for holding template information in the RF, RF2 & RF2AA format.

    NOTE:
     - RF templates only exist for proteins
     - This is a helper class to cast the templates into a more `readable` format and also
       to provide an interface layer that allows us to deal with templates as atom_arrays, if
       we ever re-create templates or add templates for non-proteins
     - RF-style templates already come encoded in atom14 representation (RFAtom14, not AF2Atom14)

    Keys:
    - xyz: Tensor([1, n_templates x n_atoms_per_template, 14, 3]), raw coordinates of all templates
    - mask: Tensor([1, n_templates x n_atom_per_template, 14]), mask of all templates
    - qmap: Tensor([1, n_templates x n_atom_per_template, 2]), alignment mapping of all templates
        - index 0: which index in the query protein this template index matches to
        - index 1: which template index this matches to
    - f0d: Tensor([1, n_templates, 8?]), [0,:,4] holds sequence identity info
    - f1d: Tensor([1, n_templates x n_atoms_per_template, 3]), something in there may be related to template confidence, gaps?
    - seq: Tensor([1, 100677]) (tensor, encoded with Chemdata.aa2num encoding)
    - ids: list[tuple[str]]  # Holds the f"{pdb_id}_{chain_id}" of the template
    - label: list[str]  # holds the lookup_id for this template
    """

    xyz: torch.Tensor  # [1, n_templates x n_atoms_per_template, 14, 3]
    mask: torch.Tensor  # [1, n_templates x n_atom_per_template, 14]
    qmap: torch.Tensor  # [1, n_templates x n_atom_per_template, 2]
    f0d: torch.Tensor  # [1, n_templates, 8?]
    f1d: torch.Tensor  # [1, n_templates x n_atoms_per_template, 3]
    seq: torch.Tensor  # [1, n_templates x n_atoms_per_template]
    ids: list[tuple[str]]  # Holds the f"{pdb_id}_{chain_id}" of the template
    label: list[str]  # holds the lookup_id for this template

    def __post_init__(self):
        self.ids = np.array(self.ids).flatten().squeeze()  # Flatten the list of tuples into an array
        # Convert all tensors to numpy
        self.xyz = self.xyz.numpy()
        self.mask = self.mask.numpy()
        self.qmap = self.qmap.numpy()
        self.f0d = self.f0d.numpy()
        self.f1d = self.f1d.numpy()
        self.seq = self.seq.numpy()
        self.label = np.array(self.label)

    @property
    def lookup_id(self) -> str:
        return self.label[0]

    @property
    def n_templates(self) -> int:
        return self.f0d.shape[1]

    @property
    def seq_similarity_to_query(self) -> np.ndarray:
        return self.f0d[0, :, 4]

    @property
    def alignment_confidence(self) -> np.ndarray:
        return self.f1d[0, :, 2]

    @property
    def pdb_ids(self) -> np.ndarray:
        return np.array([i.split("_")[0] for i in self.ids])

    @property
    def chain_ids(self) -> np.ndarray:
        return np.array([i.split("_")[1] for i in self.ids])

    @property
    def n_res_per_template(self) -> np.ndarray:
        return np.unique(self.qmap[:, :, 1], return_counts=True)[1]

    @property
    def max_aligned_query_res_idx(self) -> np.ndarray:
        aligned_query_res_idxs = self.qmap[0, :, 0]
        new_template_start_idxs = np.cumsum(self.n_res_per_template)[:-1]
        groups = np.split(aligned_query_res_idxs, new_template_start_idxs)
        # get max in each group (= template)
        return np.array([np.max(g) for g in groups])

    @property
    def template_ids(self) -> list[str]:
        return np.array(self.ids)

    def subset(self, template_idxs: list[int]) -> "RF2AATemplate":
        """
        Subset the template to only include the template indices specified in `template_idxs`.
        """
        assert np.unique(template_idxs).size == len(template_idxs), "`template_idxs` must be unique"

        # Subset the data
        template_atom_idxs = np.where(np.isin(self.qmap[0, :, 1], template_idxs))[0]
        self.xyz = self.xyz[:, template_atom_idxs]
        self.mask = self.mask[:, template_atom_idxs]
        self.qmap = self.qmap[:, template_atom_idxs]

        # Update internal template index to be from 0 to n_templates
        n_res_per_template = np.unique(self.qmap[:, :, 1], return_counts=True)[1]
        self.qmap[0, :, 1] = np.repeat(np.arange(len(template_idxs)), n_res_per_template)

        self.f0d = self.f0d[:, template_idxs]
        self.f1d = self.f1d[:, template_atom_idxs]
        self.seq = self.seq[:, template_atom_idxs]
        self.ids = self.ids[template_idxs]
        return self

    def to_atom_array(self, template_idx: int) -> AtomArray:
        assert_that(template_idx).is_instance_of(int).is_between(0, self.n_templates - 1)

        # Get pdb_id and chain_id
        template_id = self.ids[template_idx]
        pdb_id, chain_id = template_id.split("_")

        # Get indices to select the residues for the template
        template_res_idxs = np.where(self.qmap[0, :, 1] == template_idx)[0]

        # Select the template data
        # ... coordinate info
        atom14_coords = self.xyz[0, template_res_idxs, :, :]
        # ... occupancy info
        atom14_mask = self.mask[0, template_res_idxs, :]
        # ... sequence info
        seq_tokenized = self.seq[0, template_res_idxs]

        # Load encodings
        encoding_seq_tokens = np.array(chemdata.num2aa)
        encoding_token_atoms = np.array(chemdata.aa2long)[:, : chemdata.NHEAVYPROT]
        encoding_token_elements = np.array(chemdata.aa2elt)[:, : chemdata.NHEAVYPROT]

        # NOTE: There was a bug in the original code that saved the RF2 templates: Tryptophan (AA17) was using
        #  a wrong atom name ordering. This was fixed in the public version of the code:
        #  https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/c1fd92455be2a4133ad147242fc91cea35477282/rf2aa/chemical.py#L2068C1-L2070C285
        #  and we include this fix here:
        encoding_token_atoms[17] = np.array(
            [
                " N  ",
                " CA ",
                " C  ",
                " O  ",
                " CB ",
                " CG ",
                " CD1",
                " CD2",
                " NE1",
                " CE2",
                " CE3",
                " CZ2",
                " CZ3",
                " CH2",
            ]
        )
        encoding_token_elements[17] = np.array(
            [
                "N",
                "C",
                "C",
                "O",
                "C",
                "C",
                "C",
                "C",
                "N",
                "C",
                "C",
                "C",
                "C",
                "C",
            ]
        )

        # Create atom array
        atom_array = atom_array_from_encoding(
            atom14_coords,
            atom14_mask,
            seq_tokenized,
            encoding_seq_tokens=encoding_seq_tokens,
            encoding_token_atoms=encoding_token_atoms,
            encoding_token_elements=encoding_token_elements,
        )
        n_atom = len(atom_array)

        # ... repeat chain id for each atom in the residue
        atom_array.chain_id = np.repeat(np.array(chain_id), n_atom)

        # ... append custom annotation for which rewidue in the query protein this template
        #  residue aligns to
        aligned_query_res_idx = self.qmap[0, template_res_idxs, 0]  # + offset?
        atom_array.set_annotation("aligned_query_res_idx", struc.spread_residue_wise(atom_array, aligned_query_res_idx))

        # ... append custom annotation for alignment confidence
        alignment_confidence = self.f1d[0, template_res_idxs, 2]
        atom_array.set_annotation("alignment_confidence", struc.spread_residue_wise(atom_array, alignment_confidence))

        return atom_array


def blank_rf2aa_template_features(
    n_template: int,
    n_res: int,
    n_res_token: int = chemdata.NTOTAL,
    n_atom_token: int = chemdata.NAATOKENS,
    mask_token_idx: int = 20,
    init_coords: torch.Tensor | float = chemdata.INIT_CRDS,
) -> torch.Tensor:
    xyz = torch.full((n_template, n_res, n_res_token, 3), fill_value=float("nan"))
    xyz[:, :] = init_coords
    mask = torch.full((n_template, n_res, n_res_token), False, dtype=torch.bool)
    t1d = torch.full((n_template, n_res, n_atom_token), 0.0)
    t1d[..., mask_token_idx] = 1.0  # Set the mask token to 1.0
    # NOTE: In RF2AA the last dim of t1d is the `confidence`. We set it here just
    #  for code clarity.
    confidence = torch.zeros((n_template, n_res))
    t1d[..., -1] = confidence
    template_origin = np.full(n_template, "")
    return xyz, t1d, mask, template_origin


def _load_rf_template(rf_template_id: str) -> torch.Tensor:
    path_to_template = f"/projects/ml/TrRosetta/PDB-2021AUG02/torch/hhr/{rf_template_id[:3]}/{rf_template_id}.pt"
    return torch.load(path_to_template)


class AddRFTemplates(Transform):
    def __init__(
        self,
        n_template: int = 1,
        pick_top: bool = True,
        min_seq_similarity: float = 0.0,
        max_seq_similarity: float = 100.0,
        min_template_length: int = 0,
        filter_by_query_length: bool = False,
    ):
        assert_that(min_seq_similarity).is_between(0.0, 100.0)
        assert_that(max_seq_similarity).is_between(0.0, 100.0)
        assert_that(n_template).is_instance_of(int).is_greater_than(0)
        assert_that(min_template_length).is_instance_of(int).is_greater_than_or_equal_to(0)

        self.n_template = n_template
        self.pick_top = pick_top
        self.min_seq_similarity = min_seq_similarity
        self.max_seq_similarity = max_seq_similarity
        self.min_template_length = min_template_length
        self.filter_by_query_length = filter_by_query_length

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array", "chain_info", "pdb_id"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        pdb_id = data["pdb_id"]
        chain_info = data["chain_info"]

        # Load template information
        # NOTE: Currently templates only exist for proteins
        templates = {}
        for chain_id in chain_info:
            # get chain_type and convert to Enum
            chain_type = chain_info[chain_id]["type"]
            chain_type = ChainType.from_string(chain_type)
            rf_template_id = get_template_msa_lookup_id(pdb_id, chain_id, chain_type)

            if not rf_template_id:
                # early exit if no templates
                continue

            # NOTE: Could be made a lazy-load for each template only if it is selected
            #  if worker memory or speed becomes a bottleneck
            chain_templates = RF2AATemplate(**_load_rf_template(rf_template_id))
            is_valid = np.ones(chain_templates.n_templates, dtype=bool)

            # TODO: Revisit filtering logic once `cropping` is implemented to enable crop
            #  dependent filtering below (currently the below operates on the full query seq)
            if self.max_seq_similarity <= 100.0:
                # filter out templates with sequence similarity higher than cutoff
                is_valid &= chain_templates.seq_similarity_to_query <= self.max_seq_similarity

            if self.min_seq_similarity > 0.0:
                # filter out templates with sequence similarity lower than cutoff
                is_valid &= chain_templates.seq_similarity_to_query >= self.min_seq_similarity

            if self.min_template_length > 0:
                # filter out templates with fewer residues than cutoff
                is_valid &= chain_templates.n_res_per_template >= self.min_template_length

            # TODO: Ask what query lenght does? Implement once known
            # Skip poorly aligned or misaligned templates
            if self.filter_by_query_length:
                is_valid &= chain_templates.max_aligned_query_res_idx < query_length

            # TODO: Possibly filter by deposition date. This will require a query to the PDB
            #  to get the deposition date of each template

            if not np.any(is_valid):
                # early exit if no valid templates
                continue

            # pick `n_template` (or fewer if fewer exist) valid templates
            valid_template_idxs = np.where(is_valid)[0]
            if not self.pick_top:
                valid_template_idxs = np.random.permutation(valid_template_idxs)

            # Add templates to template dict
            chain_templates = chain_templates.subset(valid_template_idxs[: self.n_template])
            templates[chain_id] = [
                {
                    "id": chain_templates.ids[i],
                    "pdb_id": chain_templates.pdb_ids[i],
                    "chain_id": chain_templates.chain_ids[i],
                    "template_lookup_id": chain_templates.lookup_id,
                    "seq_similarity": chain_templates.seq_similarity_to_query[i],
                    "atom_array": chain_templates.to_atom_array(i),
                    "n_res": chain_templates.n_res_per_template[i],
                }
                for i in range(chain_templates.n_templates)
            ]
            logger.debug(f"Added {len(templates[chain_id])} templates for chain {chain_id}: {chain_templates.ids}.")

        data["template"] = templates
        return data


class FeaturizeRFTemplatesForRF2AA(Transform):
    requires_previous_transforms = [AddChainTypeAnnotation, AddRFTemplates]

    def __init__(
        self, n_template: int, mask_token_idx: int = 20, init_coords: torch.Tensor | float = chemdata.INIT_CRDS
    ):
        assert_that(n_template).is_instance_of(int).is_greater_than(0)
        self.n_template = n_template
        self.mask_token_idx = mask_token_idx
        self.init_coords = init_coords

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["template", "atom_array"])
        check_is_instance(data, "template", dict)
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_type"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        # Extract data
        atom_array = data["atom_array"]
        templates_by_chain = data["template"]

        # Featurize templates
        template_feat = {}
        for chain in struc.chain_iter(atom_array):
            chain_id = chain.chain_id[0]
            chain_type = chain.chain_type[0]

            if chain_type != ChainType.POLYPEPTIDE_L.value:
                # Only make templates for proteins
                continue

            # Initialize empty template
            n_res = struc.get_residue_count(chain)
            xyz, t1d, mask, _ = blank_rf2aa_template_features(
                self.n_template, n_res, mask_token_idx=self.mask_token_idx, init_coords=self.init_coords
            )
            template_ids = [None] * self.n_template

            if chain_id not in templates_by_chain:
                # Early exit if there are no templates for this chain
                continue

            # Fill with template data
            templates = templates_by_chain[chain_id]
            for i, templates in enumerate(templates):
                template_id = templates["id"]
                template_array = templates["atom_array"]
                # Filter out residues that are not aligned to the query (have no corresponding
                #  residue in the query protein). Convention: no alignment = -1
                template_array = template_array[template_array.aligned_query_res_idx >= 0]

                # Extract residue-wise alignment to query & confidence
                first_idx_per_res = struc.get_residue_starts(template_array)  # [n_res] (int)
                aligned_query_res_idx = template_array.aligned_query_res_idx[first_idx_per_res]  # [n_res] (int)
                confidence = template_array.alignment_confidence[first_idx_per_res]  # [n_res] (float)

                # [n_template_res, n_atoms_per_token, 3], [n_template_res, n_atoms_per_token], [n_template_res]  (n_atoms_per_token=36)
                xyz_t, mask_t, seq_t = atom_array_to_rf2aa_atom36(template_array)

                xyz[i, aligned_query_res_idx] = torch.tensor(xyz_t)
                mask[i, aligned_query_res_idx] = torch.tensor(mask_t)

                # Set the 1D template featrues
                t1d[i, aligned_query_res_idx, self.mask_token_idx] = (
                    0.0  # Reset the template features (were set to 1.0 for the mask token)
                )
                t1d[i, aligned_query_res_idx, :-1] = torch.nn.functional.one_hot(
                    torch.tensor(seq_t), chemdata.NAATOKENS - 1
                ).float()
                t1d[i, aligned_query_res_idx, -1] = torch.tensor(confidence)
                template_ids[i] = template_id

            # Save the template features
            template_feat[chain_id] = {
                "xyz": xyz,  # [n_template, n_res, n_atoms_per_token, 3] (float)
                "mask": mask,  # [n_template, n_res, n_atoms_per_token] (bool)
                "t1d": t1d,  # [n_tepmlate, n_res, 80],  0:79 coords = one-hot encoded sequence, 80 = confidence
                "template_ids": template_ids,  # [n_template] (str)
            }

        data["template_feat"] = template_feat
        return data


if __name__ == "__main__":
    # Demo:
    # Initialize the parser
    parser = cifutils_biotite.CIFParser()

    # Load the datasets
    query_df = pd.read_csv("/projects/ml/RF2_allatom/data_preprocessing/ncorley/2024-06-12/query_df.csv")
    query_molecule_df_parser = QueryMoleculeDFParser()

    from rf2aa.data_new.transforms.base import RaiseError, TransformPipelineError

    # Initialize the pipeline
    pipeline1 = Compose(
        [
            AddChainTypeAnnotation(),
            FilterAndAnnotateMolecules(),
            AddRFTemplates(
                n_template=3, pick_top=False, max_seq_similarity=60.0, min_seq_similarity=10.0, min_template_length=10
            ),
            FeaturizeRFTemplatesForRF2AA(n_template=5),
            RaiseError(),
        ]
    )

    pipeline2 = Compose(
        [
            AddChainTypeAnnotation(),
            FilterAndAnnotateMolecules(),
            AddRFTemplates(
                n_template=3, pick_top=False, max_seq_similarity=60.0, min_seq_similarity=10.0, min_template_length=10
            ),
            FeaturizeRFTemplatesForRF2AA(n_template=5),
        ]
    )

    # Choose an example
    example_row = query_df.iloc[0]
    try:
        # Load the data
        data = load_from_row(example_row, query_molecule_df_parser, parser)
        # Apply the pipeline
        data = pipeline1(data)
    except TransformPipelineError as e:
        logger.error(e)
        # Load the data
        data = load_from_row(example_row, query_molecule_df_parser, parser)
        # Apply the pipeline
        data = pipeline2(data, rng_state_dict=e.rng_state_dict)
