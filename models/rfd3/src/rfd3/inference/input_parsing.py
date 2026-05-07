import copy
import json
import logging
import os
import time
import warnings
from contextlib import contextmanager
from os import PathLike
from typing import Any, Dict, List, Optional, Union

import numpy as np
from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.io.parser import parse_atom_array
from atomworks.io.utils.bonds import get_inferred_polymer_bonds

# from atomworks.ml.datasets.datasets import BaseDataset
from atomworks.ml.transforms.base import TransformedDict
from atomworks.ml.utils.token import (
    get_token_starts,
)
from biotite import structure as struc
from biotite.structure import AtomArray, BondList, get_residue_starts
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from rfd3.constants import (
    INFERENCE_ANNOTATIONS,
    OPTIONAL_CONDITIONING_VALUES,
    REQUIRED_CONDITIONING_ANNOTATION_VALUES,
    REQUIRED_INFERENCE_ANNOTATIONS,
)
from rfd3.inference.legacy_input_parsing import (
    _check_has_backbone_connections_to_nonstandard_residues,
    create_atom_array_from_design_specification_legacy,
)
from rfd3.inference.parsing import InputSelection
from rfd3.inference.symmetry.symmetry_utils import (
    SymmetryConfig,
    center_symmetric_src_atom_array,
    make_symmetric_atom_array,
)
from rfd3.transforms.conditioning_base import (
    check_has_required_conditioning_annotations,
    convert_existing_annotations_to_bool,
    get_motif_features,
    set_default_conditioning_annotations,
)
from rfd3.transforms.util_transforms import assign_types_
from rfd3.utils.inference import (
    extract_ligand_array,
    inference_load_,
    set_com,
    set_common_annotations,
    set_indices,
)

from foundry.common import exists
from foundry.utils.components import (
    fetch_mask_from_idx,
    get_design_pattern_with_constraints,
    get_motif_components_and_breaks,
)
from foundry.utils.ddp import RankedLogger

logging.basicConfig(level=logging.DEBUG)

logger = RankedLogger(__name__, rank_zero_only=True)


#################################################################################
# Custom infer_ori functions
#################################################################################


class LegacySpecification(BaseModel):
    """Legacy specification for compatibility with legacy input parsing."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    def build(self, *args, **kwargs):
        """Build atom array using legacy input parsing."""
        atom_array = create_atom_array_from_design_specification_legacy(
            **self.model_dump(),
        )
        return atom_array, self.model_dump()

    def to_pipeline_input(self, example_id):
        atom_array, spec_dict = self.build(return_metadata=True)

        # ... Forward into
        data = prepare_pipeline_input_from_atom_array(atom_array)
        data["example_id"] = example_id

        # ... Wrap up with additional features
        if "extra" not in spec_dict:
            spec_dict["extra"] = {}
        spec_dict["extra"]["example_id"] = example_id
        data["specification"] = spec_dict
        return data


# ========================================================================
# Input specification
# ========================================================================


class DesignInputSpecification(BaseModel):
    """Validated and parsed input specification before resolution."""

    model_config = ConfigDict(
        hide_input_in_errors=False,
        arbitrary_types_allowed=True,
        validate_assignment=False,
        str_strip_whitespace=True,
        str_min_length=1,
        extra="forbid",
    )
    # fmt: off
    # ========================================================================
    # Data inputs, motif generation & selection
    # ========================================================================
    # Data inputs
    atom_array_input: Optional[AtomArray] = Field(None, description="Loaded atom array", exclude=True)
    input: Optional[str] =  Field(None, description="Path to input PDB/CIF file")
    # Motif selection from input file
    contig:  Optional[InputSelection] = Field(None, description="Contig specification string (e.g. 'A1-10,B1-5')")
    unindex: Optional[InputSelection] = Field(None, 
        description="Unindexed components selection. Components to fix in the generated structure without specifying sequence index. "\
        "Components must not overlap with `contig` argument. "\
        "E.g. 'A15-20,B6-10' or dict. We recommend specifying unindexed residues as a contig string, "\
        "then using select_fixed_atoms will subset the atoms to the specified atoms")
    # Extra args:
    length:  Optional[str] = Field(None, description="Length range as 'min-max' or int. Constrains length of contig if provided")
    ligand:  Optional[str] = Field(None, description="Ligand name or index to include in design.")
    allow_ligand_on_existing_chain: bool = Field(False, description="If True, suppress the error when a ligand shares a chain ID with the built atom array. Use with caution — chain ID is leaked to the model.")
    cif_parser_args: Optional[Dict[str, Any]] = Field(None, description="CIF parser arguments")
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extra metadata to include in output (useful for logging additional info in metadata)")
    dialect: int = Field(2, description="RFdiffusion3 input dialect. 1: legacy, 2: release.")

    # ========================================================================
    # Conditioning
    # ========================================================================
    # Sequence and coordinate conditioning
    select_fixed_atoms: Optional[InputSelection] = Field(None,
        description='''Atoms to fix coordinates for. Examples:
        - True (default when inputs provided): All atoms pulled from the input are fixed in 3d space
        - False: All atoms pulled from the input are unfixed in 3d space
        - ContigStr: Components to fix in 3d space, e.g. "A1-10,B1-3" fixes residues 1-10 in chain A and residues 1-3 in chain B.
        - {"A1": "N,CA,C,O,CB,CG", "A2-10": "BKBN"} fixes backbone and CB for residues 1 and 2, and all atoms for residues 3-10 in chain A.
    '''.replace('\t\t', '\t')
    )
    select_unfixed_sequence: Optional[InputSelection] = Field(None, description='''Components to unfix sequence for. 
        - True (default when inputs provided): All atoms from the input have fixed sequences by default.
        - False: All atoms pulled from the input have diffused sequences by default.
        - ContigStr: Components to unfix sequence for, e.g. "A5-10,B1-3" unfixes sequence for residues 5-10 in chain A and residues 1-3 in chain B.
        - Dictionary: Allowed but not recommended.
        NOTE: Excludes ligands (ligands / DNA always has fixed sequence).
    '''.replace('\t\t', '\t')
    )
    # Assignments of conditioning annotations
    # RASA accessibilty
    select_buried: Optional[InputSelection] = Field(None, description="Selection of RASA buried conditioning")
    select_partially_buried: Optional[InputSelection] = Field(None, description="Selection of RASA partially buried conditioning")
    select_exposed: Optional[InputSelection] = Field(None, description="Selection of RASA exposed conditioning")
    # Hotspots & Hbonds
    select_hbond_acceptor: Optional[InputSelection] = Field(None, description="Atom-wise hydrogen bond acceptor")
    select_hbond_donor: Optional[InputSelection] = Field(None, description="Atom-wise hydrogen bond donor")
    select_hotspots: Optional[InputSelection] = Field(None, description="Atom-level or token-level hotspots for PPI")
    redesign_motif_sidechains: Union[bool, str] = Field(False, 
        description="Perform fixed-backbone sequence design on when 'contig' is provided. Changes the default behaviour when not using `select_fixed_atoms`."
    )

    # ========================================================================
    # Global conditioning & symmetry
    # ========================================================================
    # Symmetry
    symmetry: Optional[SymmetryConfig] = Field(None, description="Symmetry specification, see docs/symmetry.md")
    # Centering & COM guidance
    ori_token: Optional[list[float]] = Field(None, description="Origin coordinates")
    infer_ori_strategy: Optional[str] = Field(None, description="Strategy for inferring origin; `com` or `hotspots`")
    # Additional global conditioning
    plddt_enhanced: Optional[bool] = Field(True, description="Enable pLDDT enhancement")
    is_non_loopy: Optional[bool] = Field(None, description="Non-loopy conditioning")
    # Partial diffusion
    partial_t: Optional[float] = Field(None, ge=0.0, description="Angstroms of noise to add for partial diffusion (None turns off partial diffusion), t <= 15 recommended.")
    # fmt: on

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def is_partial_diffusion(self) -> bool:
        """Whether partial diffusion is enabled."""
        return exists(self.partial_t)

    # ========================================================================
    # Loading / saving
    # ========================================================================

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_rfd3_out(cls, path: str):
        """Load from path to rfd3 outputs, either .cif, .cif.gz, .json or denoised / noisy trajectory files"""
        path = path.replace(".cif.gz", ".cif").replace(".cif", ".json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Output file not found at {path}")
        with open(path, "r") as f:
            data = json.load(f)
        if "input_specification" in data:
            spec_args = data["input_specification"]
            return cls(**spec_args)
        else:
            raise ValueError(f"No input specification found in json output: {path}")

    def get_dict_to_save(self, exclude_extra: bool = False) -> dict:
        # Returns dictionary for saving (reproducible) outputs to json
        return self.model_dump(
            exclude_defaults=True,
            exclude={"atom_array_input"} | set({"extra"} if exclude_extra else {}),
        )

    # ========================================================================
    # Pre-Validation / canonicalization
    # ========================================================================

    @model_validator(mode="before")
    @classmethod
    def validate_input_schema(cls, data: dict) -> dict:
        if not (
            exists(data.get("input"))
            or exists(data.get("contig"))
            or exists(data.get("length"))
        ):
            raise ValueError("Either 'input' or 'contig' / 'length' must be provided.")

        # unused input check
        if exists(data.get("input")) and not (
            (
                exists(data.get("contig"))
                or exists(data.get("unindex"))
                or exists(data.get("ligand"))
            )
            or exists(data.get("partial_t"))
        ):
            raise ValueError("Input provided but unused in composition specification.")

        if not exists(data.get("partial_t")):
            # non-partial diffusion checks
            if exists(data.get("unindex")) and not (
                exists(data.get("contig")) or exists(data.get("length"))
            ):
                raise ValueError(
                    "Unindex provided but neither a length nor contig was specified."
                )
        else:
            # partial diffusion checks
            if exists(data.get("length")):
                raise ValueError(
                    "Length argument must not be provided during partial diffusion."
                )
            if not (exists(data.get("input")) or exists(data.get("atom_array_input"))):
                raise ValueError(
                    "Partial diffusion requires input file or input atom array."
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def canonicalize(cls, data: dict) -> dict:
        # Canonicalize length argument
        data["length"] = str(data["length"]) if exists(data.get("length")) else None

        # Normalize input to str
        data["input"] = str(data["input"]) if exists(data.get("input")) else None
        return data

    @model_validator(mode="before")
    @classmethod
    def load_input(cls, data: dict) -> dict:
        with validator_context("load_input"):
            # ... Find provided selections
            selections = [
                # Motif
                "contig",
                "unindex",
                # Aux
                "select_fixed_atoms",
                "select_unfixed_sequence",
                # Conditioning
                "select_buried",
                "select_partially_buried",
                "select_exposed",
                "select_hbond_acceptor",
                "select_hbond_donor",
                "select_hotspots",
            ]
            selections = [s for s in selections if s in data]

            # ... Early return if no input file provided / atom array input
            if not exists(data.get("input")) and not exists(
                data.get("atom_array_input")
            ):
                if selections:
                    raise ValueError(
                        "Atom array input must be provided before parsing selections: {}".format(
                            selections
                        )
                    )
                return data

            # ... Load atom array from input file if provided
            if exists(data["input"]):
                if exists(data.get("atom_array_input")):
                    raise ValueError(
                        "Both 'input' and 'atom_array_input' provided; please provide only one."
                    )
                atom_array = inference_load_(
                    data["input"], cif_parser_args=data.get("cif_parser_args")
                )["atom_array"]

                # Center for symmetric design
                if exists(data.get("symmetry")) and data["symmetry"].get("id"):
                    atom_array = center_symmetric_src_atom_array(atom_array)

                if "atom_id" in atom_array.get_annotation_categories():
                    atom_array.del_annotation("atom_id")

                data["atom_array_input"] = atom_array

            atom_array = data["atom_array_input"]

            # ... Set defaults if not provided
            if not exists(data.get("select_fixed_atoms")):
                data["select_fixed_atoms"] = InputSelection.from_any(
                    True, atom_array=atom_array
                )
            if not exists(data.get("select_unfixed_sequence")):
                data["select_unfixed_sequence"] = InputSelection.from_any(
                    False, atom_array=atom_array
                )

            # Coerce selections
            for sele in selections:
                if sele in ["contig", "unindexed_breaks"]:
                    if exists(data[sele]) and not isinstance(data[sele], str):
                        raise ValueError(
                            f"{sele} selection must be a string or None, got {type(data[sele])} instead."
                        )
                if not isinstance(data.get(sele), InputSelection):
                    data[sele] = InputSelection.from_any(
                        data[sele], atom_array=atom_array
                    )
        return data

    # ========================================================================
    # Post-Validation
    # ========================================================================

    @model_validator(mode="after")
    def assert_exclusivity(self):
        with validator_context("assert_exclusivity"):
            # ... Assert and indexed do not overlap
            if exists(self.contig) and exists(self.unindex):
                indexed_set = set(self.contig.keys())
                unindexed_set = set(self.unindex.keys())
                overlap = indexed_set & unindexed_set
                if overlap:
                    raise ValueError(
                        f"Indexed and unindexed components must not overlap, got: {overlap}"
                    )

            # ... Assert mutual exclusivity of rasa binning
            exclusive_sets = [
                ("Motifs", ("contig", "unindex")),
                (
                    "RASA",
                    ("select_buried", "select_partially_buried", "select_exposed"),
                ),
            ]

            for name, excl_set in exclusive_sets:
                masks = [getattr(self, field, None) for field in excl_set]
                masks = [m.get_mask() for m in masks if m is not None]
                if not masks:
                    continue
                mask_sum = np.zeros_like(masks[0], dtype=int)
                for m in masks:
                    if m is not None:
                        mask_sum += m.astype(int)
                if np.any(mask_sum > 1):
                    raise ValueError(
                        f"Selections for `{name}` must be mutually exclusive, got overlapping selections: {excl_set}. Mask sum: {mask_sum}"
                    )

        return self

    @model_validator(mode="after")
    def attempt_expansion(self):
        if self.is_partial_diffusion and exists(self.contig):
            contig = self.contig
            length = self.length
            try:
                get_design_pattern_with_constraints(contig.raw, length=length)
            except Exception as e:
                raise ValueError(f"Failed to expand contig ({contig.raw}): {e}")
        return self

    @model_validator(mode="after")
    def _assign_types_to_input(self):
        """Assign conditioning annotations to the input atom array"""
        aa = self.atom_array_input
        if not exists(aa):
            return self

        # ... Selections and their annotation values
        selection_fields = {
            # field name:         (annotation name, assigned value, non-selected value)
            "select_fixed_atoms": ("is_motif_atom_with_fixed_coord", True, False),
            "select_unfixed_sequence": ("is_motif_atom_with_fixed_seq", False, True),
            "unindex": ("is_motif_atom_unindexed", True, False),
            "select_hotspots": ("is_atom_level_hotspot", True, False),
            "select_hbond_acceptor": ("active_acceptor", True, False),
            "select_hbond_donor": ("active_donor", True, False),
            "select_buried": ("rasa_bin", 0, 3),
            "select_partially_buried": ("rasa_bin", 1, 3),
            "select_exposed": ("rasa_bin", 2, 3),
        }
        selection_fields = {
            k: v for k, v in selection_fields.items() if exists(getattr(self, k, None))
        }

        # ... Init global
        [
            aa.set_annotation(name, np.full(aa.array_length(), val, dtype=int))
            for name, val in REQUIRED_CONDITIONING_ANNOTATION_VALUES.items()
        ]

        # Application of selections to each token fn;
        def apply_selections(start, end):
            chain_id = aa.chain_id[start]
            res_id = aa.res_id[start]

            # Assign all select fields to atom array annotations.
            for selection_name, (
                annotation_name,
                set_value,
                default_value,
            ) in selection_fields.items():
                # ... Get input values
                selection = getattr(self, selection_name)

                # Important line: selects from data dictionary based on src chain & res_id (Not name!)
                atom_names_sele = selection.get(f"{chain_id}{res_id}")

                if atom_names_sele is None:
                    continue
                mask = np.isin(aa.atom_name[start:end], atom_names_sele)
                if annotation_name in aa.get_annotation_categories():
                    # ... Set only mask overridden features if exists in atom array
                    aa.get_annotation(annotation_name)[start:end] = np.where(
                        mask, set_value, default_value
                    ).astype(np.int_)
                    # ).astype(int)
                else:
                    # ... Otherwise, set the entire annotation and use defaults for unselected
                    mask_aa = np.zeros(aa.array_length(), dtype=bool)
                    mask_aa[start:end] = mask
                    annotation_values = np.where(
                        mask_aa,
                        set_value,
                        default_value,
                    ).astype(np.int_)
                    aa.set_annotation(annotation_name, annotation_values)

        # ... Set default assignments per-token based on whether redesigning
        starts = get_residue_starts(aa, add_exclusive_stop=True)
        for start, end in zip(starts[:-1], starts[1:]):
            # ... Relax sequence and sidechains
            if aa.res_name[start] in STANDARD_AA and self.redesign_motif_sidechains:
                is_bkbn = np.isin(aa.atom_name[start:end], ["N", "CA", "C", "O"])
                aa.is_motif_atom_with_fixed_coord[start:end] = is_bkbn.astype(int)
                aa.is_motif_atom_with_fixed_seq[start:end] = np.full_like(
                    is_bkbn, False, dtype=int
                )

            # ... Apply selections on top
            apply_selections(start, end)

        return self

    # ========================================================================
    # Building
    # ========================================================================

    def build(self, return_metadata=False):
        """Main build pipeline."""
        atom_array_input_annotated = copy.deepcopy(self.atom_array_input)
        atom_array = self._build_init(atom_array_input_annotated)

        # Apply post-processing
        atom_array = self._append_ligand(atom_array, atom_array_input_annotated)
        atom_array = self._apply_symmetry(atom_array, atom_array_input_annotated)

        # Apply globals to all tokens (including diffused)
        atom_array = self._set_origin(atom_array)
        atom_array = self._apply_globals(atom_array)

        # Final validation and cleanup
        check_has_required_conditioning_annotations(
            atom_array, required=REQUIRED_INFERENCE_ANNOTATIONS
        )
        convert_existing_annotations_to_bool(atom_array)

        # ... Route return type
        if not return_metadata:
            return copy.deepcopy(atom_array)
        else:
            metadata = self.get_dict_to_save()
            metadata["extra"] = metadata.get("extra", {}) | {
                "num_tokens_in": len(get_token_starts(atom_array)),
                "num_residues_in": len(get_residue_starts(atom_array)),
                "num_chains": len(np.unique(atom_array.chain_id)),
                "num_atoms": len(atom_array),
                "num_residues": len(
                    np.unique(list(zip(atom_array.chain_id, atom_array.res_id)))
                ),
            }
            return copy.deepcopy(atom_array), metadata

    # ============================================================================
    # Building functions
    # ============================================================================

    def _build_init(self, atom_array_input_annotated):
        # ... Fetch tokens
        indexed_tokens = (
            self.contig.get_tokens(atom_array_input_annotated)
            if exists(self.contig)
            else {}
        )
        unindexed_tokens = (
            self.unindex.get_tokens(atom_array_input_annotated)
            if exists(self.unindex)
            else {}
        )
        # Subset to only fixed coordindate atoms
        unindexed_tokens = {
            k: tok[tok.is_motif_atom_with_fixed_coord.astype(bool)]
            for k, tok in unindexed_tokens.items()
        }
        unindexed_components, unindexed_breaks = self.break_unindexed(self.unindex)

        if not self.is_partial_diffusion:
            # ... Sample the contig string
            components_to_accumulate = get_design_pattern_with_constraints(
                self.contig.raw if exists(self.contig) else self.length,
                length=self.length,
            )
            self.extra["sampled_contig"] = ",".join(
                [str(x) for x in components_to_accumulate]
            )

            # ... Include unindexed components in accumulation
            unindexed_breaks = [None] * len(components_to_accumulate) + unindexed_breaks
            components_to_accumulate += unindexed_components

            # ... Accumulate from scratch
            atom_array = accumulate_components(
                components_to_accumulate,
                indexed_tokens=indexed_tokens,
                unindexed_tokens=unindexed_tokens,
                atom_array_accum=[],
                unindexed_breaks=unindexed_breaks,
                start_chain="A",
                start_resid=1,
            )
        else:
            # ... Set common annotations
            atom_array_in = assign_types_(copy.deepcopy(atom_array_input_annotated))
            atom_array_in = set_common_annotations(
                atom_array_in, set_src_component_to_res_name=False
            )

            # ... Override motif annotations from pipeline
            zeros = np.zeros(atom_array_in.array_length(), dtype=int)
            atom_array_in.is_motif_atom_unindexed = (
                zeros  # reset unindexed annotation since those are copied already.
            )
            atom_array_in.is_motif_atom_with_fixed_coord = (
                self.select_fixed_atoms.get_mask().astype(int)
                if exists(self.select_fixed_atoms)
                else zeros
            )
            atom_array_in.is_motif_atom_with_fixed_seq = (
                ~self.select_unfixed_sequence.get_mask()
                if exists(self.select_unfixed_sequence)
                else zeros
            ).astype(int)

            # ... Subset to residues only
            atom_array_in = atom_array_in[atom_array_in.is_protein]

            # ... Set chain ID for unindexed residues as whatever the input has
            start_resid = np.max(atom_array_in.res_id) + 1
            start_chain = atom_array_in.chain_id[0]

            # ... Accumulate from input
            components_to_accumulate = unindexed_components
            atom_array = accumulate_components(
                # No accumulation of components
                components_to_accumulate=components_to_accumulate,
                indexed_tokens={},
                # Append all inputs to unindexed tokens
                unindexed_tokens=unindexed_tokens,
                atom_array_accum=[atom_array_in],
                start_chain=start_chain,
                start_resid=start_resid,
                unindexed_breaks=unindexed_breaks,
            )

        return atom_array

    # ============================================================================
    # Auxiliary functions
    # ============================================================================

    @staticmethod
    def break_unindexed(unindex: InputSelection):
        if not exists(unindex):
            return [], []

        # ... If original type was string, use that
        if isinstance(unindex.raw, str):
            unindexed_string = unindex.raw
        elif isinstance(unindex.raw, dict):
            unindexed_string = ",".join(unindex.raw.keys())
        else:
            logger.info(
                "`Unindex` provided as non-string, separate keys in dictionary will be considered separate contiguous components"
            )
            unindexed_string = ",".join(unindex.keys())

        # ... Break expected unindexed contig string
        unindexed_components, breaks = get_motif_components_and_breaks(unindexed_string)

        return unindexed_components, breaks

    # ============================================================================
    # Setter functions
    # ============================================================================

    def _append_ligand(self, atom_array, atom_array_input_annotated):
        """Append ligand if specified."""
        if exists(self.ligand):
            ligand_array = extract_ligand_array(
                atom_array_input_annotated,
                self.ligand,
                fixed_atoms={},
                set_defaults=False,
                additional_annotations=set(
                    list(atom_array.get_annotation_categories())
                    + list(atom_array_input_annotated.get_annotation_categories())
                ),
            )
            # Validate chain assignments — chain ID is leaked to the model
            # so collisions are a significant deviation from convention.
            ligand_chains = np.unique(ligand_array.chain_id)
            existing_chains = set(np.unique(atom_array.chain_id))
            overlapping = sorted(existing_chains & set(ligand_chains))
            if not self.allow_ligand_on_existing_chain:
                if overlapping:
                    raise ValueError(
                        f"Ligand chain(s) {overlapping} overlap with existing "
                        f"chain(s) {sorted(existing_chains)}. Place ligands on "
                        f"separate chains or set 'allow_ligand_on_existing_chain: "
                        f"true' to restore the old behaviour."
                    )
                # Multiple ligands must each be on their own chain.
                for chain in ligand_chains:
                    n_residues = len(
                        np.unique(ligand_array.res_id[ligand_array.chain_id == chain])
                    )
                    if n_residues > 1:
                        raise ValueError(
                            f"Multiple ligand residues on chain {chain}. Each "
                            f"ligand must be on its own chain, or set "
                            f"'allow_ligand_on_existing_chain: true' to restore "
                            f"the old behaviour."
                        )
            if self.allow_ligand_on_existing_chain:
                # Legacy behaviour: offset from protein max to avoid clashes.
                ligand_array.res_id = (
                    ligand_array.res_id
                    - np.min(ligand_array.res_id)
                    + np.max(atom_array.res_id)
                    + 1
                )
            else:
                # Reset ligand res_id to start from 1 per chain, matching
                # the convention AF3 uses in its output CIF files.
                for chain in ligand_chains:
                    mask = ligand_array.chain_id == chain
                    ligand_array.res_id[mask] = 1
            # Harmonize conditioning annotations before concatenation: biotite's
            # concatenate only preserves annotations present in ALL arrays (set
            # intersection), so mismatched optional conditioning annotations
            # (e.g. is_atom_level_hotspot) get silently dropped.
            for annot, default in OPTIONAL_CONDITIONING_VALUES.items():
                if (
                    annot in ligand_array.get_annotation_categories()
                    and annot not in atom_array.get_annotation_categories()
                ):
                    atom_array.set_annotation(
                        annot, np.full(atom_array.array_length(), default)
                    )
                elif (
                    annot in atom_array.get_annotation_categories()
                    and annot not in ligand_array.get_annotation_categories()
                ):
                    ligand_array.set_annotation(
                        annot, np.full(ligand_array.array_length(), default)
                    )
            atom_array = atom_array + ligand_array
        return atom_array

    def _apply_symmetry(self, atom_array, atom_array_input_annotated):
        """Apply symmetry transformation if specified."""
        if exists(self.symmetry) and self.symmetry.id:
            atom_array = make_symmetric_atom_array(
                atom_array,
                self.symmetry,
                sm=self.ligand,
                src_atom_array=atom_array_input_annotated,
            )
        return atom_array

    def _set_origin(self, atom_array):
        """Set origin token and initialize coordinates."""
        if self.is_partial_diffusion:
            # Partial diffusion: keep all coordinates. Centering rules:
            #   1. If symmetric: skip centering to preserve chain spacing.
            #   2. If the user supplied `ori_token` or `infer_ori_strategy`,
            #      honor it (same path as regular diffusion). Previously this
            #      branch hard-coded `ori_token=None`, silently dropping the
            #      user's request.
            #   3. Otherwise default to centering on the diffused-region COM
            #      (matches training `center_option=diffuse`). Centering on the
            #      joint target+binder COM places the binder far from origin in
            #      a frame the model never saw at training, biasing denoising
            #      to drag the binder toward the target's COM.
            if exists(self.symmetry) and self.symmetry.id:
                logger.info(
                    "Partial diffusion with symmetry: skipping COM centering to preserve chain spacing"
                )
            elif exists(self.ori_token) or exists(self.infer_ori_strategy):
                atom_array = set_com(
                    atom_array,
                    ori_token=self.ori_token,
                    infer_ori_strategy=self.infer_ori_strategy,
                )
            else:
                is_motif = atom_array.is_motif_atom_with_fixed_coord.astype(bool)
                if is_motif.any() and (~is_motif).any():
                    diffused_coord = atom_array.coord[~is_motif]
                    finite = np.isfinite(diffused_coord).all(axis=-1)
                    center = np.nan_to_num(
                        np.mean(diffused_coord[finite], axis=0)
                    )
                    atom_array.coord = atom_array.coord - center
                    logger.info(
                        f"Partial diffusion: centering on diffused-region COM ({center})."
                    )
                else:
                    atom_array = set_com(
                        atom_array, ori_token=None, infer_ori_strategy="com"
                    )
        else:
            # Standard: set ori token, zero out diffused atoms
            atom_array = set_com(
                atom_array,
                ori_token=self.ori_token,
                infer_ori_strategy=self.infer_ori_strategy,
            )
            # Diffused atoms are always initialized at origin during regular diffusion (all information removed)
            atom_array.coord[
                ~atom_array.is_motif_atom_with_fixed_coord.astype(bool)
            ] = 0.0
        return atom_array

    def _apply_globals(self, atom_array):
        # Temperature conditioning
        if exists(self.is_non_loopy):
            is_non_loopy_annot = np.zeros(atom_array.array_length(), dtype=int)
            is_motif_token = get_motif_features(atom_array)["is_motif_token"]
            diffused_region_mask = ~(is_motif_token.astype(bool))
            if exists(self.is_non_loopy):
                is_non_loopy_annot[diffused_region_mask] = (
                    1 if self.is_non_loopy else -1
                )
            atom_array.set_annotation("is_non_loopy", is_non_loopy_annot)
            atom_array.set_annotation("is_non_loopy_atom_level", is_non_loopy_annot)
        else:
            zeros = np.zeros(atom_array.array_length(), dtype=int)
            atom_array.set_annotation("is_non_loopy", zeros)
            atom_array.set_annotation("is_non_loopy_atom_level", zeros)

        if self.plddt_enhanced:
            atom_array.set_annotation(
                "ref_plddt", np.full((atom_array.array_length(),), True, dtype=int)
            )

        # Partial diffusion time annotation
        if self.is_partial_diffusion:
            atom_array.set_annotation(
                "partial_t", np.full(atom_array.shape[0], self.partial_t, dtype=float)
            )
        return atom_array

    @classmethod
    def safe_init(cls, **spec_kwargs):
        if spec_kwargs.get("dialect", 2) < 2:
            warn = (
                "Using dialect==1, which is deprecated and will be removed in future releases. "
                "Please update your input specification to dialect=2 and use the new schema if possible"
            )
            warnings.warn(warn, DeprecationWarning)
            logger.warning(warn)
            return LegacySpecification(**spec_kwargs)
        else:
            return cls(**spec_kwargs)

    def to_pipeline_input(self, example_id):
        atom_array, spec_dict = self.build(return_metadata=True)

        # ... Forward into
        data = prepare_pipeline_input_from_atom_array(atom_array)
        data["example_id"] = example_id

        # ... Wrap up with additional features
        if "extra" not in spec_dict:
            spec_dict["extra"] = {}
        spec_dict["extra"]["example_id"] = example_id
        data["specification"] = spec_dict
        return data


# ============================================================================
# APIs and utils
# ============================================================================


def prepare_pipeline_input_from_atom_array(  # see atomworks.ml.datasets.parsers.base.load_example_from_metadata_row
    atom_array_orig,
) -> dict:
    """
    Load or create an example from a metadata dictionary.
    If the file path is not provided in the metadata dictionary, create a spoofed CIF file based on the length.
    Args:
        atom_array_orig: Atom array instantiated with conditioning annotations

    Returns:
        dict: A dictionary containing the parsed row data and additional loaded CIF data.
    """
    _start_parse_time = time.time()
    # HACK: Set empty bond graph:
    if atom_array_orig.bonds is None:
        atom_array_orig.bonds = BondList(atom_array_orig.array_length())

    # Temporary spoof of chain IDs to ensure duplicates aren't dropped:
    result_dict = parse_atom_array(
        atom_array_orig,
        remove_ccds=[],
        fix_arginines=False,
        add_missing_atoms=False,
        extra_fields=INFERENCE_ANNOTATIONS,
        build_assembly=None,
        hydrogen_policy="remove",
    )
    atom_array = result_dict["asym_unit"][0]

    # HACK: Set iid information manually
    # We currently do not preserve this information from the input,
    # if you want these we'd need to remove the spoofing here
    check_has_required_conditioning_annotations(
        atom_array, required=REQUIRED_INFERENCE_ANNOTATIONS
    )
    atom_array = convert_existing_annotations_to_bool(atom_array)
    atom_array.set_annotation("chain_iid", [f"{c}_1" for c in atom_array.chain_id])
    atom_array.set_annotation("pn_unit_iid", [f"{c}_1" for c in atom_array.pn_unit_id])

    # Ensure motif annotations are removed
    atom_array.del_annotation(
        "is_motif_token"
    ) if "is_motif_token" in atom_array.get_annotation_categories() else None
    atom_array.del_annotation(
        "is_motif_atom"
    ) if "is_motif_atom" in atom_array.get_annotation_categories() else None

    data = {
        "atom_array": atom_array,  # First model
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }
    _stop_parse_time = time.time()
    data = TransformedDict(data)
    return data


def create_atom_array_from_design_specification(
    **spec_kwargs,
) -> tuple[AtomArray, dict]:
    if int(spec_kwargs.get("dialect", 2)) < 2:
        warn = (
            "Using dialect==1, which is deprecated and will be removed in future releases. "
            "Please update your input specification to dialect=2 and use the new schema if possible"
        )
        warnings.warn(warn, DeprecationWarning)
        logger.warning(warn)
        atom_array = create_atom_array_from_design_specification_legacy(**spec_kwargs)
        return atom_array, {}

    # Create input specfication and build
    spec = DesignInputSpecification(**spec_kwargs)
    atom_array, metadata = spec.build(return_metadata=True)
    return atom_array, metadata


@contextmanager
def validator_context(validator_name: str, data: dict = None):
    """Context manager for validator execution with logging."""
    logger.debug(f"Starting validator: {validator_name}")
    try:
        yield
        logger.debug(f"✓ Completed validator: {validator_name}")
    except Exception as e:
        logger.error(
            f"✗ Failed in validator: {validator_name}\n"
            f"  Error: {str(e)}\n"
            f"  Error type: {type(e).__name__}"
        )
        raise e


def create_diffused_residues(n, additional_annotations=None):
    if n <= 0:
        raise ValueError(f"Negative/null residue count ({n}) not allowed.")

    atoms = []
    [
        atoms.extend(
            [
                struc.Atom(
                    np.array([0.0, 0.0, 0.0], dtype=np.float32),
                    res_name="ALA",
                    res_id=idx,
                )
                for _ in range(5)
            ]
        )
        for idx in range(1, n + 1)
    ]
    array = struc.array(atoms)
    array.set_annotation(
        "element", np.array(["N", "C", "C", "O", "C"] * n, dtype="<U2")
    )
    array.set_annotation(
        "atom_name", np.array(["N", "CA", "C", "O", "CB"] * n, dtype="<U2")
    )
    array = set_default_conditioning_annotations(
        array, motif=False, additional=additional_annotations
    )
    array = set_common_annotations(array)
    return array


def create_motif_residue(
    token,
    strip_sidechains_by_default: bool,
):
    if strip_sidechains_by_default and token.res_name in STANDARD_AA:
        n_atoms = token.shape[0]
        diffuse_oxygen = False
        if n_atoms < 3:
            raise ValueError(
                f"Not enough data for {src_chain}{src_resid} in input atom array."
            )
        if n_atoms == 3:
            # Handle cases with N, CA, C only;
            token = token + create_o_atoms(token.copy())
            diffuse_oxygen = True  # flag oxygen for generation

        # Subset to the first 4 atoms (N, CA, C, O) only
        token = token[np.isin(token.atom_name, ["N", "CA", "C", "O"])]

        # exactly N, CA, C, O but no CB. Place CB onto idealized position and conver to ALA
        # Sequence name ALA ensures the padded atoms to be diffused from the fixed backbone
        # are placed on the CB so as to not leak the identity of the residue.
        token = token + create_cb_atoms(token.copy())

        # Sequence name must be set to ALA such that the central atom is correctly CB
        token.res_name = np.full_like(token.res_name, "ALA", dtype=token.res_name.dtype)
        token.set_annotation(
            "is_motif_atom_with_fixed_coord",
            np.where(
                np.arange(token.shape[0], dtype=int) < (4 - int(diffuse_oxygen)),
                token.is_motif_atom_with_fixed_coord,
                0,
            ),
        )

    check_has_required_conditioning_annotations(token)
    token = set_common_annotations(token)
    token.set_annotation("res_id", np.full(token.shape[0], 1))  # Reset to 1

    return token


def _polymer_link_atoms_for_residue(res_name: str) -> set[frozenset[str]]:
    """
    Return the atom-name pairs that represent canonical polymerization atoms for a residue.

    Only standard AA/DNA/RNA residues are treated as having canonical backbone links; PTMs
    and other chem comp types fall back to nonstandard handling.
    """
    if res_name in STANDARD_AA:
        return {frozenset({"C", "N"})}
    if res_name in STANDARD_DNA or res_name in STANDARD_RNA:
        return {frozenset({"O3'", "P"}), frozenset({"O3*", "P"})}  # allow legacy O3*
    return set()


def _is_standard_polymer_backbone_bond(atom_a: struc.Atom, atom_b: struc.Atom) -> bool:
    if atom_a.chain_id != atom_b.chain_id:
        return False
    if abs(atom_a.res_id - atom_b.res_id) != 1:
        return False
    atom_pair = frozenset({atom_a.atom_name, atom_b.atom_name})

    pairs_a = _polymer_link_atoms_for_residue(atom_a.res_name)
    pairs_b = _polymer_link_atoms_for_residue(atom_b.res_name)
    shared_pairs = pairs_a & pairs_b
    return atom_pair in shared_pairs


def _is_polymer_backbone_like(atom_a: struc.Atom, atom_b: struc.Atom) -> bool:
    """
    Broader backbone check that treats canonical polymer atoms (C/N for peptide,
    O3'/O3*–P for nucleic) as backbone links even when the residue itself is
    non-standard (e.g., PTR/SEP).
    """
    if atom_a.chain_id != atom_b.chain_id:
        return False
    if abs(atom_a.res_id - atom_b.res_id) != 1:
        return False
    atom_pair = frozenset({atom_a.atom_name, atom_b.atom_name})
    return atom_pair in {
        frozenset({"C", "N"}),
        frozenset({"O3'", "P"}),
        frozenset({"O3*", "P"}),
    }


def _restore_component_bonds(
    atom_array_accum: struc.AtomArray,
    src_atom_array: Optional[struc.AtomArray],
    source_to_accum_idx: Dict[int, int],
    source_idx_to_component: Dict[int, str],
    unindexed_components: set[str],
) -> struc.AtomArray:
    """
    Rehydrate bonds from the input structure onto the accumulated array.

    - Replays bonds from `src_atom_array` using the provided source→accum mappings.
    - Skips canonical polymer backbone bonds (peptide/nucleic) that are reconstructed elsewhere.
    - Protects unindexed components by disallowing cross-residue bonds that involve
      an unindexed residue (except standard backbone).
    - Emits warnings when a source bond cannot be remapped because one endpoint was
      dropped during accumulation.
    """
    if atom_array_accum.bonds is None:
        atom_array_accum.bonds = struc.BondList(atom_array_accum.array_length())

    if (
        src_atom_array is None
        or not hasattr(src_atom_array, "bonds")
        or src_atom_array.bonds is None
        or not source_to_accum_idx
    ):
        return atom_array_accum

    bonds_to_add: List[List[int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    src_bonds = np.asarray(src_atom_array.bonds.as_array(), dtype=np.int64)

    def _is_unindexed_source(idx: int) -> bool:
        component = source_idx_to_component.get(idx)
        return component in unindexed_components if component is not None else False

    def _fmt_atom(atom: struc.Atom) -> str:
        # Use a readable residue/atom separator to avoid names running together
        return f"{atom.chain_id}{atom.res_id}:{atom.res_name}_{atom.atom_name}"

    for atom_i_idx, atom_j_idx, bond_type in src_bonds:
        atom_i_idx = int(atom_i_idx)
        atom_j_idx = int(atom_j_idx)
        bond_type = int(bond_type)
        mapped_i = source_to_accum_idx.get(atom_i_idx)
        mapped_j = source_to_accum_idx.get(atom_j_idx)

        atom_i = src_atom_array[atom_i_idx]
        atom_j = src_atom_array[atom_j_idx]

        # If we only have one side of the bond, assert if the mapped atom is from an
        # unindexed component and the bond would connect across residues.
        if mapped_i is None or mapped_j is None:
            if _is_standard_polymer_backbone_bond(
                atom_i, atom_j
            ) or _is_polymer_backbone_like(atom_i, atom_j):
                continue
            if mapped_i is not None and _is_unindexed_source(atom_i_idx):
                if (
                    atom_i.chain_id != atom_j.chain_id
                    or atom_i.res_id != atom_j.res_id
                    or not (
                        _is_standard_polymer_backbone_bond(atom_i, atom_j)
                        or _is_polymer_backbone_like(atom_i, atom_j)
                    )
                ):
                    raise AssertionError(
                        f"Unsupported bond between unindexed component {atom_i.chain_id}{atom_i.res_id} "
                        f"and omitted residue {atom_j.chain_id}{atom_j.res_id}."
                    )
            if mapped_j is not None and _is_unindexed_source(atom_j_idx):
                if (
                    atom_i.chain_id != atom_j.chain_id
                    or atom_i.res_id != atom_j.res_id
                    or not (
                        _is_standard_polymer_backbone_bond(atom_i, atom_j)
                        or _is_polymer_backbone_like(atom_i, atom_j)
                    )
                ):
                    raise AssertionError(
                        f"Unsupported bond between unindexed component {atom_j.chain_id}{atom_j.res_id} "
                        f"and omitted residue {atom_i.chain_id}{atom_i.res_id}."
                    )
            # Only warn when we retained one side of a cross-residue/chain linkage
            # (e.g., glycan partner missing), not for missing intra-residue atoms.
            if (mapped_i is not None or mapped_j is not None) and (
                atom_i.chain_id != atom_j.chain_id or atom_i.res_id != atom_j.res_id
            ):
                logger.warning(
                    (
                        "Skipping non-backbone bond from source structure between %s and %s (type %d): "
                        "one atom is not present in accumulated components. "
                        "Bond cannot be inferred automatically; set it manually if needed."
                    )
                    % (_fmt_atom(atom_i), _fmt_atom(atom_j), bond_type)
                )
            continue

        # Do not connect unindexed residues to anything else for now.
        comp_i = source_idx_to_component.get(atom_i_idx)
        comp_j = source_idx_to_component.get(atom_j_idx)
        if (comp_i in unindexed_components or comp_j in unindexed_components) and (
            atom_i.chain_id != atom_j.chain_id or atom_i.res_id != atom_j.res_id
        ):
            if not (
                _is_standard_polymer_backbone_bond(atom_i, atom_j)
                or _is_polymer_backbone_like(atom_i, atom_j)
            ):
                raise AssertionError(
                    "Bonds involving unindexed residues are not yet supported."
                )
            continue

        if _is_standard_polymer_backbone_bond(
            atom_i, atom_j
        ) or _is_polymer_backbone_like(atom_i, atom_j):
            continue

        pair = (min(mapped_i, mapped_j), max(mapped_i, mapped_j))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        bonds_to_add.append([mapped_i, mapped_j, bond_type])

    bond_array = (
        np.array(bonds_to_add, dtype=np.int64)
        if bonds_to_add
        else np.empty((0, 3), dtype=np.int64)
    )
    new_bonds = struc.BondList(atom_array_accum.array_length(), bond_array)
    atom_array_accum.bonds = atom_array_accum.bonds.merge(new_bonds)
    return atom_array_accum


def _add_backbone_bonds_for_nonstandard_residues(
    atom_array_accum: struc.AtomArray,
) -> struc.AtomArray:
    """
    Add backbone/polymer bonds for cases where at least one residue is non-standard.

    Uses `atomworks.io.utils.bonds.get_inferred_polymer_bonds`, which consults CCD
    chem-comp metadata to decide the correct polymerization atoms (C/N, CG/N, etc.).
    Only bonds involving at least one non-standard residue are added; standard
    AA/DNA/RNA pairs are assumed to already carry their backbone bonds.
    """
    if atom_array_accum.bonds is None:
        atom_array_accum.bonds = struc.BondList(atom_array_accum.array_length())

    unindexed_mask = (
        atom_array_accum.get_annotation("is_motif_atom_unindexed")
        if "is_motif_atom_unindexed" in atom_array_accum.get_annotation_categories()
        else np.zeros(atom_array_accum.array_length(), dtype=bool)
    )

    existing_pairs = {
        (min(a, b), max(a, b)) for a, b, _ in atom_array_accum.bonds.as_array()
    }
    bonds_to_add: List[List[int]] = []

    inferred_bonds, _ = get_inferred_polymer_bonds(atom_array_accum)
    for atom_i_idx, atom_j_idx, bond_type in inferred_bonds:
        atom_i_idx = int(atom_i_idx)
        atom_j_idx = int(atom_j_idx)

        # Do not connect unindexed residues across residue boundaries
        if (unindexed_mask[atom_i_idx] or unindexed_mask[atom_j_idx]) and (
            atom_array_accum.chain_id[atom_i_idx]
            != atom_array_accum.chain_id[atom_j_idx]
            or atom_array_accum.res_id[atom_i_idx]
            != atom_array_accum.res_id[atom_j_idx]
        ):
            continue

        # Only synthesize bonds when at least one residue is non-standard; standard
        # backbone bonds should already exist.
        if _is_standard_polymer_backbone_bond(
            atom_array_accum[atom_i_idx], atom_array_accum[atom_j_idx]
        ):
            continue

        pair = (min(atom_i_idx, atom_j_idx), max(atom_i_idx, atom_j_idx))
        if pair in existing_pairs:
            continue
        existing_pairs.add(pair)
        bonds_to_add.append([pair[0], pair[1], int(bond_type)])

    if bonds_to_add:
        new_bonds = struc.BondList(
            atom_array_accum.array_length(), np.array(bonds_to_add, dtype=np.int64)
        )
        atom_array_accum.bonds = atom_array_accum.bonds.merge(new_bonds)
    return atom_array_accum


def _sort_bonds(atom_array_accum: struc.AtomArray) -> struc.AtomArray:
    """Sort bonds deterministically by atom indices then bond type."""
    bonds_arr = atom_array_accum.bonds.as_array().copy()
    # ensure lower index first
    swap_mask = bonds_arr[:, 0] > bonds_arr[:, 1]
    bonds_arr[swap_mask, :2] = bonds_arr[swap_mask][:, [1, 0]]
    order = np.lexsort((bonds_arr[:, 2], bonds_arr[:, 1], bonds_arr[:, 0]))
    bonds_arr = bonds_arr[order]
    atom_array_accum.bonds = struc.BondList(
        atom_array_accum.array_length(), bonds_arr.astype(np.int64)
    )
    return atom_array_accum


def accumulate_components(
    components_to_accumulate: List[Union[str, int]],
    *,
    # Tokens from input
    indexed_tokens: Dict[str, AtomArray],
    unindexed_tokens: Dict[str, AtomArray],
    # Additional parameters
    atom_array_accum=[],
    start_chain: str = "A",
    start_resid: int = 1,
    unindexed_breaks: Optional[List[bool]] = [],
    src_atom_array: Optional[AtomArray] = None,
    strip_sidechains_by_default: bool = False,
    **kwargs,
) -> AtomArray:
    # ... Create list of components
    assert (
        x := (set(list(indexed_tokens.keys()) + list(unindexed_tokens.keys())))
    ).issubset(
        (y := set(components_to_accumulate))
    ), "Unindexed and indexed set {} is not subset of components to accumulate {}".format(
        x, y
    )
    all_tokens = indexed_tokens | unindexed_tokens
    all_annots = []
    [
        all_annots.extend(list(tok.get_annotation_categories()))
        for tok in all_tokens.values()
    ]
    all_annots = set(all_annots)
    atom_array_accum = [] if atom_array_accum is None else atom_array_accum
    unindexed_breaks = (
        [None] * len(components_to_accumulate)
        if unindexed_breaks is None
        else unindexed_breaks
    )

    # ... For-loop accum variables
    unindexed_components_started = (
        False  # once one unindexed component is added, stop adding diffused residues
    )
    chain = start_chain
    res_id = start_resid
    molecule_id = 0
    source_to_accum_idx: Dict[int, int] = {}
    source_idx_to_component: Dict[int, str] = {}
    unindexed_component_names = set(unindexed_tokens.keys())
    current_accum_idx = sum(len(arr) for arr in atom_array_accum)

    # ... Insert contig information one- by one-
    assert len(components_to_accumulate) == len(
        unindexed_breaks
    ), "Mismatch in number of components to accumulate and breaks"
    for component, is_break in zip(components_to_accumulate, unindexed_breaks):
        src_indices = None
        if exists(is_break) and is_break:
            if not unindexed_components_started:
                chain = start_chain
                res_id = start_resid
                unindexed_components_started = True

        if component == "/0":
            # Reset iterators on next chain
            chain = chr(ord(chain) + 1)
            molecule_id += 1
            res_id = 1
            continue

        # ... Create array to insert
        if str(component)[0].isalpha():  # motif (e.g. "A22")
            n = 1

            # ... Fetch the motif residue
            token = all_tokens[component]
            if src_atom_array is not None:
                src_mask = fetch_mask_from_idx(component, atom_array=src_atom_array)
                src_indices = np.where(src_mask)[0]
                # try:
                # except ComponentValidationError as e:
                #     src_indices = None
                #     print(e)

            # ... Ensure motif residues are set properly
            token = create_motif_residue(
                token, strip_sidechains_by_default=strip_sidechains_by_default
            )

            # ... Insert breakpoint when break clause is met
            if exists(is_break) and is_break:
                token.set_annotation(
                    "is_motif_atom_unindexed_motif_breakpoint",
                    np.ones(token.shape[0], dtype=int),
                )
            else:
                token.set_annotation(
                    "is_motif_atom_unindexed_motif_breakpoint",
                    np.zeros(token.shape[0], dtype=int),
                )
        else:
            ## foundry components update sends P for protein tokens
            n = int(component[:-1])
            # ... Skip if none or unindexed
            if n == 0 or unindexed_components_started:
                res_id += n
                continue

            # ... Create diffused residues
            token = create_diffused_residues(n, all_annots)

        # ... Set index of insertion
        token = set_indices(
            array=token,
            chain=chain,
            res_id_start=res_id,
            molecule_id=molecule_id,
            component=component,
        )

        assert (
            len(get_token_starts(token)) == n
        ), f"Mismatch in number of residues: expected {n}, got {len(get_token_starts(token))} in \n{token}"

        if (
            src_atom_array is not None
            and str(component)[0].isalpha()
            and src_indices is not None
            and len(src_indices) == len(token)
        ):
            for i, src_idx in enumerate(src_indices):
                source_to_accum_idx[int(src_idx)] = current_accum_idx + i
                source_idx_to_component[int(src_idx)] = str(component)

        # ... Insert & Increment residue ID
        atom_array_accum.append(token)
        res_id += n
        current_accum_idx += len(token)

    # ... Concatenate all components
    atom_array_accum = struc.concatenate(atom_array_accum)
    atom_array_accum.set_annotation("pn_unit_iid", atom_array_accum.chain_id)
    should_restore_bonds = (
        src_atom_array is not None
        and bool(source_to_accum_idx)
        and _check_has_backbone_connections_to_nonstandard_residues(
            atom_array_accum, src_atom_array
        )
    )
    if should_restore_bonds:
        atom_array_accum = _restore_component_bonds(
            atom_array_accum=atom_array_accum,
            src_atom_array=src_atom_array,
            source_to_accum_idx=source_to_accum_idx,
            source_idx_to_component=source_idx_to_component,
            unindexed_components=unindexed_component_names,
        )
        atom_array_accum = _add_backbone_bonds_for_nonstandard_residues(
            atom_array_accum=atom_array_accum
        )
        atom_array_accum = _sort_bonds(atom_array_accum)

    # Reset res_id for unindexed residues to avoid duplicates (ridiculously long lines of code, cleanup later)
    if np.any(atom_array_accum.is_motif_atom_unindexed.astype(bool)) and not np.all(
        atom_array_accum.is_motif_atom_unindexed.astype(bool)
    ):
        max_id = np.max(
            atom_array_accum[
                ~atom_array_accum.is_motif_atom_unindexed.astype(bool)
            ].res_id
        )
        min_id_udx = np.min(
            atom_array_accum[
                atom_array_accum.is_motif_atom_unindexed.astype(bool)
            ].res_id
        )
        atom_array_accum.res_id[
            atom_array_accum.is_motif_atom_unindexed.astype(bool)
        ] += max_id - min_id_udx + 1

    # ... Bonds
    if atom_array_accum.bonds is None:
        atom_array_accum.bonds = BondList(atom_array_accum.array_length())
    return atom_array_accum


def ensure_input_is_abspath(args: Dict[str, Any], path: PathLike | None):
    """
    Ensures the input source is an absolute path if exists, if not it will convert

    args:
        args: Inference specification for atom array
        path: None or file to which the input is relative to.
    """
    if isinstance(args, str):
        raise ValueError(
            "Expected args to be a dictionary, got a string: {}. If you are using an input JSON ensure it contains dictionaries of arguments".format(
                args
            )
        )
    if "input" not in args or not exists(args["input"]):
        return args
    input = str(args["input"])
    if not os.path.isabs(input):
        if path is None:
            raise ValueError(
                "Input path is relative, but no base path was provided to resolve it against."
            )
        input = os.path.abspath(os.path.join(os.path.dirname(str(path)), input))
        logger.info(
            f"Input source path is relative, converted to absolute path: {input}"
        )
        args["input"] = input
    return args
