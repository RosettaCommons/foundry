"""
CLI, JSON config handling, and structure-aware inputs for MPNN inference.

This module implements:

- Argument parser and CLI -> JSON builder.
- MPNNInferenceInput construction utilities.
"""

import argparse
import ast
import copy
import json
import logging
import re
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.io import parse
from atomworks.io.parser import STANDARD_PARSER_ARGS, parse_atom_array
from atomworks.io.utils.atom_array_plus import (
    AtomArrayPlus,
    as_atom_array_plus,
)
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.utils.token import get_token_starts, spread_token_wise
from biotite.structure import AtomArray
from mpnn.transforms.feature_aggregation.token_encodings import MPNN_TOKEN_ENCODING

logger = logging.getLogger(__name__)

MPNN_GLOBAL_INFERENCE_DEFAULTS: dict[str, Any] = {
    # Top-level Config JSON
    "config_json": None,
    # Model Type and Weights
    "checkpoint_path": None,
    "model_type": None,
    "is_legacy_weights": None,
    # Output controls
    "out_directory": None,
    "write_fasta": True,
    "write_structures": True,
}

MPNN_PER_INPUT_INFERENCE_DEFAULTS: dict[str, Any] = {
    # Structure Path and Name
    "structure_path": None,
    "name": None,
    # Sampling Parameters
    "seed": None,
    "batch_size": 1,
    "number_of_batches": 1,
    # Parser Overrides
    "remove_ccds": [],
    "remove_waters": None,
    # Pipeline Setup Overrides
    "occupancy_threshold_sidechain": 0.0,
    "occupancy_threshold_backbone": 0.0,
    "undesired_res_names": [],
    # Scalar User Settings
    "structure_noise": 0.0,
    "decode_type": "auto_regressive",
    "causality_pattern": "auto_regressive",
    "initialize_sequence_embedding_with_ground_truth": False,
    "features_to_return": None,
    # Only applicable for LigandMPNN
    "atomize_side_chains": False,
    # Design scope - if all None, design all residues
    "fixed_residues": None,
    "designed_residues": None,
    "fixed_chains": None,
    "designed_chains": None,
    # Bias, Omission, and Pair Bias
    "bias": None,
    "bias_per_residue": None,
    "omit": ["UNK"],
    "omit_per_residue": None,
    "pair_bias": None,
    "pair_bias_per_residue_pair": None,
    # Temperature
    "temperature": 0.1,
    "temperature_per_residue": None,
    # Symmetry
    "symmetry_residues": None,
    "symmetry_residues_weights": None,
    "homo_oligomer_chains": None,
}


################################################################################
# CLI / Arg parser
################################################################################


def str2bool(v: str) -> bool:
    """Helper function to parse boolean CLI args."""
    if v in ("True", "1"):
        return True
    elif v in ("False", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def none_or_type(v: Any, specified_type) -> Any | None:
    """
    CLI type parser that turns 'None' into None. Otherwise, returns the value
    cast to the given type. This function is useful for the parser/pipeline
    override arguments where None has a special meaning (use default behavior).
    """
    if v == "None":
        return None
    return specified_type(v)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the MPNN inference arg parser."""
    parser = argparse.ArgumentParser(
        description="MPNN JSON-driven inference CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------- Top-level Config JSON ---------------- #
    parser.add_argument(
        "--config_json",
        type=str,
        help=(
            "Path to existing JSON config file. When provided, all other CLI "
            "flags are parsed but ignored."
        ),
        default=MPNN_GLOBAL_INFERENCE_DEFAULTS["config_json"],
    )

    # ---------------- Model Type and Weights ---------------- #
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["protein_mpnn", "ligand_mpnn"],
        help="Model type to use.",
        default=MPNN_GLOBAL_INFERENCE_DEFAULTS["model_type"],
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to model checkpoint.",
        default=MPNN_GLOBAL_INFERENCE_DEFAULTS["checkpoint_path"],
    )
    parser.add_argument(
        "--is_legacy_weights",
        type=str2bool,
        choices=[True, False],
        help="Whether to interpret checkpoint as legacy-weight ordering.",
        default=MPNN_GLOBAL_INFERENCE_DEFAULTS["is_legacy_weights"],
    )

    # --------------- Output controls ---------------- #
    parser.add_argument(
        "--out_directory",
        type=str,
        help="Output directory for CIF/FASTA.",
        default=MPNN_GLOBAL_INFERENCE_DEFAULTS["out_directory"],
    )
    parser.add_argument(
        "--write_fasta",
        type=str2bool,
        choices=[True, False],
        help="Whether to write FASTA outputs.",
        default=MPNN_GLOBAL_INFERENCE_DEFAULTS["write_fasta"],
    )
    parser.add_argument(
        "--write_structures",
        type=str2bool,
        choices=[True, False],
        help="Whether to write designed structures (CIF).",
        default=MPNN_GLOBAL_INFERENCE_DEFAULTS["write_structures"],
    )

    # ---------------- Structure Path and Name ---------------- #
    parser.add_argument(
        "--structure_path",
        type=str,
        help="Path to structure file (CIF or PDB).",
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["structure_path"],
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Optional name / label for the input.",
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["name"],
    )

    # ---------------- Sampling Parameters ---------------- #
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for sampling.",
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["seed"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help=(
            "Batch size for sampling. At inference, this also controls "
            "the effective repeat_sample_num passed to the pipeline."
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["batch_size"],
    )
    parser.add_argument(
        "--number_of_batches",
        type=int,
        help="Number of batches of size batch_size to draw.",
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["number_of_batches"],
    )

    # ---------------- Parser overrides ---------------- #
    parser.add_argument(
        "--remove_ccds",
        type=lambda v: none_or_type(v, str),
        help=(
            "Comma-separated list of CCD residue names to remove as solvents/"
            "crystallization components during parsing "
            "(overrides STANDARD_PARSER_ARGS). 'None' has special behavior: "
            "use the parser default behavior."
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["remove_ccds"],
    )
    parser.add_argument(
        "--remove_waters",
        type=lambda v: none_or_type(v, str2bool),
        choices=[True, False, None],
        help=(
            "If set, override the parser default for removing water-like "
            "residues (overrides STANDARD_PARSER_ARGS). 'None' "
            "has special behavior: use the parser default behavior."
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["remove_waters"],
    )

    # ---------------- Pipeline Setup Overrides ---------------- #
    parser.add_argument(
        "--occupancy_threshold_sidechain",
        type=lambda v: none_or_type(v, float),
        help=(
            "Sidechain occupancy threshold used in the MPNN pipeline. 'None' "
            "has special behavior: use the pipeline default behavior."
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["occupancy_threshold_sidechain"],
    )
    parser.add_argument(
        "--occupancy_threshold_backbone",
        type=lambda v: none_or_type(v, float),
        help=(
            "Backbone occupancy threshold used in the MPNN pipeline. 'None' "
            "has special behavior: use the pipeline default behavior."
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["occupancy_threshold_backbone"],
    )
    parser.add_argument(
        "--undesired_res_names",
        type=lambda v: none_or_type(v, str),
        help=(
            "JSON or comma-separated list of residue names to treat as "
            "undesired in the pipeline. 'None' has special behavior: use the "
            "pipeline default behavior."
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["undesired_res_names"],
    )

    # ---------------- Scalar User Settings ---------------- #
    parser.add_argument(
        "--structure_noise",
        type=float,
        help=("Structure noise (Angstroms) used in user settings."),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["structure_noise"],
    )
    parser.add_argument(
        "--decode_type",
        type=str,
        choices=["auto_regressive", "teacher_forcing"],
        help=(
            "Decoding type for MPNN inference. "
            "\t- auto_regressive: use previously predicted residues for all "
            "previous positions when predicting each residue. This is the "
            "default for inference."
            "\t- teacher_forcing: use ground-truth residues from the structure "
            "for all previous positions when predicting each residue."
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["decode_type"],
    )
    parser.add_argument(
        "--causality_pattern",
        type=str,
        choices=[
            "auto_regressive",
            "unconditional",
            "conditional",
            "conditional_minus_self",
        ],
        help=(
            "Causality pattern for decoding. "
            "\t- auto_regressive: each position attends to the sequence and "
            "decoder representation of all previously decoded positions. This "
            "is the default for inference."
            "\t- unconditional: each position does not attend to the sequence "
            "or decoder representation of any other positions (encoder "
            "representations only)."
            "\t- conditional: each position attends to the sequence and "
            "decoder representation of all other positions."
            "\t- conditional_minus_self: each position attends to the sequence "
            "and decoder representation of all other positions, except for "
            "itself (as a destination node)."
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["causality_pattern"],
    )
    parser.add_argument(
        "--initialize_sequence_embedding_with_ground_truth",
        type=str2bool,
        choices=[True, False],
        help=(
            "Whether to initialize the sequence embedding with ground truth "
            "residues from the input structure. "
            "\t- False: initialize the sequence embedding with zeros. If doing "
            "auto-regressive decoding, initialize S_sampled with unknown "
            "residues. This is the default for inference."
            "\t- True: initialize the sequence embedding with the ground truth "
            "sequence from the input structure. If doing auto-regressive "
            "decoding, also initialize S_sampled with the ground truth. This "
            "affects the pair bias application."
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS[
            "initialize_sequence_embedding_with_ground_truth"
        ],
    )
    parser.add_argument(
        "--features_to_return",
        type=str,
        help=(
            "JSON dict for features_to_return; "
            'e.g. \'{"input_features": '
            '["mask_for_loss"], "decoder_features": ["log_probs"]}\''
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["features_to_return"],
    )
    # Only applicable for LigandMPNN.
    parser.add_argument(
        "--atomize_side_chains",
        type=str2bool,
        choices=[True, False],
        help=(
            "Whether to atomize side chains of fixed residues. Only applicable "
            "for LigandMPNN."
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["atomize_side_chains"],
    )

    # ---------------- Design scope (mutually exclusive) ---------------- #
    design_group = parser.add_mutually_exclusive_group(required=False)
    design_group.add_argument(
        "--fixed_residues",
        type=str,
        help=(
            'List of residue IDs to fix: e.g. \'["A35","B40","C52"]\' or "A35,B40,C52"'
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["fixed_residues"],
    )
    design_group.add_argument(
        "--designed_residues",
        type=str,
        help=(
            "List of residue IDs to design: "
            'e.g. \'["A35","B40","C52"]\' or "A35,B40,C52"'
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["designed_residues"],
    )
    design_group.add_argument(
        "--fixed_chains",
        type=str,
        help=('List of chain IDs to fix: e.g. \'["A","B"]\' or "A,B"'),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["fixed_chains"],
    )
    design_group.add_argument(
        "--designed_chains",
        type=str,
        help=('List of chain IDs to design: e.g. \'["A","B"]\' or "A,B"'),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["designed_chains"],
    )

    # ---------------- Bias, Omission, and Pair Bias ---------------- #
    parser.add_argument(
        "--bias",
        type=str,
        help='Bias dict: e.g. \'{"ALA": -1.0, "GLY": 0.5}\'',
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["bias"],
    )
    parser.add_argument(
        "--bias_per_residue",
        type=str,
        help=(
            'Per-residue bias dict: e.g. \'{"A35": {"ALA": -2.0}}\'. Overwrites --bias.'
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["bias_per_residue"],
    )
    parser.add_argument(
        "--omit",
        type=str,
        help=('List of residue types to omit: e.g. \'["ALA","GLY","UNK"]\'.'),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["omit"],
    )
    parser.add_argument(
        "--omit_per_residue",
        type=str,
        help=(
            "Per-residue list of residue types to omit: "
            'e.g. \'{"A35": ["ALA","GLY","UNK"]}\'. Overwrites --omit.'
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["omit_per_residue"],
    )
    parser.add_argument(
        "--pair_bias",
        type=str,
        help=(
            "Controls the bias applied due to residue selections at "
            "neighboring positions: "
            '\'{"ALA": {"GLY": -0.5}, "GLY": {"ALA": -0.5}}\''
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["pair_bias"],
    )
    parser.add_argument(
        "--pair_bias_per_residue_pair",
        type=str,
        help=(
            "Per-residue-pair dict for controlling bias due to residue "
            "selections at neighboring positions: "
            '\'{"A35": {"B40": {"ALA": {"GLY": -1.0}}}}\' . Overwrites '
            "--pair_bias. Note that this is NOT applied symmetrically; if "
            "the outer residue ID corresponds to the first token; the inner "
            "residue ID corresponds to the second token. This should be read "
            'as follows: for residue pair (i,j) (e.g. ("A35","B40")), the '
            "inner dictionaries dictate that if residue i is assigned as the "
            'first token (e.g. "ALA"), then the bias for assigning residue j '
            'is the innermost dict (e.g. {"GLY": -1.0} ).'
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["pair_bias_per_residue_pair"],
    )

    # ---------------- Temperature ---------------- #
    parser.add_argument(
        "--temperature",
        type=float,
        help=("Temperature for sampling."),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["temperature"],
    )
    parser.add_argument(
        "--temperature_per_residue",
        type=str,
        help=(
            "Per-residue temperature dict: e.g. '{\"A35\": 0.1}'. Overwrites "
            "--temperature."
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["temperature_per_residue"],
    )

    # ---------------- Symmetry ---------------- #
    sym_group = parser.add_mutually_exclusive_group(required=False)
    sym_group.add_argument(
        "--symmetry_residues",
        type=str,
        help=(
            "Residue-based symmetry groups, each a list of residue IDs. "
            "Example: "
            '\'[["A35","B35"],["A40","B40","C40"]]\''
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["symmetry_residues"],
    )
    sym_group.add_argument(
        "--homo_oligomer_chains",
        type=str,
        help=(
            "Homo-oligomer chain groups, each a list of chain IDs. "
            "Within each group, chains must have the same number of residues "
            "in the same order; residues at matching positions across chains "
            "are treated as symmetry-equivalent. Example: "
            '\'[["A","B","C"]]\''
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["homo_oligomer_chains"],
    )

    # Symmetry weights
    parser.add_argument(
        "--symmetry_residues_weights",
        type=str,
        help=(
            "Optional list of symmetry weights matching the shape of "
            "symmetry_residues. Example: "
            "'[[1.0, 1.0], [1.0, 0.5, -0.5]]'. "
            "Ignored if homo_oligomer_chains is used."
        ),
        default=MPNN_PER_INPUT_INFERENCE_DEFAULTS["symmetry_residues_weights"],
    )

    return parser


###############################################################################
# JSON builder
###############################################################################


def parse_json_like(value: Any) -> Any:
    """Parse a JSON-like string into a Python object.

    Tries JSON first, then falls back to ast.literal_eval for
    simple Python literals, and finally comma-separated lists.

    If value is not a string (e.g. already parsed from JSON), it is
    returned unchanged.

    Args:
        value: The input to parse.
    Returns:
        Any: The parsed Python object, or the original value if it is
        not a string.
            - None -> None
            - non-str -> returned unchanged
            - JSON-like string -> dict / list / ...
            - python literal string (list, dict, etc.) -> list | dict | ...
            - comma-separated list string -> list[str]
    """
    # Pass through None or non-string values unchanged.
    if value is None or not isinstance(value, str):
        return value

    # Try JSON loading first.
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    # Fallback: try Python literal eval
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass

    # Fallback: treat as comma-separated list of strings
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]

    # Single string value.
    return value


def parse_list_like(value: str | None) -> list[Any] | None:
    """Parse list-like CLI strings into Python lists."""
    # First, try JSON-like parsing.
    parsed = parse_json_like(value)

    # Handle None and regular list.
    if parsed is None:
        return None
    if isinstance(parsed, list):
        return parsed

    # If single value, return singleton list.
    return [parsed]


def _absolute_path_or_none(path_str: str | None) -> str | None:
    """
    Convert a path string to an absolute path if the string is not None or
    empty.
    """
    if not path_str:
        return None
    return str(Path(path_str).expanduser().resolve())


def cli_to_json(args: argparse.Namespace) -> dict[str, Any]:
    """Convert CLI args into the top-level JSON config dict."""
    # If a config JSON is provided, load and return it directly. Ignore the
    # other CLI args.
    if args.config_json:
        config_path = _absolute_path_or_none(args.config_json)
        with open(config_path, "r") as f:
            return json.load(f)

    # Build a single-input JSON object from CLI
    if (
        args.model_type is None
        or args.checkpoint_path is None
        or args.is_legacy_weights is None
        or args.structure_path is None
    ):
        raise ValueError(
            "When --config_json is not provided, "
            "--model_type, "
            "--checkpoint_path, "
            "--is_legacy_weights, "
            "--structure_path "
            "must all be specified."
        )

    config: dict[str, Any] = {
        # Model Type and Weights
        "model_type": args.model_type,
        "checkpoint_path": args.checkpoint_path,
        "is_legacy_weights": args.is_legacy_weights,
        # Output controls
        "out_directory": args.out_directory,
        "write_fasta": args.write_fasta,
        "write_structures": args.write_structures,
        # Singleton inputs list (CLI only supports single input at a time).
        "inputs": [
            {
                # Structure Path and Name
                "structure_path": args.structure_path,
                "name": args.name,
                # Sampling Parameters
                "seed": args.seed,
                "batch_size": args.batch_size,
                "number_of_batches": args.number_of_batches,
                # Parser Overrides
                "remove_ccds": parse_list_like(args.remove_ccds),
                "remove_waters": args.remove_waters,
                # Pipeline Setup Overrides
                "occupancy_threshold_sidechain": args.occupancy_threshold_sidechain,
                "occupancy_threshold_backbone": args.occupancy_threshold_backbone,
                "undesired_res_names": parse_list_like(args.undesired_res_names),
                # Scalar User Settings
                "structure_noise": args.structure_noise,
                "decode_type": args.decode_type,
                "causality_pattern": args.causality_pattern,
                "initialize_sequence_embedding_with_ground_truth": args.initialize_sequence_embedding_with_ground_truth,
                "features_to_return": parse_json_like(args.features_to_return),
                # Only applicable for LigandMPNN
                "atomize_side_chains": args.atomize_side_chains,
                # Design scope - if all None, design all residues
                "fixed_residues": parse_list_like(args.fixed_residues),
                "designed_residues": parse_list_like(args.designed_residues),
                "fixed_chains": parse_list_like(args.fixed_chains),
                "designed_chains": parse_list_like(args.designed_chains),
                # Bias, Omission, and Pair Bias
                "bias": parse_json_like(args.bias),
                "bias_per_residue": parse_json_like(args.bias_per_residue),
                "omit": parse_json_like(args.omit),
                "omit_per_residue": parse_json_like(args.omit_per_residue),
                "pair_bias": parse_json_like(args.pair_bias),
                "pair_bias_per_residue_pair": parse_json_like(
                    args.pair_bias_per_residue_pair
                ),
                # Temperature
                "temperature": args.temperature,
                "temperature_per_residue": parse_json_like(
                    args.temperature_per_residue
                ),
                # Symmetry
                "symmetry_residues": parse_json_like(args.symmetry_residues),
                "symmetry_residues_weights": parse_json_like(
                    args.symmetry_residues_weights
                ),
                "homo_oligomer_chains": parse_json_like(args.homo_oligomer_chains),
            }
        ],
    }

    return config


###############################################################################
# MPNNInferenceInput
###############################################################################


@dataclass
class MPNNInferenceInput:
    """Container for structure + input_dict passed into inference."""

    atom_array: AtomArray
    input_dict: dict[str, Any]

    @staticmethod
    def from_atom_array_and_dict(
        *,
        atom_array: AtomArray | None = None,
        input_dict: dict[str, Any] | None = None,
    ) -> "MPNNInferenceInput":
        """Construct from an optional AtomArray and/or input dict.

        This method is responsible for per-input sanitization and defaulting.

        NOTE: if the user provides both an atom array and an input dictionary,
        the atom array is treated as the authoritative source for annotations.
        If the user passes an atom array with annotations such as:
            - mpnn_designed_residue_mask
            - mpnn_temperature
            - mpnn_bias
            - mpnn_symmetry_equivalence_group
            - mpnn_symmetry_weight
            - mpnn_pair_bias
        then those annotations will be used directly, and any corresponding
        fields from the input dictionary will be ignored. If you would like
        to override those annotations, please either do so in the atom array
        or delete the annotations from the atom array before passing it in.
        """
        # Copy input dictionary.
        input_dict = copy.deepcopy(input_dict) if input_dict is not None else dict()

        # Copy atom array.
        atom_array = atom_array.copy() if atom_array is not None else None
        parser_output = parse_atom_array(atom_array) if atom_array is not None else {}
        atom_array = (
            parser_output["assemblies"]["1"][0]
            if len(parser_output.get("assemblies", {})) > 0
            else None
        )

        # Validate the input dictionary.
        MPNNInferenceInput._validate_all(
            input_dict=input_dict,
            require_structure_path=(atom_array is None),
        )

        # Apply centralized defaults (in place), without overwriting
        # user-provided values.
        MPNNInferenceInput.apply_defaults(input_dict)

        # Process structure_path, name, and repeat_sample_num (in place).
        MPNNInferenceInput.post_process_inputs(input_dict)

        # Construct AtomArray if not provided.
        if atom_array is None:
            atom_array = MPNNInferenceInput.build_atom_array(input_dict)

        # Annotate the atom array with per-residue information from the
        # input dictionary.
        annotated = MPNNInferenceInput.annotate_atom_array(atom_array, input_dict)
        logger.info(f"Annotated AtomArray has {annotated.array_length()} atoms ")
        return MPNNInferenceInput(atom_array=annotated, input_dict=input_dict)

    @staticmethod
    def _parse_id(
        id_str: str,
        res_num_required: bool = False,
        res_num_allowed: bool = True,
    ) -> tuple[str, int | None, str | None]:
        """
        Parse flexible id strings into (chain_id, res_num, insertion_code).

        Supported formats
        -----------------
        - '<chain>'                         (e.g. 'A', 'AB')
        - '<chain><integer_res_num>'        (e.g. 'A35', 'AB12')
        - '<chain><integer_res_num><icode>' (e.g. 'A35B', 'AB12C')

        Args:
            id_str (str): chain/res_num/insertion_code string.
            res_num_required (bool): whether a residue number is required.
            res_num_allowed (bool): whether a residue number is allowed.
        Returns:
            tuple[str, int | None, str | None]: the parsed
            (chain_id, res_num, insertion_code), where res_num and/or
            insertion_code can be None if not provided.

        Examples
        --------
        'A'        -> ('A',   None, None)
        'AB'       -> ('AB',  None, None)
        'A35'      -> ('A',   35,   None)
        'A35B'     -> ('A',   35,   'B')
        'AB12C'    -> ('AB',  12,   'C')
        """
        # Match:
        #   [A-Za-z]+   : 1+ letters for chain ID
        #   (\d+)?      : optional integer residue number
        #   ([A-Za-z]*) : optional insertion code (0+ letters)
        m = re.fullmatch(r"([A-Za-z]+)(\d+)?([A-Za-z]*)", id_str)

        # Check for valid format.
        if not m:
            raise ValueError(
                f"ID '{id_str}' must look like "
                "'<letters>', '<letters><number>', or "
                "'<letters><number><letters>'."
            )

        # Extract matched groups.
        chain_id, res_num_str, insertion_code_str = m.groups()

        # Handle residue number.
        if res_num_str is None:
            if res_num_required:
                raise ValueError(f"ID '{id_str}' must contain a residue number.")
            res_num = None
        else:
            try:
                res_num = int(res_num_str)
            except ValueError as exc:
                raise ValueError(
                    f"ID '{id_str}' must contain a valid integer "
                    "residue index after the chain ID."
                ) from exc

            if not res_num_allowed:
                raise ValueError(
                    f"ID '{id_str}' is not allowed to contain a residue number."
                )

        # Handle insertion code (None or "" mapped to None).
        if not insertion_code_str:
            insertion_code = None
        else:
            insertion_code = insertion_code_str

        return chain_id, res_num, insertion_code

    @staticmethod
    def _mask_from_ids(
        atom_array: AtomArray,
        targets: list[str],
    ) -> np.ndarray:
        """
        Return a boolean mask over entries in 'atom_array' matching any ID
        specifier in 'targets'.

        Each target string can be one of:
        - '<chain>'                         (e.g. 'A')
        - '<chain><integer_res_num>'        (e.g. 'A35')
        - '<chain><integer_res_num><icode>' (e.g. 'A35B')

        Args:
            atom_array (AtomArray): The AtomArray to mask.
            targets (list[str]): List of ID strings to match.

        Matching rules
        --------------
        - If only chain is provided: match all entries in that chain.
        - If chain + res_num:
            * requires that 'atom_array' has a 'res_id' annotation/field.
            * additionally require residue-number equality.
        - If chain + res_num + icode:
            * requires both 'res_id' and 'ins_code' to be present.
            * additionally require insertion-code equality.

        Safety checks
        -------------
        - If a target specifies a residue number but 'res_id' is missing, a
          ValueError is raised.
        - If a target specifies an insertion code but 'ins_code' is missing, a
          ValueError is raised.
        - If ANY target matches zero entries in 'atom_array', a ValueError is
          raised to avoid silently ending up with an empty specification.

        Raises
        ------
        ValueError
            For malformed IDs, missing required fields, or IDs that match
            no entries in 'atom_array'.
        """
        mask = np.zeros(atom_array.array_length(), dtype=bool)

        chain_ids = atom_array.chain_id
        res_ids = getattr(atom_array, "res_id", None)
        ins_codes = getattr(atom_array, "ins_code", None)

        for id_str in targets:
            chain_id, res_num, insertion_code = MPNNInferenceInput._parse_id(
                id_str, res_num_required=False, res_num_allowed=True
            )

            # Always constrain by chain.
            local_mask = chain_ids == chain_id

            # Optionally constrain by residue number.
            if res_num is not None:
                if res_ids is None:
                    raise ValueError(
                        f"ID '{id_str}' specifies a residue number, but "
                        "the provided AtomArray does not have a 'res_id' "
                        "annotation loaded."
                    )
                local_mask &= res_ids == res_num

            # Optionally constrain by insertion code.
            if insertion_code is not None:
                if ins_codes is None:
                    raise ValueError(
                        f"ID '{id_str}' specifies an insertion code, but "
                        "the provided AtomArray does not have an 'ins_code' "
                        "annotation loaded."
                    )
                local_mask &= ins_codes == insertion_code

            # Disallow IDs that match nothing
            if not np.any(local_mask):
                raise ValueError(
                    f"ID '{id_str}' did not match any entries in the structure."
                )

            mask |= local_mask

        return mask

    @staticmethod
    def _validate_structure_path_and_name(
        input_dict: dict[str, Any],
        require_structure_path: bool,
    ) -> None:
        """
        Validate structure_path and name fields.

        Args:
            input_dict (dict[str, Any]): Input dictionary containing fields.
            require_structure_path (bool): If True, structure_path must be
                provided and must exist on disk. This should typically be True
                when 'atom_array' is not provided.
        """
        structure_path = input_dict.get("structure_path")

        # Check presence of structure_path if required.
        if require_structure_path and structure_path is None:
            raise ValueError(
                "structure_path is required when atom_array is not provided."
            )

        # Check structure_path validity if provided.
        if structure_path is not None:
            if not isinstance(structure_path, str):
                raise TypeError("structure_path must be a string path when provided.")
            structure_path_abs = _absolute_path_or_none(structure_path)
            if structure_path_abs is None or not Path(structure_path_abs).is_file():
                raise FileNotFoundError(
                    f"structure_path does not exist: {structure_path}"
                )

        # Check name type if provided.
        name = input_dict.get("name")
        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string when provided.")

    @staticmethod
    def _validate_sampling_parameters(input_dict: dict[str, Any]) -> None:
        """
        Validate seed / batch_size / number_of_batches and repeat_sample_num.
        """
        # Check seed, batch_size, number_of_batches types and values.
        for key in ("seed", "batch_size", "number_of_batches"):
            val = input_dict.get(key)
            if val is None:
                continue
            if not isinstance(val, int):
                raise TypeError(f"{key} must be an int when provided.")
            if key in ("batch_size", "number_of_batches") and val <= 0:
                raise ValueError(f"{key} must be positive if provided.")

        # repeat_sample_num is derived internally from batch_size and must not
        # appear in user JSON.
        if "repeat_sample_num" in input_dict:
            raise ValueError(
                "repeat_sample_num is not allowed in the JSON config; "
                "use batch_size instead."
            )

    @staticmethod
    def _validate_parser_overrides(input_dict: dict[str, Any]) -> None:
        """Validate parser override fields: remove_ccds, remove_waters."""

        # Check that remove_ccds is a list of strings if provided.
        remove_ccds = input_dict.get("remove_ccds")
        if remove_ccds is not None:
            if not isinstance(remove_ccds, list):
                raise TypeError("remove_ccds must be a list of CCD residue names.")
            for item in remove_ccds:
                if not isinstance(item, str):
                    raise TypeError(
                        f"remove_ccds entries must be strings, got {type(item)}"
                    )

        # Check that remove_waters is a boolean if provided.
        remove_waters = input_dict.get("remove_waters")
        if remove_waters is not None and not isinstance(remove_waters, bool):
            raise TypeError("remove_waters must be a bool when provided.")

    @staticmethod
    def _validate_pipeline_override_fields(input_dict: dict[str, Any]) -> None:
        """
        Validate pipeline set-up override fields:
        occupancy thresholds and undesired_res_names.
        """
        # Check occupancy threshold types.
        for key in ("occupancy_threshold_sidechain", "occupancy_threshold_backbone"):
            val = input_dict.get(key)
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                raise TypeError(f"{key} must be numeric when provided.")

        # Check undesired_res_names is a list of strings if provided.
        undesired_res_names = input_dict.get("undesired_res_names")
        if undesired_res_names is not None:
            if not isinstance(undesired_res_names, list):
                raise TypeError("undesired_res_names must be a list when provided.")
            for item in undesired_res_names:
                if not isinstance(item, str):
                    raise TypeError(
                        f"undesired_res_names entries must be strings, got {type(item)}"
                    )

    @staticmethod
    def _validate_scalar_user_settings(input_dict: dict[str, Any]) -> None:
        """
        Validate scalar user settings and related fields:
        structure_noise, decode_type, causality_pattern,
        initialize_sequence_embedding_with_ground_truth,
        features_to_return, atomize_side_chains.
        """
        # Check type of structure_noise.
        if input_dict.get("structure_noise") is not None and not isinstance(
            input_dict["structure_noise"], (int, float)
        ):
            raise TypeError("structure_noise must be numeric when provided.")

        # Check type and value of decode_type.
        decode_type = input_dict.get("decode_type")
        if decode_type is not None:
            if not isinstance(decode_type, str):
                raise TypeError("decode_type must be a string when provided.")
            allowed = {"auto_regressive", "teacher_forcing"}
            if decode_type not in allowed:
                raise ValueError(
                    f"decode_type must be one of {sorted(allowed)}, got '{decode_type}'"
                )

        # Check type and value of causality_pattern.
        causality_pattern = input_dict.get("causality_pattern")
        if causality_pattern is not None:
            if not isinstance(causality_pattern, str):
                raise TypeError("causality_pattern must be a string when provided.")
            allowed = {
                "auto_regressive",
                "unconditional",
                "conditional",
                "conditional_minus_self",
            }
            if causality_pattern not in allowed:
                raise ValueError(
                    f"causality_pattern must be one of {sorted(allowed)}, "
                    f"got '{causality_pattern}'"
                )

        # Check type of initialize_sequence_embedding_with_ground_truth.
        initialize_sequence_embedding_with_ground_truth = input_dict.get(
            "initialize_sequence_embedding_with_ground_truth"
        )
        if (
            initialize_sequence_embedding_with_ground_truth is not None
            and not isinstance(initialize_sequence_embedding_with_ground_truth, bool)
        ):
            raise TypeError(
                "initialize_sequence_embedding_with_ground_truth must be a "
                "bool when provided."
            )

        features_to_return = input_dict.get("features_to_return")
        if features_to_return is not None and not isinstance(features_to_return, dict):
            raise TypeError("features_to_return must be a dict when provided.")

        # Check type of atomize_side_chains.
        atomize_side_chains = input_dict.get("atomize_side_chains")
        if atomize_side_chains is not None and not isinstance(
            atomize_side_chains, bool
        ):
            raise TypeError("atomize_side_chains must be a bool when provided.")

    @staticmethod
    def _validate_design_scope(input_dict: dict[str, Any]) -> None:
        """
        Validate fixed/designed residue and chain fields.

        - Lists must actually be lists.
        - Residue IDs must parse as <chain><integer> (e.g. 'A35').
        - Chain IDs must be strings.
        - Mutually exclusive combinations are disallowed.
        """
        fixed_res = input_dict.get("fixed_residues")
        designed_res = input_dict.get("designed_residues")
        fixed_chains = input_dict.get("fixed_chains")
        designed_chains = input_dict.get("designed_chains")

        # Check types + residue-id parsing
        for key in ("fixed_residues", "designed_residues"):
            val = input_dict.get(key)
            if val is None:
                continue
            if not isinstance(val, list):
                raise TypeError(f"{key} must be a list if provided.")
            for res_id in val:
                if not isinstance(res_id, str):
                    raise TypeError(
                        f"{key} entries must be residue-id strings, got {type(res_id)}"
                    )
                MPNNInferenceInput._parse_id(res_id, res_num_required=True)

        # Check chain ID types
        for key in ("fixed_chains", "designed_chains"):
            val = input_dict.get(key)
            if val is None:
                continue
            if not isinstance(val, list):
                raise TypeError(f"{key} must be a list if provided.")
            for chain_id in val:
                if not isinstance(chain_id, str):
                    raise TypeError(
                        f"{key} entries must be chain-id strings, got {type(chain_id)}"
                    )
                MPNNInferenceInput._parse_id(chain_id, res_num_allowed=False)

        # Mutual exclusivity rules
        if fixed_res is not None and designed_res is not None:
            raise ValueError("Cannot set both fixed_residues and designed_residues.")
        if fixed_chains is not None and designed_chains is not None:
            raise ValueError("Cannot set both fixed_chains and designed_chains.")
        if (fixed_res or designed_res) and (fixed_chains or designed_chains):
            raise ValueError(
                "Cannot mix residue-based and chain-based design constraints "
                "in the same input."
            )

    @staticmethod
    def _validate_bias_omit_and_pair_bias(input_dict: dict[str, Any]) -> None:
        """
        Validate global/per-residue bias & omit and pair-bias containers.

        This centralizes checks for:
        - bias / bias_per_residue
        - omit / omit_per_residue
        - pair_bias
        - pair_bias_per_residue_pair
        """
        token_to_idx = MPNN_TOKEN_ENCODING.token_to_idx

        # Check bias type, token membership, and value types.
        bias = input_dict.get("bias")
        if bias is not None and not isinstance(bias, dict):
            raise TypeError("bias must be a dict {token_name: bias_value}.")
        if isinstance(bias, dict):
            for token_name, value in bias.items():
                if token_name not in token_to_idx:
                    raise ValueError(
                        f"bias key '{token_name}' is not in the MPNN token vocabulary."
                    )
                if not isinstance(value, (int, float)):
                    raise TypeError(
                        f"bias['{token_name}'] must be numeric, got {type(value)}"
                    )

        # Check bias_per_residue type, residue-id parsing, token membership,
        # and value types.
        bias_per_residue = input_dict.get("bias_per_residue")
        if bias_per_residue is not None and not isinstance(bias_per_residue, dict):
            raise TypeError("bias_per_residue must be a dict.")
        if isinstance(bias_per_residue, dict):
            for res_id, res_id_bias in bias_per_residue.items():
                # Check residue ID type and parsing
                if not isinstance(res_id, str):
                    raise TypeError(
                        "bias_per_residue keys must be residue-id strings, "
                        f"got {type(res_id)}"
                    )
                MPNNInferenceInput._parse_id(res_id, res_num_required=True)

                # Check bias for this res_id.
                if not isinstance(res_id_bias, dict):
                    raise TypeError(
                        f"bias_per_residue[{res_id}] must be a dict, "
                        f"got {type(res_id_bias)}"
                    )
                for token_name, value in res_id_bias.items():
                    if token_name not in token_to_idx:
                        raise ValueError(
                            f"bias_per_residue[{res_id}] key '{token_name}' is "
                            "not in the MPNN token vocabulary."
                        )
                    if not isinstance(value, (int, float)):
                        raise TypeError(
                            "bias_per_residue"
                            f"[{res_id}]['{token_name}'] must be numeric, "
                            f"got {type(value)}"
                        )

        # Check omit type and token membership.
        omit = input_dict.get("omit")
        if omit is not None and not isinstance(omit, list):
            raise TypeError("omit must be a list of residue codes.")
        if isinstance(omit, list):
            for token_name in omit:
                if token_name not in token_to_idx:
                    raise ValueError(
                        f"omit entry '{token_name}' is not in the MPNN token "
                        "vocabulary."
                    )

        # Check omit_per_residue type, residue-id parsing, and token membership.
        omit_per_residue = input_dict.get("omit_per_residue")
        if omit_per_residue is not None and not isinstance(omit_per_residue, dict):
            raise TypeError("omit_per_residue must be a dict.")
        if isinstance(omit_per_residue, dict):
            for res_id, res_id_omit in omit_per_residue.items():
                # Check residue ID type and parsing.
                if not isinstance(res_id, str):
                    raise TypeError(
                        "omit_per_residue keys must be residue-id strings, "
                        f"got {type(res_id)}"
                    )
                MPNNInferenceInput._parse_id(res_id, res_num_required=True)

                # Check omit list for this res_id.
                if not isinstance(res_id_omit, list):
                    raise TypeError(
                        f"omit_per_residue[{res_id}] must be a list, "
                        f"got {type(res_id_omit)}"
                    )
                for token_name in res_id_omit:
                    if token_name not in token_to_idx:
                        raise ValueError(
                            f"omit_per_residue[{res_id}] entry '{token_name}' "
                            "is not in the MPNN token vocabulary."
                        )

        # Check pair_bias type, token membership, and value types.
        pair_bias = input_dict.get("pair_bias")
        if pair_bias is not None and not isinstance(pair_bias, dict):
            raise TypeError("pair_bias must be a nested dict when provided.")
        if isinstance(pair_bias, dict):
            for token_i, token_j_to_bias in pair_bias.items():
                # Check outer token membership.
                if token_i not in token_to_idx:
                    raise ValueError(
                        f"pair_bias key '{token_i}' is not in the MPNN token "
                        "vocabulary."
                    )

                # Check token_j_to_bias type.
                if not isinstance(token_j_to_bias, dict):
                    raise TypeError(
                        f"pair_bias['{token_i}'] must be a dict mapping "
                        "token_name_j -> bias."
                    )

                # Check inner token membership and value types.
                for token_j, value in token_j_to_bias.items():
                    if token_j not in token_to_idx:
                        raise ValueError(
                            f"pair_bias['{token_i}'] key '{token_j}' is not in "
                            "the MPNN token vocabulary."
                        )
                    if not isinstance(value, (int, float)):
                        raise TypeError(
                            f"pair_bias['{token_i}']['{token_j}'] must be "
                            f"numeric, got {type(value)}"
                        )

        # ---------------- pair_bias_per_residue_pair ---------------- #
        pair_bias_per = input_dict.get("pair_bias_per_residue_pair")
        if pair_bias_per is not None and not isinstance(pair_bias_per, dict):
            raise TypeError("pair_bias_per_residue_pair must be a dict when provided.")
        if isinstance(pair_bias_per, dict):
            for res_id_i, res_id_j_to_pair_bias in pair_bias_per.items():
                # Check residue ID type and parsing.
                if not isinstance(res_id_i, str):
                    raise TypeError(
                        "pair_bias_per_residue_pair keys must be residue-id "
                        f"strings, got {type(res_id_i)}"
                    )
                MPNNInferenceInput._parse_id(res_id_i, res_num_required=True)

                # Check res_id_j_to_pair_bias type.
                if not isinstance(res_id_j_to_pair_bias, dict):
                    raise TypeError(
                        f"pair_bias_per_residue_pair['{res_id_i}'] must be a "
                        "dict mapping res_id_j -> dict."
                    )

                for res_id_j, i_j_pair_bias in res_id_j_to_pair_bias.items():
                    # Check residue ID type and parsing.
                    if not isinstance(res_id_j, str):
                        raise TypeError(
                            "pair_bias_per_residue_pair inner keys must be "
                            f"residue-id strings, got {type(res_id_j)}"
                        )
                    MPNNInferenceInput._parse_id(res_id_j, res_num_required=True)

                    # Check the res i, res j pair bias dict.
                    if not isinstance(i_j_pair_bias, dict):
                        raise TypeError(
                            f"pair_bias_per_residue_pair['{res_id_i}']"
                            f"['{res_id_j}'] must be "
                            "a dict mapping token_name_i -> dict."
                        )
                    for token_i, token_j_to_bias in i_j_pair_bias.items():
                        # Check outer token membership.
                        if token_i not in token_to_idx:
                            raise ValueError(
                                "pair_bias_per_residue_pair"
                                f"['{res_id_i}']['{res_id_j}'] key '{token_i}' "
                                "is not in the MPNN token vocabulary."
                            )

                        # Check token_j_to_bias type.
                        if not isinstance(token_j_to_bias, dict):
                            raise TypeError(
                                "pair_bias_per_residue_pair"
                                f"['{res_id_i}']['{res_id_j}']['{token_i}'] "
                                "must be a dict mapping token_name_j -> bias."
                            )

                        # Check inner token membership and value types.
                        for token_j, value in token_j_to_bias.items():
                            if token_j not in token_to_idx:
                                raise ValueError(
                                    "pair_bias_per_residue_pair"
                                    f"['{res_id_i}']['{res_id_j}']"
                                    f"['{token_i}'] key '{token_j}' is not in "
                                    "the MPNN token vocabulary."
                                )
                            if not isinstance(value, (int, float)):
                                raise TypeError(
                                    "pair_bias_per_residue_pair"
                                    f"['{res_id_i}']['{res_id_j}']"
                                    f"['{token_i}']['{token_j}'] "
                                    f"must be numeric, got {type(value)}"
                                )

    @staticmethod
    def _validate_temperature(input_dict: dict[str, Any]) -> None:
        """Validate temperature scalars and per-residue mappings."""
        # Check temperature type if provided.
        temperature = input_dict.get("temperature")
        if temperature is not None and not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be numeric when provided.")

        # Check temperature_per_residue type, residue-id parsing, and value
        # types.
        temperature_per_residue = input_dict.get("temperature_per_residue")
        if temperature_per_residue is not None and not isinstance(
            temperature_per_residue, dict
        ):
            raise TypeError("temperature_per_residue must be a dict.")
        if isinstance(temperature_per_residue, dict):
            # Check each residue ID and temperature value.
            for res_id, res_id_temperature in temperature_per_residue.items():
                # Check residue ID type and parsing.
                if not isinstance(res_id, str):
                    raise TypeError(
                        "temperature_per_residue keys must be residue-id "
                        f"strings, got {type(res_id)}"
                    )
                MPNNInferenceInput._parse_id(res_id, res_num_required=True)

                # Check temperature value type.
                if not isinstance(res_id_temperature, (int, float)):
                    raise TypeError(
                        f"temperature_per_residue[{res_id}] must be numeric; "
                        f"got {type(res_id_temperature)}"
                    )

    @staticmethod
    def _validate_symmetry(input_dict: dict[str, Any]) -> None:
        """
        Validate symmetry-related fields, including residue-id parsing
        and mutual exclusivity between residue-based symmetry and
        homo-oligomer chain symmetry.
        """
        symmetry_residues = input_dict.get("symmetry_residues")
        symmetry_residues_weights = input_dict.get("symmetry_residues_weights")
        homo_oligomer_chains = input_dict.get("homo_oligomer_chains")

        # Check symmetry_residues type and residue-id parsing.
        if symmetry_residues is not None:
            if not isinstance(symmetry_residues, list):
                raise TypeError(
                    "symmetry_residues must be a list of lists when provided."
                )

            seen_res_ids = set()
            for symmetry_residue_group in symmetry_residues:
                if not isinstance(symmetry_residue_group, list):
                    raise TypeError("Each element of symmetry_residues must be a list.")
                for res_id in symmetry_residue_group:
                    # Check the residue id.
                    if not isinstance(res_id, str):
                        raise TypeError(
                            "symmetry_residues entries must be residue-id "
                            f"strings, got {type(res_id)}"
                        )
                    MPNNInferenceInput._parse_id(res_id, res_num_required=True)

                    # Check for duplicates across all groups.
                    if res_id in seen_res_ids:
                        raise ValueError(
                            f"symmetry_residues contains duplicate residue "
                            f"ID '{res_id}' across groups."
                        )
                    seen_res_ids.add(res_id)

        # Check symmetry_residues_weights type, shape, and value types.
        if symmetry_residues_weights is not None:
            if not isinstance(symmetry_residues_weights, list):
                raise TypeError(
                    "symmetry_residues_weights must be a list of lists when provided."
                )

            # Check that symmetry_residues is also provided, and that the
            # outer lengths match.
            if symmetry_residues is None:
                raise ValueError(
                    "symmetry_residues_weights provided without symmetry_residues."
                )
            if len(symmetry_residues_weights) != len(symmetry_residues):
                raise ValueError(
                    "symmetry_residues_weights must have the same outer length "
                    "as symmetry_residues."
                )

            # Check that each symmetry_residues_weights group is a list and
            # that the inner lengths match symmetry_residues, also check
            # weight values.
            for symmetry_residue_group, symmetry_residue_group_weights in zip(
                symmetry_residues, symmetry_residues_weights
            ):
                # Check group type.
                if not isinstance(symmetry_residue_group_weights, list):
                    raise TypeError(
                        "Each element of symmetry_residues_weights must be a list."
                    )

                # Length check.
                if len(symmetry_residue_group) != len(symmetry_residue_group_weights):
                    raise ValueError(
                        f"symmetry_residues group {symmetry_residue_group} "
                        "has different length than corresponding weights "
                        f"group {symmetry_residue_group_weights}."
                    )

                # Weight type check.
                for weight in symmetry_residue_group_weights:
                    if not isinstance(weight, (int, float)):
                        raise TypeError(
                            "symmetry_residues_weights entries must be "
                            f"numeric; got {type(weight)}"
                        )

        # Check homo_oligomer_chains type and chain-id parsing.
        if homo_oligomer_chains is not None:
            if not isinstance(homo_oligomer_chains, list):
                raise TypeError(
                    "homo_oligomer_chains must be a list of lists when provided."
                )

            # Check each chain group.
            for chain_group in homo_oligomer_chains:
                # Check group type.
                if not isinstance(chain_group, list):
                    raise TypeError(
                        "Each element of homo_oligomer_chains must be a list."
                    )

                # Check that each group has at least 2 chains.
                if len(chain_group) < 2:
                    raise ValueError(
                        "Each homo_oligomer_chains group must contain at "
                        "least 2 chains."
                    )

                # Check each chain ID.
                for chain_id in chain_group:
                    if not isinstance(chain_id, str):
                        raise TypeError(
                            "homo_oligomer_chains entries must be chain-id "
                            f"strings, got {type(chain_id)}"
                        )
                    MPNNInferenceInput._parse_id(chain_id, res_num_allowed=False)

        # Check mutual exclusivity of symmetry_residues and
        # homo_oligomer_chains.
        if symmetry_residues is not None and homo_oligomer_chains is not None:
            raise ValueError(
                "Residue-based symmetry (symmetry_residues / "
                "symmetry_residues_weights) and homo-oligomer symmetry "
                "(homo_oligomer_chains) are mutually exclusive; "
                "please specify only one."
            )

    @staticmethod
    def _validate_all(
        input_dict: dict[str, Any],
        require_structure_path: bool,
    ) -> None:
        """
        Run all JSON-level validation routines on a single input dict.

        Args:
            input_dict (dict[str, Any]): JSON config for single input.
            require_structure_path (bool): If True, a valid on-disk
                structure_path must be present.
        """
        MPNNInferenceInput._validate_structure_path_and_name(
            input_dict=input_dict,
            require_structure_path=require_structure_path,
        )
        MPNNInferenceInput._validate_sampling_parameters(input_dict)
        MPNNInferenceInput._validate_parser_overrides(input_dict)
        MPNNInferenceInput._validate_pipeline_override_fields(input_dict)
        MPNNInferenceInput._validate_scalar_user_settings(input_dict)
        MPNNInferenceInput._validate_design_scope(input_dict)
        MPNNInferenceInput._validate_bias_omit_and_pair_bias(input_dict)
        MPNNInferenceInput._validate_temperature(input_dict)
        MPNNInferenceInput._validate_symmetry(input_dict)

    @staticmethod
    def apply_defaults(input_dict: dict[str, Any]) -> None:
        """Apply JSON-level defaults. Modifies in place."""
        for key, default_value in MPNN_PER_INPUT_INFERENCE_DEFAULTS.items():
            if key not in input_dict:
                input_dict[key] = default_value

    @staticmethod
    def post_process_inputs(input_dict: dict[str, Any]) -> None:
        """Apply post-processing to input dict. Modifies in place."""
        # Ensure structure_path is absolute.
        input_dict["structure_path"] = _absolute_path_or_none(
            input_dict["structure_path"]
        )

        # Set name if missing.
        if input_dict["name"] is None:
            if input_dict["structure_path"] is not None:
                input_dict["name"] = Path(input_dict["structure_path"]).stem
            else:
                input_dict["name"] = "unnamed"

        # Set repeat_sample_num from batch_size.
        input_dict["repeat_sample_num"] = input_dict["batch_size"]

    @staticmethod
    def build_atom_array(input_dict: dict[str, Any]) -> AtomArray:
        """Build AtomArray from structure_path and parser overrides."""
        # Override parser args if specified.
        parser_args = dict(STANDARD_PARSER_ARGS)
        if input_dict["remove_ccds"] is not None:
            parser_args["remove_ccds"] = input_dict["remove_ccds"]
        if input_dict["remove_waters"] is not None:
            parser_args["remove_waters"] = input_dict["remove_waters"]

        # Parse structure file.
        data = parse(
            filename=input_dict["structure_path"],
            keep_cif_block=True,
            **parser_args,
        )

        # Use assembly 1 if present, otherwise use asymmetric unit.
        if "assemblies" in data:
            atom_array = data["assemblies"]["1"][0]
        else:
            atom_array = data["asym_unit"][0]

        return atom_array

    @staticmethod
    def _annotate_design_scope(
        atom_array: AtomArray,
        input_dict: dict[str, Any],
    ) -> None:
        """
        Attach 'mpnn_designed_residue_mask' from design-scope fields.

        This function assumes that no existing 'mpnn_designed_residue_mask'
        annotation is present; callers should skip invocation if the annotation
        already exists.

        Semantics
        ---------
        - If all design-scope fields are None/empty, this function is a no-op
          and leaves the implicit "design all residues" behavior.
        - If designed_* fields are present, they define the design mask
          (starting from all False).
        - Otherwise, we start from all True and then clear fixed_* fields.
        """
        fixed_residues = input_dict["fixed_residues"] or []
        designed_residues = input_dict["designed_residues"] or []
        fixed_chains = input_dict["fixed_chains"] or []
        designed_chains = input_dict["designed_chains"] or []

        # If absolutely nothing is specified -> design all, and rely on the
        # implicit "design all residues" behavior in the model code.
        if not (fixed_residues or designed_residues or fixed_chains or designed_chains):
            return

        # Gather the token-level array.
        token_starts = get_token_starts(atom_array)
        token_level = atom_array[token_starts]
        n_tokens = token_level.array_length()

        # Initialize mask depending on which fields are present.
        if designed_residues or designed_chains:
            designed_residue_mask_token_level = np.zeros(n_tokens, dtype=bool)
        elif fixed_residues or fixed_chains:
            designed_residue_mask_token_level = np.ones(n_tokens, dtype=bool)
        else:
            raise RuntimeError("Unreachable state in _annotate_design_scope.")

        # Residue-based constraints.
        if fixed_residues:
            mask = MPNNInferenceInput._mask_from_ids(token_level, fixed_residues)
            designed_residue_mask_token_level[mask] = False

        if designed_residues:
            mask = MPNNInferenceInput._mask_from_ids(token_level, designed_residues)
            designed_residue_mask_token_level[mask] = True

        # Chain-based constraints.
        if fixed_chains:
            mask = MPNNInferenceInput._mask_from_ids(token_level, fixed_chains)
            designed_residue_mask_token_level[mask] = False

        if designed_chains:
            mask = MPNNInferenceInput._mask_from_ids(token_level, designed_chains)
            designed_residue_mask_token_level[mask] = True

        # Spread to atom level.
        mpnn_designed_residue_mask = spread_token_wise(
            atom_array, designed_residue_mask_token_level
        )

        # Annotate.
        atom_array.set_annotation(
            "mpnn_designed_residue_mask",
            mpnn_designed_residue_mask.astype(bool),
        )

    @staticmethod
    def _annotate_temperature(
        atom_array: AtomArray,
        input_dict: dict[str, Any],
    ) -> None:
        """
        Attach 'mpnn_temperature' annotation from scalar + per-residue
        temperature settings.

        This function assumes that no existing 'mpnn_temperature' annotation
        is present; callers should skip invocation if the annotation already
        exists.

        Semantics
        ---------
        - Per-residue values override the global scalar for the specified
          residues (not additive).
        - If neither a global temperature nor any per-residue temperatures
          are specified, this function is a no-op.
        """
        temperature = input_dict["temperature"]
        temperature_per_residue = input_dict["temperature_per_residue"] or {}

        # If there is no global or per-residue temperature, this is a no-op.
        if temperature is None and not temperature_per_residue:
            return
        elif temperature is None and temperature_per_residue:
            raise RuntimeError(
                "temperature_per_residue provided without global temperature."
            )

        # Gather token-level array.
        token_starts = get_token_starts(atom_array)
        token_level = atom_array[token_starts]
        n_tokens = token_level.array_length()

        # Create the global temperature token array.
        temperature_token_level = np.full(n_tokens, temperature, dtype=np.float32)

        # Per-residue overrides for temperature.
        for res_id_str, res_id_temperature in temperature_per_residue.items():
            token_mask = MPNNInferenceInput._mask_from_ids(token_level, [res_id_str])
            temperature_token_level[token_mask] = res_id_temperature

        # Spread to atom level.
        mpnn_temperature = spread_token_wise(
            atom_array, temperature_token_level.astype(np.float32)
        )

        # Annotate.
        atom_array.set_annotation(
            "mpnn_temperature",
            mpnn_temperature.astype(np.float32),
        )

    @staticmethod
    def _build_bias_vector_from_dict(
        bias_dict: dict[str, float] | None,
    ) -> np.ndarray:
        """Convert {token_name: bias} dict to vocab-length vector."""
        # Create a zero bias vector.
        vocab_size = MPNN_TOKEN_ENCODING.n_tokens
        bias_vector = np.zeros((vocab_size,), dtype=np.float32)

        # If no bias dict, return zero vector.
        if not bias_dict:
            return bias_vector

        # Populate the bias vector.
        token_to_idx = MPNN_TOKEN_ENCODING.token_to_idx
        for token_name, token_bias in bias_dict.items():
            bias_vector[token_to_idx[token_name]] = token_bias
        return bias_vector

    @staticmethod
    def _annotate_bias_and_omit(
        atom_array: AtomArray,
        input_dict: dict[str, Any],
        omit_bias_value: float = -1e8,
    ) -> None:
        """
        Attach 'mpnn_bias' annotation from:
        - bias
        - bias_per_residue
        - omit
        - omit_per_residue

        This function assumes that no existing 'mpnn_bias' annotation is
        present; callers should skip invocation if the annotation already
        exists.

        Behavior
        --------
        - bias_per_residue overrides global bias for overlapping tokens.
        - omit forms its own bias matrix (global + per-residue), then is
          added to the bias matrix.
        - If the resulting bias matrix is all zeros, this function is a no-op
          and does not create an annotation.
        """
        bias = input_dict["bias"] or {}
        bias_per_residue = input_dict["bias_per_residue"] or {}
        omit = input_dict["omit"] or []
        omit_per_residue = input_dict["omit_per_residue"] or {}

        # Gather token-level array.
        token_starts = get_token_starts(atom_array)
        token_level = atom_array[token_starts]
        n_tokens = token_level.array_length()

        vocab_size = MPNN_TOKEN_ENCODING.n_tokens

        # ---------------- Bias ---------------- #
        # Initialize a zero bias matrix.
        bias_token_level = np.zeros((n_tokens, vocab_size), dtype=np.float32)

        # Compute the vector for the global bias setting.
        global_bias_vector = MPNNInferenceInput._build_bias_vector_from_dict(bias)

        # If there is a global bias, set all residues to it.
        if np.any(global_bias_vector != 0.0):
            bias_token_level[:] = global_bias_vector

        # Per-residue bias overrides.
        for res_id_str, res_id_bias in bias_per_residue.items():
            # Construct the per-residue bias vector and mask.
            per_residue_bias_vector = MPNNInferenceInput._build_bias_vector_from_dict(
                res_id_bias
            )
            token_mask = MPNNInferenceInput._mask_from_ids(token_level, [res_id_str])

            # Apply the per-residue bias vector.
            bias_token_level[token_mask] = per_residue_bias_vector

        # ---------------- Omit ---------------- #
        # Initialize a zero omit bias matrix.
        omit_bias_token_level = np.zeros((n_tokens, vocab_size), dtype=np.float32)

        # Compute the vector for the global omit setting.
        global_omit_bias_vector = MPNNInferenceInput._build_bias_vector_from_dict(
            {token_name: omit_bias_value for token_name in omit}
        )

        # If there is a global omit, set all residues to it.
        if np.any(global_omit_bias_vector != 0.0):
            omit_bias_token_level[:] = global_omit_bias_vector

        # Per-residue omit overrides.
        for res_id_str, res_id_omit in omit_per_residue.items():
            # Construct the per-residue omit bias vector and mask.
            per_residue_omit_bias_vector = (
                MPNNInferenceInput._build_bias_vector_from_dict(
                    {token_name: omit_bias_value for token_name in res_id_omit}
                )
            )
            token_mask = MPNNInferenceInput._mask_from_ids(token_level, [res_id_str])

            # Apply the per-residue omit vector.
            omit_bias_token_level[token_mask] = per_residue_omit_bias_vector

        # ---------------- Combine bias and omit ---------------- #
        # Add omit into the bias matrix.
        bias_token_level = bias_token_level + omit_bias_token_level

        # No-op if there is no non-zero bias information.
        if not np.any(bias_token_level != 0.0):
            return

        # Spread to atom level.
        mpnn_bias = spread_token_wise(atom_array, bias_token_level)

        # Annotate.
        atom_array.set_annotation("mpnn_bias", mpnn_bias.astype(np.float32))

    @staticmethod
    def _annotate_symmetry(
        atom_array: AtomArray,
        input_dict: dict[str, Any],
    ) -> None:
        """
        Attach symmetry-related annotations:

        - 'mpnn_symmetry_equivalence_group' (int group IDs)
        - 'mpnn_symmetry_weight' (optional weights)

        This function assumes that no existing symmetry annotations are
        present; callers should skip invocation if either
        'mpnn_symmetry_equivalence_group' or 'mpnn_symmetry_weight' already
        exist on the atom array.

        Semantics
        ---------
        - Supports either residue-based symmetry or homo-oligomer chain
          symmetry (not both, as enforced by validation).
        - If no symmetry information is provided, this function is a no-op.
        - If no weights are provided, 'mpnn_symmetry_weight' is not created.
        - Any residues or chains that are not explicitly included in a symmetry
          group are treated as individual symmetry groups, each with its own
          unique group ID. If weights are present, these singleton groups have
          weight 1.0.
        """
        symmetry_residues = input_dict["symmetry_residues"]
        symmetry_residues_weights = input_dict["symmetry_residues_weights"]
        homo_oligomer_chains = input_dict["homo_oligomer_chains"]

        # If no symmetry information, this is a no-op.
        if symmetry_residues is None and homo_oligomer_chains is None:
            return

        # Gather token-level array.
        token_starts = get_token_starts(atom_array)
        token_level = atom_array[token_starts]
        n_tokens = token_level.array_length()

        # By default, every token is its own symmetry group. We will overwrite
        # these IDs for any tokens that participate in an explicit symmetry
        # group. The absolute group ID values do not matter, only equality.
        symmetry_equivalent_group_token_level = np.arange(n_tokens, dtype=np.int32)

        # Optional weights: only created if weights are provided. If present,
        # default weight is 1.0 for all tokens; explicit symmetry groups
        # overwrite the weights for the tokens they cover.
        symmetry_weight_token_level = None

        # Residue-based symmetry
        if symmetry_residues is not None:
            # If weights are provided, initialize the weights array.
            if symmetry_residues_weights is not None:
                symmetry_weight_token_level = np.ones(n_tokens, dtype=np.float32)

            # Start assigning new group IDs above the current maximum. Each
            # explicit symmetry group gets a fresh ID, so all residues in the
            # group share that ID.
            next_group_id = int(symmetry_equivalent_group_token_level.max()) + 1

            for group_index, symmetry_residue_group in enumerate(symmetry_residues):
                # Assign a unique group ID for this explicit residue group.
                group_id = next_group_id
                next_group_id += 1

                # Get the corresponding weights for this group if present.
                if symmetry_residues_weights is not None:
                    group_weights = symmetry_residues_weights[group_index]
                else:
                    group_weights = None

                # Assign group ID and weights to each residue in the group.
                for position, res_id_str in enumerate(symmetry_residue_group):
                    # Get the token mask for this residue ID.
                    token_mask = MPNNInferenceInput._mask_from_ids(
                        token_level, [res_id_str]
                    )

                    # Write the group ID.
                    symmetry_equivalent_group_token_level[token_mask] = group_id

                    # Write the weight if applicable.
                    if (
                        symmetry_weight_token_level is not None
                        and group_weights is not None
                    ):
                        weight_value = group_weights[position]
                        symmetry_weight_token_level[token_mask] = weight_value

        # Homo-oligomer chain symmetry. We rely on the implicit behavior of
        # mpnn_symmetry_weights = None -> weight of 1.0 for all tokens.
        elif homo_oligomer_chains is not None:
            # Start assigning new group IDs above the current maximum. Each
            # explicit symmetry group gets a fresh ID, so all residues in the
            # group share that ID.
            next_group_id = int(symmetry_equivalent_group_token_level.max()) + 1

            for chain_group in homo_oligomer_chains:
                # For each chain in the group, collect token indices.
                per_chain_indices = []
                for chain_id_str in chain_group:
                    chain_mask = MPNNInferenceInput._mask_from_ids(
                        token_level, [chain_id_str]
                    )
                    per_chain_indices.append(np.nonzero(chain_mask)[0])

                # All chains in the group must have the same number of tokens.
                lengths = {indices.size for indices in per_chain_indices}
                if len(lengths) != 1:
                    raise ValueError(
                        "All chains in a homo_oligomer_chains group must have "
                        "the same number of residues/tokens. "
                        f"Group {chain_group!r} has token counts {lengths}."
                    )

                n_positions = next(iter(lengths))

                # Interleave by position: tokens at the same position along
                # each chain are symmetry-equivalent.
                for position_index in range(n_positions):
                    group_id = next_group_id
                    next_group_id += 1

                    # Grab the token indices for the current group.
                    token_indices = [
                        int(indices[position_index]) for indices in per_chain_indices
                    ]

                    # Assign the group ID to the tokens.
                    symmetry_equivalent_group_token_level[token_indices] = group_id

        # Spread to atom level for equivalence groups.
        mpnn_symmetry_equivalence_group = spread_token_wise(
            atom_array, symmetry_equivalent_group_token_level
        )

        # Annotate equivalence groups.
        atom_array.set_annotation(
            "mpnn_symmetry_equivalence_group",
            mpnn_symmetry_equivalence_group.astype(np.int32),
        )

        # If symmetry weights are present:
        if symmetry_weight_token_level is not None:
            # Spread to atom level for weights.
            mpnn_symmetry_weight = spread_token_wise(
                atom_array, symmetry_weight_token_level
            )

            # Annotate weights.
            atom_array.set_annotation(
                "mpnn_symmetry_weight",
                mpnn_symmetry_weight.astype(np.float32),
            )

    @staticmethod
    def _build_pair_bias_matrix_from_dict(
        pair_bias_dict: dict[str, dict[str, float]] | None,
    ) -> np.ndarray:
        """Convert {token_i: {token_j: bias}} into a [vocab, vocab] matrix."""
        # Create a zero pair-bias matrix.
        vocab_size = MPNN_TOKEN_ENCODING.n_tokens
        pair_bias_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

        # If no pair-bias dict, return zero matrix.
        if not pair_bias_dict:
            return pair_bias_matrix

        # Populate the pair-bias matrix.
        token_to_idx = MPNN_TOKEN_ENCODING.token_to_idx
        for token_i, token_j_to_bias in pair_bias_dict.items():
            for token_j, value in token_j_to_bias.items():
                pair_bias_matrix[token_to_idx[token_i], token_to_idx[token_j]] = value

        return pair_bias_matrix

    @staticmethod
    def _annotate_pair_bias(
        atom_array: AtomArrayPlus,
        input_dict: dict[str, Any],
    ) -> None:
        """
        Attach 2D 'mpnn_pair_bias' annotation, with pairs stored as:
        - pairs_arr: [num_pairs, 2] int32 indices (atom indices)
        - values_arr: [num_pairs, vocab, vocab] float32 bias matrices

        This function assumes that no existing 'mpnn_pair_bias' 2D annotation
        is present; callers should skip invocation if the annotation already
        exists.

        Semantics
        ---------
        - Global pair_bias: applies to all residue pairs (CA representatives).
        - pair_bias_per_residue_pair:
          * For residue pairs present here, the matrix overrides the global
          pair_bias matrix.
        - If there is no pair-bias information at all, or if all matrices are
          zero, this function is a no-op.
        """
        pair_bias = input_dict["pair_bias"]
        pair_bias_per_residue_pair = input_dict["pair_bias_per_residue_pair"]

        # If there is no pair-bias information, this is a no-op.
        if not pair_bias and not pair_bias_per_residue_pair:
            return

        # Identify CA atoms as residue-level representatives.
        ca_mask = atom_array.atom_name == "CA"
        ca_indices = np.nonzero(ca_mask)[0]
        ca_array = atom_array[ca_mask]
        n_tokens_ca = ca_array.array_length()

        # Check for presence of CA atoms.
        if n_tokens_ca == 0:
            raise ValueError(
                "No CA atoms found in the structure; cannot build pair bias."
            )

        # Compute the global pair-bias matrix.
        global_pair_bias_matrix = MPNNInferenceInput._build_pair_bias_matrix_from_dict(
            pair_bias
        )

        # Define a dictionary to keep track of (token_index_i, token_index_j)
        # -> pair_bias_matrix mappings.
        pair_bias_matrices = {}

        # If there is a global pair-bias matrix, apply it to all residue pairs.
        if np.any(global_pair_bias_matrix != 0.0):
            for token_index_i in range(n_tokens_ca):
                for token_index_j in range(n_tokens_ca):
                    pair_bias_matrices[(int(token_index_i), int(token_index_j))] = (
                        global_pair_bias_matrix
                    )

        # Per-residue-pair overrides.
        if pair_bias_per_residue_pair:
            for res_id_i, res_id_j_to_pair_bias in pair_bias_per_residue_pair.items():
                # Map res_id_i to token index of corresponding CA atom.
                mask_i = MPNNInferenceInput._mask_from_ids(ca_array, [res_id_i])
                i_indices = np.nonzero(mask_i)[0]
                if i_indices.size != 1:
                    raise ValueError(
                        f"Residue ID '{res_id_i}' maps to "
                        f"{i_indices.size} CA atoms; expected exactly 1."
                    )
                token_index_i = int(i_indices[0])

                # Iterate over all res_id_j entries for this res_id_i.
                for res_id_j, i_j_pair_bias in res_id_j_to_pair_bias.items():
                    # Map res_id_j to token index of corresponding CA atom.
                    mask_j = MPNNInferenceInput._mask_from_ids(ca_array, [res_id_j])
                    j_indices = np.nonzero(mask_j)[0]
                    if j_indices.size != 1:
                        raise ValueError(
                            f"Residue ID '{res_id_j}' maps to "
                            f"{j_indices.size} CA atoms; expected exactly 1."
                        )
                    token_index_j = int(j_indices[0])

                    # Build the pair-bias matrix for this specific pair.
                    pair_bias_matrix_ij = (
                        MPNNInferenceInput._build_pair_bias_matrix_from_dict(
                            i_j_pair_bias
                        )
                    )

                    # Skip if the matrix is all zeros.
                    if not np.any(pair_bias_matrix_ij != 0.0):
                        continue

                    # Override global for this specific pair.
                    pair_bias_matrices[(token_index_i, token_index_j)] = (
                        pair_bias_matrix_ij
                    )

        # If there are no non-zero matrices, this is a no-op.
        if not pair_bias_matrices:
            return

        # Build the pairs and values arrays.
        items = list(pair_bias_matrices.items())
        pairs_arr = np.asarray(
            [
                [int(ca_indices[token_index_i]), int(ca_indices[token_index_j])]
                for (token_index_i, token_index_j), _ in items
            ],
            dtype=np.int32,
        )
        values_arr = np.stack(
            [values for _, values in items],
            axis=0,
        ).astype(np.float32)

        # Annotate.
        atom_array.set_annotation_2d("mpnn_pair_bias", pairs_arr, values_arr)

    @staticmethod
    def annotate_atom_array(
        atom_array: AtomArray | AtomArrayPlus,
        input_dict: dict[str, Any],
    ) -> AtomArray | AtomArrayPlus:
        """
        Attach all MPNN-specific annotations to an AtomArray based on the
        (already-validated, default-applied) JSON input dict.

        This function possibly creates the following annotations:
        - 'mpnn_designed_residue_mask'      (bool array)
        - 'mpnn_temperature'                (float32 array)
        - 'mpnn_bias'                       (float32 array)
        - 'mpnn_symmetry_equivalence_group' (int32 array)
        - 'mpnn_symmetry_weight'            (float32 array)
        - 'mpnn_pair_bias'                  (2D annotation)

        NOTE:
        If an annotation already exists on the atom array, the corresponding
        JSON settings are ignored:
        - 'mpnn_designed_residue_mask'      -> design scope fields are ignored
        - 'mpnn_temperature'                -> temperature fields are ignored
        - 'mpnn_bias'                       -> bias/omit fields are ignored
        - 'mpnn_symmetry_equivalence_group' -> symmetry fields are ignored
        - 'mpnn_pair_bias'                  -> pair-bias fields are ignored

        Raises:
            RuntimeError: If pre-existing 'mpnn_symmetry_weight' atom array
                annotation is found without a corresponding
                'mpnn_symmetry_equivalence_group' annotation, raises an error.
        """
        # Discover existing annotations.
        annotation_categories = set(atom_array.get_annotation_categories())
        # 2D annotations, dependent on AtomArrayPlus.
        if isinstance(atom_array, AtomArrayPlus):
            annotation_2d_categories = set(atom_array.get_annotation_2d_categories())
        else:
            annotation_2d_categories = set()

        # Design scope
        if "mpnn_designed_residue_mask" not in annotation_categories:
            MPNNInferenceInput._annotate_design_scope(atom_array, input_dict)

        # Temperature
        if "mpnn_temperature" not in annotation_categories:
            MPNNInferenceInput._annotate_temperature(atom_array, input_dict)

        # Bias / omit
        if "mpnn_bias" not in annotation_categories:
            MPNNInferenceInput._annotate_bias_and_omit(atom_array, input_dict)

        # Symmetry
        if "mpnn_symmetry_equivalence_group" not in annotation_categories:
            # Disallow having symmetry weight annotation without equivalence
            # group annotation.
            if "mpnn_symmetry_weight" in annotation_categories:
                raise RuntimeError(
                    "Inconsistent existing symmetry annotations in atom array: "
                    "'mpnn_symmetry_weight' annotation exists but "
                    "'mpnn_symmetry_equivalence_group' annotation does not."
                )
            MPNNInferenceInput._annotate_symmetry(atom_array, input_dict)

        # Pair bias (2D annotation)
        if "mpnn_pair_bias" not in annotation_2d_categories:
            # Create an AtomArrayPlus.
            atom_array_plus = as_atom_array_plus(atom_array)

            MPNNInferenceInput._annotate_pair_bias(atom_array_plus, input_dict)

            # If pair bias annotation was added, upgrade to AtomArrayPlus.
            new_has_pair_bias = (
                "mpnn_pair_bias" in atom_array_plus.get_annotation_2d_categories()
            )
            if new_has_pair_bias:
                atom_array = atom_array_plus

        return atom_array


###############################################################################
# MPNNInferenceOutput
###############################################################################


@dataclass
class MPNNInferenceOutput:
    """Container for inference output.

    Attributes
    ----------
    atom_array:
        The final, per-design AtomArray to be written/saved.
    output_dict:
        Per-design metadata, not stored in the AtomArray:
            - 'batch_idx'
            - 'design_idx'
            - 'designed_sequence'
            - 'sequence_recovery'
            - 'ligand_interface_sequence_recovery'
            - 'model_type'
            - 'checkpoint_path'
            - 'is_legacy_weights'
    input_dict:
        The JSON-like config dict used for this design.
    """

    atom_array: AtomArray
    output_dict: dict[str, Any]
    input_dict: dict[str, Any]

    def _build_extra_categories(
        self,
    ) -> dict[str, dict[str, Any]]:
        """Convert 'input_dict' and 'output_dict' into CIF 'extra_categories'.

        The result is:
            {
                "mpnn_input": {"col1": [val1], "col2": [val2], ...}
                "mpnn_output": {"col1": [val1], "col2": [val2], ...}
            }

        Nested structures and non-scalar values are converted to strings.
        """
        categories = dict()

        # For both inputs and outputs:
        for category_name, category_dict in [
            ("mpnn_input", self.input_dict),
            ("mpnn_output", self.output_dict),
        ]:
            # Initialize category dict.
            categories[category_name] = dict()

            for key, value in category_dict.items():
                # For scalar values, store directly.
                if isinstance(value, (str, int, float, bool)):
                    categories[category_name][key] = [value]
                # JSON-serializable types: convert to JSON string.
                elif isinstance(value, (list, dict, type(None))):
                    categories[category_name][key] = [json.dumps(value)]
                else:
                    raise ValueError(
                        f"Cannot serialize key {key!r} with value {value!r} "
                        f"of type {type(value)} in category {category_name!r}."
                    )

        return categories

    def write_structure(
        self,
        *,
        base_path: PathLike | None = None,
        file_type: str = "cif",
    ):
        """
        Write this design as a CIF file.

        Parameters
        ----------
        base_path:
            Base path *without* an enforced suffix; the 'file_type' argument
            controls how the suffix is added (e.g. 'cif.gz' -> '.cif.gz').
        file_type:
            One of {'cif', 'bcif', 'cif.gz'}. This is forwarded directly to
            'atomworks.io.utils.io_utils.to_cif_file'.
        """
        if base_path is None:
            raise ValueError("base_path must be provided to write structure.")

        extra_categories = self._build_extra_categories()

        # NOTE: It is not currently possible to save mpnn_bias and
        # mpnn_pair_bias annotations to the CIF file due to shape limitations,
        # so we exclude them here.
        possible_extra_fields = [
            "mpnn_designed_residue_mask",
            "mpnn_temperature",
            "mpnn_symmetry_equivalence_group",
            "mpnn_symmetry_weight",
        ]

        # Limit to fields actually present in the atom array.
        extra_fields = [
            field
            for field in possible_extra_fields
            if field in self.atom_array.get_annotation_categories()
        ]

        # Save to CIF file.
        to_cif_file(
            self.atom_array,
            base_path,
            file_type=file_type,
            extra_fields=extra_fields,
            extra_categories=extra_categories,
        )

    def write_fasta(
        self,
        *,
        base_path: PathLike | None = None,
        handle=None,
    ) -> None:
        """
        Write a single FASTA record for this design.

        Parameters
        ----------
        base_path:
            Base path *without* an enforced suffix; if provided, the final
            path will be '{base_path}.fa'. If None, 'handle' must be provided.
        handle:
            An open writable file-like handle. If None, 'base_path' must be
            provided.
        """
        # At least one of handle or base_path must be provided, and they
        # are mutually exclusive.
        if handle is None and base_path is None:
            raise ValueError("At least one of handle or base_path must be provided.")
        if handle is not None and base_path is not None:
            raise ValueError("handle and base_path are mutually exclusive arguments.")

        # Extract sequence.
        seq = self.output_dict.get("designed_sequence")
        if not seq:
            raise ValueError("No designed_sequence found for FASTA output.")

        # Extract name, batch_idx, and design_idx.
        name = self.input_dict["name"]
        batch_idx = self.output_dict["batch_idx"]
        design_idx = self.output_dict["design_idx"]

        # Extract recovery metrics.
        sequence_recovery = self.output_dict["sequence_recovery"]
        ligand_interface_sequence_recovery = self.output_dict[
            "ligand_interface_sequence_recovery"
        ]

        # Initialize the header fields list.
        header_fields = []

        # Construct the decorated name for the header.
        name_fields = []
        if name is not None:
            name_fields.append(name)
        if batch_idx is not None:
            name_fields.append(f"b{batch_idx}")
        if design_idx is not None:
            name_fields.append(f"d{design_idx}")

        if name_fields:
            decorated_name = "_".join(name_fields)
            header_fields.append(decorated_name)

        # Construct the recovery fields for the header.
        if sequence_recovery is not None:
            header_fields.append(f"sequence_recovery={float(sequence_recovery):.4f}")
        if ligand_interface_sequence_recovery is not None:
            header_fields.append(
                f"ligand_interface_sequence_recovery="
                f"{float(ligand_interface_sequence_recovery):.4f}"
            )

        # Construct the header string.
        header = ">" + ", ".join(header_fields)

        # If the handle is provided, write to it directly.
        if handle is not None:
            # Write the header.
            handle.write(f"{header}\n")

            # Write the sequence.
            handle.write(f"{seq}\n")
        # Otherwise, open the file at base_path and write to it.
        else:
            fasta_path = Path(base_path).with_suffix(".fa")
            with open(fasta_path, "w") as handle:
                # Write the header.
                handle.write(f"{header}\n")

                # Write the sequence.
                handle.write(f"{seq}\n")
