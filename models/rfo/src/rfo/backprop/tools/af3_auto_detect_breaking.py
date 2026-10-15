import json
import re
import textwrap
from collections import defaultdict
from os import PathLike
from pathlib import Path

import numpy as np
import torch
import tree
from atomworks.constants import DICT_THREE_TO_ONE
from atomworks.io.parser import ParseConfig, parse
from atomworks.io.tools.inference import (
    build_msa_paths_by_chain_id_from_component_list,
    components_to_atom_array,
)
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.io.utils.query import AtomSelectionStack
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.transforms.template import add_input_file_template
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from rfo.backprop.tools.datahub_custom import featurize_templates_like_af3_custom

from .utils import write_json


def decode_restype(restype_onehot, encoding, protein_mask=None, prot_dim=20):
    """
    Decode a one-hot encoded residue type tensor.
    Use argmax to get the index of the maximum value in the tensor.
    """
    if protein_mask is None:
        protein_mask = torch.ones(restype_onehot.size(0), dtype=torch.bool).to(
            restype_onehot.device
        )
    restype_str = encoding.decode(
        torch.argmax(
            restype_onehot[protein_mask.to(restype_onehot.device), :prot_dim], dim=-1
        )
        .cpu()
        .numpy()
    )
    return "".join([DICT_THREE_TO_ONE[res] for res in restype_str])


def decode_alltype(restype_onehot, encoding, seq_mask=None):
    """
    Decode a one-hot encoded residue type tensor.
    Use argmax to get the index of the maximum value in the tensor.
    """
    if seq_mask is None:
        seq_mask = torch.ones(restype_onehot.size(0), dtype=torch.bool).to(
            restype_onehot.device
        )
    restype_str = encoding.decode(
        torch.argmax(restype_onehot[seq_mask.to(restype_onehot.device), :], dim=-1)
        .cpu()
        .numpy()
    )
    return "".join([DICT_THREE_TO_ONE[res] for res in restype_str])


def seq2cif(seq, add_info=None, protein_mask=None, out_file=None):
    """
    Convert a all biomolecule sequence to a CIF file.

    Args:
        seq (str): Protein sequence string.
        add_info (dict, optional): Additional information like chain IDs, SMILES, MSA paths.
        protein_mask (torch.Tensor, optional): Mask indicating protein residues.
        out_file (str, optional): Output CIF file path. Defaults to './tmp.cif.gz'.

    Returns:
        tuple: Path to the saved CIF file and inputs dictionary.
    """
    if add_info is None:
        add_info = {
            "chain_id": None,
            "smiles": None,
            "msa_paths": None,
            "chain_iid_token_level": None,
        }
    if out_file is None:
        out_file = "./tmp.cif.gz"
    if protein_mask is not None:
        seq = "".join([seq[i] for i in range(len(seq)) if protein_mask[i]])
    if isinstance(protein_mask, torch.Tensor):
        protein_mask = protein_mask.clone().detach().cpu().numpy()

    # multiple chains handling
    if add_info["chain_iid_token_level"] is not None:
        # If we have more than one chain, it's necessary to construct inputs as a list of dictionaries
        # Each dictionary corresponds to a chain and contains the chain ID and MSA path
        chains = np.unique(add_info["chain_iid_token_level"][protein_mask])
        inputs = []
        for chain in chains:
            chain_id = str(chain)
            inputs.append(
                {
                    "seq": "".join(
                        [
                            seq[i]
                            for i in range(len(seq))
                            if add_info["chain_iid_token_level"][i] == chain
                        ]
                    ),
                    "chain_id": chain_id.split("_")[0],
                    "chain_type": "polypeptide(L)",
                    "msa_path": None,
                }
            )
    else:
        inputs = [
            {
                "seq": seq,
                "chain_id": add_info["chain_id"],
                "chain_type": "polypeptide(L)",
                "msa_path": None,
            },
        ]

    if add_info["msa_paths"] is not None:
        # If we have MSA paths, we need to add them to the inputs
        # This is a list of dictionaries, each containing the MSA path for a chain
        for chain_id, msa_path in add_info["msa_paths"].items():
            # Find the corresponding input dictionary for this chain
            for input_dict in inputs:
                if input_dict["chain_id"] == chain_id:
                    input_dict["msa_path"] = msa_path
                    print(f"MSA: Using MSA path for chain {chain_id}: {msa_path}")
    if add_info["smiles"] is not None:
        inputs.append({"smiles": add_info["smiles"]})

    if add_info.get("ligand_cif") is not None:
        inputs.append({"path": add_info.get("ligand_cif")})

    # remove the key if the value is None
    cleaned_inputs = [
        {key: value for key, value in item.items() if value is not None}
        for item in inputs
    ]
    # remove the empty dict
    cleaned_inputs = [item for item in cleaned_inputs if item]
    atom_array, components = components_to_atom_array(
        cleaned_inputs, return_components=True
    )
    msa_paths_by_chain_id = build_msa_paths_by_chain_id_from_component_list(components)
    # save the spoofed CIF file
    save_path = to_cif_file(
        atom_array,
        out_file,
        extra_categories={"msa_paths_by_chain_id": msa_paths_by_chain_id}
        if msa_paths_by_chain_id
        else None,
    )

    return save_path, inputs


def apply_chain_break_renumbering(
    atom_array, mode="big_gap", max_distance=3.0, angstroms_per_aa=4.0
):
    """
    Apply chain break renumbering by detecting breaking points and adding gaps in residue numbering.

    Args:
        atom_array: The AtomArray to process
        mode (str): Renumbering mode - 'big_gap', 'input_numbering', or 'rough_guess'
        max_distance (float): Maximum distance in Angstroms to consider residues connected
        angstroms_per_aa (float): Angstroms per amino acid for rough distance estimation

    Returns:
        AtomArray: Modified atom_array with renumbered residues
    """
    if mode == "input_numbering":
        # Keep original numbering
        return atom_array

    # Process each chain separately
    unique_chains = np.unique(atom_array.chain_id)

    for chain_id in unique_chains:
        # Get atoms for this chain
        chain_mask = atom_array.chain_id == chain_id
        chain_atoms = atom_array[chain_mask]

        # Get unique residues in this chain, sorted by original res_id
        unique_res_ids = np.unique(chain_atoms.res_id)
        unique_res_ids = np.sort(unique_res_ids)

        if len(unique_res_ids) <= 1:
            continue

        new_res_ids = np.zeros_like(chain_atoms.res_id)
        last_new_res_id = 0

        for i, res_id in enumerate(unique_res_ids):
            # Get atoms for current residue
            current_res_mask = chain_atoms.res_id == res_id
            current_res_atoms = chain_atoms[current_res_mask]

            # Check for chain break if not the first residue
            is_chain_break = False
            if i > 0:
                # Get previous residue
                prev_res_id = unique_res_ids[i - 1]
                prev_res_mask = chain_atoms.res_id == prev_res_id
                prev_res_atoms = chain_atoms[prev_res_mask]

                # Find C atom of previous residue and N atom of current residue
                prev_c_mask = prev_res_atoms.atom_name == "C"
                curr_n_mask = current_res_atoms.atom_name == "N"

                if np.any(prev_c_mask) and np.any(curr_n_mask):
                    prev_c_coord = prev_res_atoms.coord[prev_c_mask][0]
                    curr_n_coord = current_res_atoms.coord[curr_n_mask][0]

                    # Calculate distance
                    distance = np.linalg.norm(curr_n_coord - prev_c_coord)
                    is_chain_break = distance > max_distance

            # Assign new residue ID
            if is_chain_break:
                if mode == "big_gap":
                    new_res_id = last_new_res_id + 201
                elif mode == "rough_guess":
                    gap_size = int(np.ceil(distance / angstroms_per_aa))
                    new_res_id = last_new_res_id + gap_size
                else:
                    new_res_id = last_new_res_id + 1
            else:
                new_res_id = last_new_res_id + 1

            # Update all atoms in this residue
            new_res_ids[current_res_mask] = new_res_id
            last_new_res_id = new_res_id

        # Apply new residue IDs to the original atom_array
        atom_array.res_id[chain_mask] = new_res_ids

    return atom_array


def cif2input(cif, pipeline, device="cpu", handle_chain_breaks=True):
    """
    Transform a CIF file into network input suitable for the AF3 model pipeline.

    Args:
        cif (PathLike): Path to the CIF file
        pipeline: The pipeline transform from the model
        device (str): Device to place tensors on
        handle_chain_breaks (bool): Whether to apply chain break renumbering (default: True)

    Returns:
        tuple: (network_input, confidence_feats, pipeline_output)
    """
    if not isinstance(cif, PathLike):
        cif = Path(cif)
    # ... parse into an AtomArray (`parse` handles all valid formats)
    out = parse(cif, config=ParseConfig(hydrogen_policy="remove"))
    # ... get the atom array and set NaN coordinates to random
    atom_array = (
        out["assemblies"]["1"][0] if "assemblies" in out else out["asym_unit"][0]
    )
    # AW3 preserves file order; sequence reconstruction sorts chain IDs.
    # Keep initial and reconstructed token order aligned (binder A first).
    atom_array = atom_array[np.argsort(atom_array.chain_id, kind="stable")]
    atom_array.coord[np.isnan(atom_array.coord)] = np.random.rand(
        *atom_array.coord[np.isnan(atom_array.coord)].shape
    )
    # ... apply chain break renumbering to handle discontinuous sequences (default: enabled)
    if handle_chain_breaks:
        atom_array = apply_chain_break_renumbering(atom_array, mode="big_gap")
    # ... assemble the pipeline input in a format compatible with the DataHub pipeline
    pipeline_input = {
        "example_id": cif.name.split(".")[0],
        "atom_array": atom_array,
        "chain_info": out["chain_info"],
    }
    # ... run dataloading and featurization
    pipeline_output = pipeline(pipeline_input)

    network_input = {
        "X_noisy_L": torch.nan_to_num(pipeline_output["coord_atom_lvl_to_be_noised"])
        + pipeline_output["noise"],
        "t": pipeline_output["t"],
        "f": pipeline_output["feats"],
    }

    confidence_feats = {}

    if "confidence_feats" in pipeline_output:
        # map "rf2aa_seq" to "seq" (required by confidence head)
        if "rf2aa_seq" in pipeline_output["confidence_feats"]:
            confidence_feats["seq"] = pipeline_output["confidence_feats"]["rf2aa_seq"]

        # map "pae_frame_idx_token_lvl_from_atom_lvl" to "frame_atom_idxs" (required by confidence head)
        if (
            "pae_frame_idx_token_lvl_from_atom_lvl"
            in pipeline_output["confidence_feats"]
        ):
            confidence_feats["frame_atom_idxs"] = pipeline_output["confidence_feats"][
                "pae_frame_idx_token_lvl_from_atom_lvl"
            ]

        # copy other potentially useful fields directly
        for key in ["atom_frames", "is_real_atom"]:
            if key in pipeline_output["confidence_feats"]:
                confidence_feats[key] = pipeline_output["confidence_feats"][key]

    if "ground_truth" in pipeline_output:
        # map rep_atom_idxs from ground_truth (required by confidence head)
        if "rep_atom_idxs" in pipeline_output["ground_truth"]:
            confidence_feats["rep_atom_idxs"] = pipeline_output["ground_truth"][
                "rep_atom_idxs"
            ]

        # copy chain_iid_token_lvl for chain information
        if "chain_iid_token_lvl" in pipeline_output["ground_truth"]:
            confidence_feats["chain_iid_token_lvl"] = pipeline_output["ground_truth"][
                "chain_iid_token_lvl"
            ]

    network_input = tree.map_structure(
        lambda x: x.to(device) if hasattr(x, "to") else x, network_input
    )
    confidence_feats = tree.map_structure(
        lambda x: x.to(device) if hasattr(x, "to") else x, confidence_feats
    )
    pipeline_output = tree.map_structure(
        lambda x: x.to(device) if hasattr(x, "to") else x, pipeline_output
    )

    return network_input, confidence_feats, pipeline_output


def input_cif_prep(path, temp_dir="./temp"):
    """
    Prepare a CIF file for input to the AF3 model.

    Args:
        path (str or Path): Path to the input file (JSON, CIF, or PDB)
        temp_dir (str): Directory to store temporary files

    Returns:
        Path: Path to the prepared CIF file
    """
    path = Path(path)
    if path.suffix in {".json"}:
        with open(path, "r") as json_file:
            # Load the JSON data
            inputs = json.load(json_file)

            # Build components
            atom_array, components = components_to_atom_array(
                inputs[0]["components"], return_components=True
            )
            msa_paths_by_chain_id = build_msa_paths_by_chain_id_from_component_list(
                components
            )

            # Create a temporary CIF file from the JSON data
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            cif_path = temp_dir / f"{path.stem}.cif"
            save_path = to_cif_file(
                atom_array,
                cif_path,
                extra_categories={"msa_paths_by_chain_id": msa_paths_by_chain_id}
                if msa_paths_by_chain_id
                else None,
            )
            return Path(save_path)

    elif path.suffix in {".cif", ".pdb"}:
        print(f"Using existing structure file: {path}")
        return path

    else:
        raise ValueError(f"Unsupported file format: {path}")


def modelhub2af3(modelhub_inputs, output_path=None):
    """
    Convert modelhub inputs to AF3 format JSON.

    Args:
        modelhub_inputs (list): List of modelhub input dictionaries
        output_path (str, optional): Path to save the JSON output

    Returns:
        list: List of AF3 format dictionaries, or None if output_path is provided
    """
    af3_json = []
    for index, modelhub_input in enumerate(modelhub_inputs):
        af3_entities = {
            "name": f"{index}",
            "sequences": [],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1,
        }

        for entity in modelhub_input:
            item = {
                "protein": {
                    "id": entity["chain_id"],
                    "sequence": entity["seq"],
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": [],
                    "unpairedMsaPath": entity.get("msa_path", ""),
                }
            }
            af3_entities["sequences"].append(item)
        af3_json.append(af3_entities)

    if output_path:
        write_json(af3_json, output_path)
        return None
    else:
        return af3_json


def fix_cif_auth_seq_id(cif_in, cif_out=None):
    """
    read a mmCIF file, if it lacks _atom_site.auth_seq_id / auth_asym_id,
    copy them from label_seq_id / label_asym_id, and write out a new CIF file.
    """
    cif_in = cif_in
    if cif_out is None:
        cif_out = cif_in.replace(".cif", "_fix.cif")
    cif_in = Path(cif_in)
    cif_out = Path(cif_out)
    cif_dict = MMCIF2Dict(str(cif_in))

    atom_count = len(cif_dict.get("_atom_site.label_atom_id", []))

    def ensure(key, fallback_key=None, default="?"):
        if key not in cif_dict:
            if fallback_key and fallback_key in cif_dict:
                cif_dict[key] = cif_dict[fallback_key]
            else:
                cif_dict[key] = [default] * atom_count

    ensure("_atom_site.auth_seq_id", fallback_key="_atom_site.label_seq_id")
    ensure("_atom_site.auth_asym_id", fallback_key="_atom_site.label_asym_id")

    with cif_out.open("w") as f:
        f.write(f"data_{cif_out.stem}\n#\nloop_\n")

        columns = [c for c in cif_dict if c.startswith("_atom_site.")]
        columns.sort()
        for col in columns:
            f.write(f"{col}\n")
        for i in range(atom_count):
            line = ""
            for col in columns:
                line += f"{cif_dict[col][i]} "
            f.write(textwrap.fill(line.rstrip(), width=9999) + "\n")
        f.write("#\n")

    print(f"[fix] {cif_in.name} → {cif_out.name}")


def mark_whole_chain_as_template(atom_array, chain_id: str):
    mask = atom_array.chain_id == chain_id
    atom_array.set_annotation("is_input_file_templated", mask)
    return atom_array


CHAIN_ONLY_RE = re.compile(r"^[A-Za-z](,[A-Za-z])*$")


def apply_template_selection(atom_array, self_template_selection_syntax: str | None):
    """
    mark atom_array with `is_input_file_templated` annotation based on the provided self_template_selection_syntax.
    self_template_selection_syntax
        - only chain:   "A" or "A,B"   → mark the whole chain as template
        - segment syntax: "A1103-1120"   → still use AtomSelectionStack
        - None:         do not mark as template
    """
    if self_template_selection_syntax is None:
        mask = np.zeros(len(atom_array), dtype=bool)

    elif CHAIN_ONLY_RE.match(self_template_selection_syntax.replace(" ", "")):
        chain_ids = [cid.strip() for cid in self_template_selection_syntax.split(",")]
        mask = np.isin(atom_array.chain_id, chain_ids)

    else:
        # Compact contigs and path queries are distinct AtomWorks grammars.
        # Passing A1103-1120 to from_query treats it as a literal chain ID.
        syntax = self_template_selection_syntax.replace(" ", "")
        if re.fullmatch(r"[A-Za-z]+[0-9]+-[0-9]+(?:,[A-Za-z]+[0-9]+-[0-9]+)*", syntax):
            selector = AtomSelectionStack.from_contig(syntax)
        else:
            selector = AtomSelectionStack.from_query(self_template_selection_syntax)
        mask = selector.get_mask(atom_array)

    if np.any(mask):
        assert not np.all(
            np.isnan(atom_array.coord[mask])
        ), "Selected atoms for templating have NaN coords."
        assert np.all(
            atom_array[mask].is_polymer
        ), "Templating only supports polymer now"

    atom_array.set_annotation("is_input_file_templated", mask)
    return atom_array


def cif2input_with_template(
    cif,
    pipeline,
    device: str = "cpu",
    self_template_selection_syntax: str | None = None,
    fix_template_dict: dict | None = None,
    msa_paths_by_chain_id: dict[str, str] | None = None,
    handle_chain_breaks: bool = True,
):
    cif = Path(cif) if not isinstance(cif, PathLike) else cif
    out = parse(cif, config=ParseConfig(hydrogen_policy="remove"))

    atom_array = (
        out["assemblies"]["1"][0] if "assemblies" in out else out["asym_unit"][0]
    )
    # AW3 preserves file order; sequence reconstruction sorts chain IDs.
    # Keep initial and reconstructed token order aligned (binder A first).
    atom_array = atom_array[np.argsort(atom_array.chain_id, kind="stable")]
    nan_mask = np.isnan(atom_array.coord)
    atom_array.coord[nan_mask] = np.random.rand(*atom_array.coord[nan_mask].shape)
    # ... apply chain break renumbering to handle discontinuous sequences (default: enabled)
    if handle_chain_breaks:
        atom_array = apply_chain_break_renumbering(atom_array, mode="big_gap")

    atom_array = apply_template_selection(atom_array, self_template_selection_syntax)

    # set the msa info
    chain_info = out["chain_info"]
    if msa_paths_by_chain_id:
        for chain_id, msa_path in msa_paths_by_chain_id.items():
            if chain_id not in chain_info:
                raise ValueError(f"Chain {chain_id} not found in CIF.")
            chain_info[chain_id]["msa_path"] = str(msa_path)

    pipeline_input = {
        "example_id": cif.stem,
        "atom_array": atom_array,
        "chain_info": out["chain_info"],
    }

    pipeline_output = pipeline(pipeline_input)

    # for i, t in enumerate(pipeline.transforms):
    #     if isinstance(t, AddWithinPolyResIdxAnnotation):
    #         print(f"Template featurization starts at index {i}")

    pipeline_data_clean = pipeline[:16]
    if fix_template_dict is not None:
        template_feats = template_dict2template_feats(
            template_dict=fix_template_dict,
            pipeline_data_clean=pipeline_data_clean,
            pipeline_output=pipeline_output,
            encoding=None,
        )
        for key, value in template_feats.items():
            if key == "template_restype":
                # someone does the one-hot encoding for template_restype separately,
                # so we need to do it here too
                value = torch.nn.functional.one_hot(
                    value, num_classes=32
                ).float()  # encoding.n_tokens

            pipeline_output["feats"][key] = value
    network_input = {
        "X_noisy_L": torch.nan_to_num(pipeline_output["coord_atom_lvl_to_be_noised"])
        + pipeline_output["noise"],
        "t": pipeline_output["t"],
        "f": pipeline_output["feats"],
    }

    confidence_feats = {}
    if "confidence_feats" in pipeline_output:
        cf = pipeline_output["confidence_feats"]
        if "rf2aa_seq" in cf:
            confidence_feats["seq"] = cf["rf2aa_seq"]
        if "pae_frame_idx_token_lvl_from_atom_lvl" in cf:
            confidence_feats["frame_atom_idxs"] = cf[
                "pae_frame_idx_token_lvl_from_atom_lvl"
            ]
        for k in ("atom_frames", "is_real_atom"):
            if k in cf:
                confidence_feats[k] = cf[k]

    if "ground_truth" in pipeline_output:
        gt = pipeline_output["ground_truth"]
        if "rep_atom_idxs" in gt:
            confidence_feats["rep_atom_idxs"] = gt["rep_atom_idxs"]
        if "chain_iid_token_lvl" in gt:
            confidence_feats["chain_iid_token_lvl"] = gt["chain_iid_token_lvl"]

    network_input = tree.map_structure(
        lambda x: x.to(device) if hasattr(x, "to") else x, network_input
    )
    confidence_feats = tree.map_structure(
        lambda x: x.to(device) if hasattr(x, "to") else x, confidence_feats
    )
    pipeline_output = tree.map_structure(
        lambda x: x.to(device) if hasattr(x, "to") else x, pipeline_output
    )

    return network_input, confidence_feats, pipeline_output


def template_dict2template_feats(
    template_dict: dict,
    pipeline_data_clean,  # pipeline data
    pipeline_output,  # data
    encoding=None,
):
    templates_by_chain = defaultdict(list)
    for chain_id, template_file in template_dict.items():
        template_out = parse(
            template_file, config=ParseConfig(hydrogen_policy="remove")
        )

        template_aa = (
            template_out["assemblies"]["1"][0]
            if "assemblies" in template_out
            else template_out["asym_unit"][0]
        )

        template_aa = pipeline_data_clean(
            {
                "example_id": "template",
                "atom_array": template_aa,
                "chain_info": template_out["chain_info"],
            }
        )["atom_array"]

        nan_mask = np.isnan(template_aa.coord)
        template_aa.coord[nan_mask] = np.random.rand(*template_aa.coord[nan_mask].shape)
        template_aa.chain_id[:] = chain_id
        template_aa.set_annotation(
            "is_input_file_templated", np.ones(len(template_aa), bool)
        )

        tmpl_dict = add_input_file_template(template_aa)  # {'A':[...]}
        for cid, lst in tmpl_dict.items():
            templates_by_chain[cid].extend(lst)

    templates_by_chain = dict(templates_by_chain)
    all_atom_array = pipeline_output["atom_array"]
    if encoding is None:
        encoding = AF3SequenceEncoding()
    template_feats = featurize_templates_like_af3_custom(
        atom_array=all_atom_array,
        templates_by_chain=templates_by_chain,
        sequence_encoding=encoding,
    )

    return template_feats
