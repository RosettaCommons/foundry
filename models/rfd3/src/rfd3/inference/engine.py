import json
import logging
import os
import time
from os import PathLike
from pathlib import Path
from typing import Dict

import hydra
import torch
import yaml
from lightning.fabric import seed_everything
from omegaconf import OmegaConf
from rfd3.constants import (
    SAVED_CONDITIONING_ANNOTATIONS,
)
from rfd3.inference.datasets import (
    assemble_distributed_inference_loader_from_json,
)
from rfd3.inference.inference_utils import ensure_input_is_abspath
from rfd3.inference.input_parsing import InputSpecification
from toolz import merge_with

from modelhub.common import exists
from modelhub.inference_engines.af3 import AF3InferenceEngine
from modelhub.utils.ddp import RankedLogger, set_accelerator_based_on_availability
from modelhub.utils.io import (
    CIF_LIKE_EXTENSIONS,
    extract_example_id_from_path,
    find_files_with_extension,
)
from modelhub.utils.logging import print_config_tree

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def normalize_inputs(inputs: str | list | None) -> list[str | None]:
    """
    inputs: str | list[str] | None
        - Can be:
            - A single path to a JSON, YAML, or regular input file (cif or pdb)
            - A comma-separated string of paths (e.g. "a.json,b.json")
            - A list of file paths
            - None or an empty list, in which case a dummy input is added (used for e.g. motif-only design)
        - Returns list of paths or [None] if no inputs are provided
    """
    if inputs is None or (isinstance(inputs, list) and len(inputs) == 0):
        inputs = [None]
    elif isinstance(inputs, str):
        inputs = inputs.split(",")
    elif not isinstance(inputs, list):
        raise ValueError(
            f"Invalid input type: {type(inputs)}. Expected str, list, or None.\nInput: {inputs}"
        )
    return inputs


def process_input(
    inputs: str | list | None,
    json_keys_subset: str | list | None = None,
    global_prefix: str = None,
    specification_overrides: dict = {},
) -> Dict[str, dict]:
    """
    inputs: Any -> list[str | None] (see normalize_inputs)
    json_keys_subset: extract only subset of JSON keys. None will keep all keys
    prefix: If provided, prefix all example ids with said prefix

    returns: Dictionaries of specifcation args pre-batching:
        {
            'jsonfile_jsonkey1': {
                **args_from_key1
            },
            'jsonfile_jsonkey2': {
                **args_from_key2
            }
        }
    """
    merge_args = lambda d: merge_with(lambda x: x[-1], d, specification_overrides)  # noqa
    inputs = normalize_inputs(inputs)

    # If global_prefix is not provided, then default to using the basename of the JSON or YAML file (when provided)
    if global_prefix is None:
        use_json_basename_prefix = True
    else:
        use_json_basename_prefix = False

    # ... Convert all inputs to list of inputs (e.g. if comma-separated)
    if exists(inputs) and "," in inputs:
        inputs = inputs.split(",")
    elif not exists(inputs):
        # If inputs is None or empty, we will create a dummy input
        inputs = []
    inputs = inputs if isinstance(inputs, list) else [inputs]
    if len(inputs) == 0:
        inputs = [None]

    # ... Determine prefix of sample to create
    all_specs = {}
    for input in inputs:
        if exists(input) and (input.endswith(".json") or input.endswith(".yaml")):
            # ... Load JSON or YAML file
            with open(input, "r") as f:
                data = json.load(f) if input.endswith(".json") else yaml.safe_load(f)

            # ... Apply any global args for this input file
            if "global_args" in data:
                global_args = data.pop("global_args")
                for example in data:
                    data[example].update(global_args)

            # ... Subset to keys
            if json_keys_subset is not None:
                json_keys_subset = (
                    json_keys_subset.split(",")
                    if isinstance(json_keys_subset, str)
                    else json_keys_subset
                )
                data = {
                    example: data[example]
                    for example in json_keys_subset
                    if example in data
                }

            # ... Extract each accumulated example in data.
            for example, args in data.items():
                args = ensure_input_is_abspath(args, input)
                if use_json_basename_prefix:
                    name = os.path.splitext(os.path.basename(input))[0]
                    prefix = f"{name}_{example}"
                else:
                    prefix = f"{global_prefix}{example}"
                args["extra"] = args.get("extra", {}) | {"example": example}
                all_specs[prefix] = dict(merge_args(args))

        elif exists(input):
            prefix = os.path.basename(os.path.splitext(input)[0])
            if global_prefix is not None:
                prefix = f"{global_prefix}{prefix}"
            all_specs[prefix] = dict(merge_args({"input": input}))
        else:
            all_specs["backbone"] = specification_overrides

    return all_specs


class RFD3InferenceEngine(AF3InferenceEngine):
    """Inference engine for RFdiffusion3"""

    def __init__(
        self,
        # Required args:
        out_dir: str | PathLike,
        ckpt_path: str | PathLike,
        inputs: str | PathLike,
        json_keys_subset: None | list[str],
        n_batches: int,
        *,
        # Default design specification args:
        specification: dict,
        # Base inference engine args
        diffusion_batch_size: int,
        skip_existing: bool,
        inference_sampler: dict,
        # Structure dumping arguments
        cleanup_guideposts: bool,
        cleanup_virtual_atoms: bool,
        read_sequence_from_sequence_head: bool,
        output_full_json: bool,
        dump_prediction_metadata_json: bool,
        dump_trajectories: bool,
        align_trajectory_structures: bool,
        one_model_per_file: bool,
        global_prefix: str | None,
        ###############################################
        num_nodes: int,
        devices_per_node: int,
        print_config: bool,
        seed: int | None,
        temp_dir=None,
        low_memory_mode: bool = False,
        prevalidate_inputs: bool = True,
        # Atom array instantiation args collapsed into default dict:
    ):
        """
        Design specification args:
            # Main:
            inputs: JSON, YAML, PDB or CIF (comma-separated string or single) containing coordinate data to be parsed or args to override
            length: length of designed structure (int or str like '20-100'). Default: None (specified by contig string)
            contig: string of residues to use as backbone. Default: None (specified by length) Example: '10-20,A11-13,10-20'
            fixed_atoms: Dict[str] of atom names to use as indexed motif atoms with fixed coordinates. Default: None
            unindex: list of residues in input_src to unindex in design. Default: None

            redesign_motif_sidechains: bool or str. Specifies which motif residues have fixed sidechains by default. Default: True
            ligand: str. Ligands in input file to include as motif
            atomwise_rasa: Dict[str] of atomwise rasas to use for design. Default: None
            ori_token: list of 3 floats indicating the origin to center the design around. Default: None
            seed: int or None. If None, a random seed will be sampled.

        Additional args:
            ckpt_path: Path to checkpoint file
            n_batches: Number of samples to
        """
        if not os.path.isabs(out_dir):
            out_dir = os.path.abspath(out_dir)
            ranked_logger.info("Using absolute path for out_dir: {}".format(out_dir))

        # Convert input sources to design specification dictionaries
        inputs = process_input(
            inputs,
            json_keys_subset=json_keys_subset,
            global_prefix=global_prefix,
            specification_overrides=specification,
        )  # any -> Dict[Name: InputSpecification]
        self.design_specifications = {}
        for prefix, example_spec in inputs.items():
            # ... Set example key as the prefix
            if prevalidate_inputs:
                ranked_logger.info(
                    f"Prevalidating design specification for example: {prefix}"
                )
                InputSpecification.safe_init(**example_spec)

            # ... Create n_batches for example
            for batch_i in range(n_batches):
                # ... Example ID
                example_id = f"{prefix}_{batch_i}"
                self.design_specifications[example_id] = example_spec

        ############################################################
        # Feed-forward inputs similar to MH-AF3 inference engine
        ############################################################

        # ... set the random seed for reproducibility (and for augmentation, e.g., for antibodies)
        if not exists(seed):
            seed = int(time.time() * 1000) % (2**31)
        ranked_logger.info(f"Seeding everything with seed={seed}...")
        seed_everything(seed, workers=True, verbose=True)
        self.seed = seed

        # We only extract the `train_cfg` from the checkpoint initially
        self.load_and_override_ckpt_config(
            ckpt_path=ckpt_path,
            num_nodes=num_nodes,
            devices_per_node=devices_per_node,
            inference_sampler=inference_sampler,
        )

        set_accelerator_based_on_availability(self.cfg)

        # (b) based on the dataset (we will apply when constructing the pipeline)
        self.dataset_overrides = {
            "diffusion_batch_size": diffusion_batch_size,
        }

        # ... instantiate the trainer with the (modified) configuration
        self.trainer = hydra.utils.instantiate(
            self.cfg.trainer,
            _convert_="partial",
            _recursive_=False,
        )

        # Set the output directory for the CIF files (e.g., predicted structures)
        self.cif_out_dir = Path(out_dir) if out_dir else Path("./")

        # Structure dumping
        self.dump_prediction_metadata_json = dump_prediction_metadata_json
        self.dump_trajectories = dump_trajectories
        self.one_model_per_file = one_model_per_file
        self.align_trajectory_structures = align_trajectory_structures

        self.trainer.cleanup_virtual_atoms = cleanup_virtual_atoms
        self.trainer.cleanup_guideposts = cleanup_guideposts
        self.trainer.read_sequence_from_sequence_head = read_sequence_from_sequence_head
        self.trainer.output_full_json = output_full_json
        self.trainer.inference_sampler_overrides = inference_sampler
        self.prediction_extra_fields = SAVED_CONDITIONING_ANNOTATIONS
        self.skip_existing = skip_existing
        self.dump_predictions = True
        self.print_config = print_config

        if not cleanup_guideposts:
            ranked_logger.warning(
                "Guideposts will not be cleaned up. This is intended for debugging purposes."
            )
        if not cleanup_virtual_atoms:
            ranked_logger.warning(
                "Virtual atoms will not be cleaned up. Some tools like MPNN may run, but outputs will not be like native structures."
            )

        # Check which example ids already exist in the output directory
        self.existing_example_ids = set(
            extract_example_id_from_path(path, CIF_LIKE_EXTENSIONS)
            for path in find_files_with_extension(out_dir, CIF_LIKE_EXTENSIONS)
        )
        ranked_logger.info(
            f"Found {len(self.existing_example_ids)} existing example IDs in the output directory."
        )

        if low_memory_mode:
            ranked_logger.info("Low memory mode enabled.")
            # HACK: Set attribute to the diffusion module
            os.environ["RFD3_LOW_MEMORY_MODE"] = "1"

    def load_and_override_ckpt_config(
        self, ckpt_path, num_nodes, devices_per_node, inference_sampler
    ):
        assert exists(ckpt_path), f"Checkpoint path ({ckpt_path}) not provided."
        ranked_logger.info(f"Loading checkpoint from {Path(ckpt_path).resolve()}...")
        self.ckpt_path = ckpt_path

        self.cfg = OmegaConf.create(
            torch.load(self.ckpt_path, "cpu", weights_only=False)["train_cfg"]
        )

        # Override specific parameters within the Hydra config:
        #  (a) based on the input arguments
        self.cfg.trainer.num_nodes = num_nodes
        self.cfg.trainer.devices_per_node = devices_per_node
        for k, v in inference_sampler.items():
            if v is None:
                continue
            setattr(self.cfg.model.net.inference_sampler, k, v)

        # Set metrics / callbacks to be null s.t. they aren't loaded
        self.cfg.trainer.metrics = None

        # Record the random seed to be dumped in the output JSON
        self.cfg.trainer.seed = self.seed

    def example_id_exists(self, example_id, verbose=False):
        # TODO: Move this to another file to standardize better with src
        if not self.one_model_per_file:
            # Check if one file exists
            all_exist = example_id in self.existing_example_ids
            if all_exist and verbose:
                ranked_logger.info(
                    f"Model file for example {example_id} already exists in the output directory."
                )
        else:
            all_exist = all(
                [
                    (f"{example_id}_model_{i}" in self.existing_example_ids)
                    for i in range(self.dataset_overrides["diffusion_batch_size"])
                ]
            )
            if all_exist and verbose:
                ranked_logger.info(
                    f"All models for example {example_id} already exist in the output directory."
                )
        return all_exist

    def eval(self):
        """
        Run design on a set of specifications
        """
        if self.print_config:
            print_config_tree(
                self.cfg.model, resolve=True, title="INFERENCE MODEL CONFIGURATION"
            )

        # ... spawn processes for distributed training, if using multiple GPUs
        ranked_logger.info(
            f"Spawning {self.trainer.fabric.world_size} processes from {self.trainer.fabric.global_rank}..."
        )

        # ==============================================================================
        # Construct the model and load the checkpoint
        # ==============================================================================
        self.trainer.initialize_or_update_trainer_state({"train_cfg": self.cfg})
        self.trainer.construct_model()
        self.trainer.load_checkpoint(ckpt_path=self.ckpt_path, is_inference=True)

        # Ensure optimizer isn't loaded
        self.trainer.state["optimizer"] = None
        self.trainer.state["train_cfg"].model.optimizer = None

        self.trainer.setup_model_optimizers_and_schedulers()
        self.trainer.state["model"].eval()

        # ==============================================================================
        # Prepare pipeline and inference loader
        # ==============================================================================
        # TODO: have name be the basename of the JSON or YAML file
        loader = assemble_distributed_inference_loader_from_json(
            # Passed directly to ContigJSONDataset
            data=self.design_specifications,
            transform=self.construct_pipeline(),
            name="inference-dataset",
            cif_parser_args={},
            subset_to_keys=None,
            eval_every_n=1,
            # Sampler args
            world_size=self.trainer.fabric.world_size,
            rank=self.trainer.fabric.global_rank,
        )
        loader = self.trainer.fabric.setup_dataloaders(
            loader,
            use_distributed_sampler=False,
        )

        # ==============================================================================
        # Evaluate, using `validation_step``
        # ==============================================================================

        for batch_idx, batch in enumerate(loader):
            pipeline_output = batch[0]
            example_id = pipeline_output["example_id"]

            if self.skip_existing:
                if self.example_id_exists(example_id, verbose=True):
                    ranked_logger.info(
                        f"Skipping structure {batch_idx + 1}/{len(loader)}: {example_id} | Already exists."
                    )
                    continue
            else:
                ranked_logger.info(
                    f"Predicting structure {batch_idx + 1}/{len(loader)}: {example_id}"
                )

            # Model inference
            t0 = time.time()
            with torch.no_grad():
                pipeline_output = self.trainer.fabric.to_device(pipeline_output)
                output = self.trainer.validation_step(
                    batch=pipeline_output,
                    batch_idx=0,
                    compute_metrics=False,
                )
            t_end = time.time()

            # Add additional information to prediction metadata
            for key in output["prediction_metadata"].keys():
                ckpt = Path(self.ckpt_path)
                if ckpt.is_symlink():
                    ckpt = ckpt.resolve(strict=True)  # follow symlink to target
                output["prediction_metadata"][key]["ckpt_path"] = str(ckpt)
                output["prediction_metadata"][key]["seed"] = self.seed

            ranked_logger.info(f"Finished inference batch in {t_end - t0:.2f} seconds.")
            self.save_batch_outputs(
                example_id=example_id,
                network_output=output["network_output"],
                prediction_metadata=output["prediction_metadata"],
                predicted_atom_array_stack=output["predicted_atom_array_stack"],
                pipeline_output=pipeline_output,
            )
