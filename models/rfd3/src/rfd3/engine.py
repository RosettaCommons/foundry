import json
import logging
import os
import time
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from biotite.structure import AtomArray
from toolz import merge_with

from modelhub.common import exists
from modelhub.inference_engines.base import BaseInferenceEngine
from modelhub.utils.ddp import RankedLogger
from rfd3.constants import SAVED_CONDITIONING_ANNOTATIONS
from rfd3.inference.datasets import (
    assemble_distributed_inference_loader_from_json,
)
from rfd3.inference.input_parsing import DesignInputSpecification
from rfd3.model.inference_sampler import SampleDiffusionConfig
from rfd3.utils.inference import ensure_input_is_abspath
from rfd3.utils.io import (
    CIF_LIKE_EXTENSIONS,
    dump_metadata,
    dump_structures,
    dump_trajectories,
    extract_example_id_from_path,
    find_files_with_extension,
)

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass(kw_only=True)
class RFD3InferenceConfig:
    ckpt_path: str = "/projects/ml/aa_design/models/rfd3_latest_cleaned.ckpt"
    diffusion_batch_size: int = 16

    # RFD3 specific
    skip_existing: bool = False
    json_keys_subset: Optional[List[str]] = None
    skip_existing: bool = True
    specification: Optional[dict] = field(default_factory=dict)
    inference_sampler: SampleDiffusionConfig | dict = field(default_factory=dict)

    # Saving args
    cleanup_guideposts: bool = True
    cleanup_virtual_atoms: bool = True
    read_sequence_from_sequence_head: bool = True
    output_full_json: bool = True

    # Prefix to add to all output samples
    # Default: None      -> f'{jsonfilebasename}_{jsonkey}_{batch}_{model}'
    # Otherwise: string  -> f'{string}{jsonkey}_{batch}_{model}'
    # e.g. Empty string  -> f'{jsonkey}_{batch}_{model}'
    # e.g. Chunk string  -> f'{chunkprefix_}{jsonkey}_{batch}_{model}' (pipelines usage)
    global_prefix: Optional[str] = None
    dump_prediction_metadata_json: bool = True
    dump_trajectories: bool = False
    align_trajectory_structures: bool = False
    prevalidate_inputs: bool = True
    low_memory_mode: bool = (
        False  # False for standard mode, True for memory efficient tokenization mode
    )

    # Other:
    num_nodes: int = 1
    devices_per_node: int = 1
    print_config: bool = False
    seed: Optional[int] = None

    # For use as mapping:
    def keys(self):
        return self.__dataclass_fields__.keys()

    def __getitem__(self, key):
        return getattr(self, key)


class RFD3InferenceEngine(BaseInferenceEngine):
    """Inference engine for RFdiffusion3"""

    def __init__(
        self,
        *,
        # Default input handling args
        skip_existing: bool,
        json_keys_subset: None | List[str],
        prevalidate_inputs: bool,
        # Base inference engine args
        diffusion_batch_size: int,
        inference_sampler: dict,
        specification: dict | None,
        # Structure dumping arguments
        global_prefix: str | None,
        cleanup_guideposts: bool,
        cleanup_virtual_atoms: bool,
        read_sequence_from_sequence_head: bool,
        output_full_json: bool,
        dump_prediction_metadata_json: bool,
        dump_trajectories: bool,
        align_trajectory_structures: bool,
        low_memory_mode: bool,
        **kwargs,
    ):
        super().__init__(
            transform_overrides={"diffusion_batch_size": diffusion_batch_size},
            inference_sampler_overrides={**inference_sampler},
            trainer_overrides={
                "cleanup_guideposts": cleanup_guideposts,
                "cleanup_virtual_atoms": cleanup_virtual_atoms,
                "read_sequence_from_sequence_head": read_sequence_from_sequence_head,
                "output_full_json": output_full_json,
            },
            **kwargs,
        )
        # save
        self.specification_overrides = dict(specification or {})

        # Setup output directories and args
        self.global_prefix = global_prefix
        self.json_keys_subset = json_keys_subset
        self.prevalidate_inputs = prevalidate_inputs
        self.skip_existing = skip_existing

        # Saving / other args
        self.dump_prediction_metadata_json = dump_prediction_metadata_json
        self.dump_trajectories = dump_trajectories
        self.align_trajectory_structures = align_trajectory_structures
        if not cleanup_guideposts:
            ranked_logger.warning(
                "Guideposts will not be cleaned up. This is intended for debugging purposes."
            )
        if not cleanup_virtual_atoms:
            ranked_logger.warning(
                "Virtual atoms will not be cleaned up. Some tools like MPNN may run, but outputs will not be like native structures."
            )

        # Check which example ids already exist in the output directory
        if low_memory_mode:
            ranked_logger.info("Low memory mode enabled.")
            # HACK: Set attribute to the diffusion module
            os.environ["RFD3_LOW_MEMORY_MODE"] = "1"

    def run(
        self,
        *,
        inputs: str | PathLike | AtomArray | DesignInputSpecification,
        n_batches: int | None = None,
        out_dir: str | PathLike | None = None,
    ):
        self._set_out_dir(out_dir)
        inputs = self._canonicalize_inputs(inputs)
        design_specifications = self._multiply_specifications(
            inputs=inputs,
            n_batches=n_batches,
        )
        # init before
        self.initialize()
        outputs = self._run_multi(design_specifications)
        return outputs

    def _set_out_dir(self, out_dir: str | PathLike | None):
        out_dir = Path(out_dir) if out_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            ranked_logger.info(f"Outputs will be written to {out_dir.resolve()}.")
        self.out_dir = out_dir

    def _run_multi(self, specs):
        # ==============================================================================
        # Prepare pipeline and inference loader
        # ==============================================================================
        loader = assemble_distributed_inference_loader_from_json(
            # Passed directly to ContigJSONDataset
            # data={spec.example_id: spec for spec in spec.values()},
            data=specs,
            transform=self.pipeline,
            name="inference-dataset",
            cif_parser_args=None,
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
        # Evaluate, using `validation_step`
        # ==============================================================================
        outputs = {}
        for batch_idx, batch in enumerate(loader):
            pipeline_output = batch[0]
            output = self._model_forward(pipeline_output)

            if self.out_dir:
                self.save_batch_outputs(
                    out_dir=self.out_dir,
                    example_id=pipeline_output["example_id"],
                    network_output=output["network_output"],
                    prediction_metadata=output["prediction_metadata"],
                    predicted_atom_array_stack=output["predicted_atom_array_stack"],
                    pipeline_output=pipeline_output,
                )
            else:
                outputs[pipeline_output["example_id"]] = {
                    "network_output": output["network_output"],
                    "prediction_metadata": output["prediction_metadata"],
                    "predicted_atom_array_stack": output["predicted_atom_array_stack"],
                }
        return outputs

    def _model_forward(self, pipeline_output):
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
        return output

    ###############################################
    # Input merging
    ###############################################

    def _canonicalize_inputs(
        self, inputs
    ) -> Dict[str, dict | DesignInputSpecification]:
        is_json_like = (isinstance(inputs, (str, PathLike, Path))) or (
            isinstance(inputs, list)
            and all([isinstance(i, (str, PathLike, Path)) for i in inputs])
        )
        is_specification_like = isinstance(inputs, DesignInputSpecification) or (
            isinstance(inputs, list)
            and all([isinstance(i, DesignInputSpecification) for i in inputs])
        )
        is_atom_array_like = isinstance(inputs, (AtomArray, list)) or (
            isinstance(inputs, list) and all([isinstance(i, AtomArray) for i in inputs])
        )
        if inputs is None:
            # Create empty specification dictionary
            return {"": {**self.specification_overrides}}
        elif is_json_like:
            # List of file paths
            inputs = process_input(
                inputs,
                json_keys_subset=self.json_keys_subset,
                global_prefix=self.global_prefix,
                specification_overrides=self.specification_overrides,
                validate=self.prevalidate_inputs,
            )  # any -> Dict[Name: DesignInputSpecification]
        elif is_specification_like:
            # List of DesignInputSpecifications
            if isinstance(inputs, DesignInputSpecification):
                inputs = [inputs]
            inputs = {f"backbone_{i}": spec for i, spec in enumerate(inputs)}
        elif is_atom_array_like:
            raise NotImplementedError("AtomArray inputs not yet supported.")
        else:
            raise ValueError(
                f"Invalid input type: {type(inputs)}. Expected JSON/YAML file paths, AtomArray, or DesignInputSpecification.\nInput: {inputs}"
            )

        return inputs

    def _multiply_specifications(
        self, inputs: Dict[str, dict | DesignInputSpecification], n_batches=None
    ) -> Dict[str, Dict[str, Any]]:
        # Find existing example IDS in output directory
        if exists(self.out_dir):
            existing_example_ids = set(
                extract_example_id_from_path(path, CIF_LIKE_EXTENSIONS)
                for path in find_files_with_extension(self.out_dir, CIF_LIKE_EXTENSIONS)
            )
            ranked_logger.info(
                f"Found {len(existing_example_ids)} existing example IDs in the output directory."
            )

        # Based on inputs, construct the specifications to loop through
        design_specifications = {}
        for prefix, example_spec in inputs.items():
            # ... Create n_batches for example
            for batch_id in range((n_batches) if exists(n_batches) else 1):
                # ... Example ID
                example_id = f"{prefix}_{batch_id}" if exists(n_batches) else prefix

                if (
                    self.skip_existing
                    and exists(self.out_dir)
                    and example_id in existing_example_ids
                ):
                    ranked_logger.info(
                        f"Skipping design specification for example {example_id} | Already exists."
                    )
                    continue
                design_specifications[example_id] = example_spec
        return design_specifications

    def save_batch_outputs(
        self,
        *,
        out_dir,
        network_output,
        prediction_metadata,
        predicted_atom_array_stack,
        pipeline_output,
        example_id,
    ):
        out_dir = Path(out_dir)
        dump_structures(
            atom_arrays=predicted_atom_array_stack,
            base_path=out_dir / example_id,
            one_model_per_file=True,
            extra_fields=SAVED_CONDITIONING_ANNOTATIONS,
        )

        if self.dump_prediction_metadata_json:
            dump_metadata(
                prediction_metadata=prediction_metadata,
                base_path=out_dir / example_id,
                one_model_per_file=True,
            )

        if self.dump_trajectories:
            dump_trajectories(
                trajectory_list=network_output["X_denoised_L_traj"],
                atom_array=pipeline_output["atom_array"],
                base_path=out_dir / f"{example_id}_denoised",
                align_structures=self.align_trajectory_structures,
            )
            dump_trajectories(
                trajectory_list=network_output["X_noisy_L_traj"],
                atom_array=pipeline_output["atom_array"],
                base_path=out_dir / f"{example_id}_noisy",
                align_structures=self.align_trajectory_structures,
            )

        ranked_logger.info(
            f"Outputs for {example_id} written to {out_dir / example_id}."
        )


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
    global_prefix: str | None = None,
    specification_overrides: dict | None = None,
    validate: bool = True,
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
    specification_overrides = dict(specification_overrides or {})

    def merge_args(example_args: dict) -> dict:
        return merge_with(lambda x: x[-1], example_args, specification_overrides)

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
            all_specs["backbone"] = dict(specification_overrides)

    if validate:
        for prefix, example_spec in all_specs.items():
            ranked_logger.info(
                f"Prevalidating design specification for example: {prefix}"
            )
            DesignInputSpecification.safe_init(**example_spec)

    return all_specs
