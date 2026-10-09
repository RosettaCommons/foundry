import json
import logging
import os
import time
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, cast

import torch
import yaml
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray, AtomArrayStack
from toolz import merge_with

from foundry.common import exists
from foundry.inference_engines.base import BaseInferenceEngine
from foundry.model.layers.attention import validate_attention_backend
from foundry.utils.alignment import weighted_rigid_align
from foundry.utils.ddp import RankedLogger
from rfd3.constants import SAVED_CONDITIONING_ANNOTATIONS
from rfd3.inference.batching import (
    DEFAULT_COMPILE_ATOM_BUCKETS,
    DEFAULT_COMPILE_SLOT_BUCKETS,
    DEFAULT_COMPILE_TOKEN_BUCKETS,
    padding_capacity,
    validate_buckets,
)
from rfd3.inference.datasets import (
    assemble_distributed_inference_loader_from_json,
)
from rfd3.inference.input_parsing import (
    DesignInputSpecification,
    ensure_input_is_abspath,
)
from rfd3.model.inference_sampler import SampleDiffusionConfig
from rfd3.utils.inference import (
    ensure_inference_sampler_matches_design_spec,
)
from rfd3.utils.io import (
    CIF_LIKE_EXTENSIONS,
    build_stack_from_atom_array_and_batched_coords,
    extract_example_id_from_path,
    find_files_with_extension,
)

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass(kw_only=True)
class RFD3InferenceConfig:
    ckpt_path: str | Path = (
        "rfd3"  # Defaults to foundry installation upon instantiation
    )
    diffusion_batch_size: int = 16
    inference_batch_size: int = 1
    inference_num_workers: int = 2
    atom_padding_multiple: int = 128
    token_padding_multiple: int = 16
    compile_atom_buckets: tuple[int, ...] = DEFAULT_COMPILE_ATOM_BUCKETS
    compile_token_buckets: tuple[int, ...] = DEFAULT_COMPILE_TOKEN_BUCKETS
    compile_slot_buckets: tuple[int, ...] = DEFAULT_COMPILE_SLOT_BUCKETS

    # RFD3 specific
    skip_existing: bool = True
    json_keys_subset: Optional[List[str]] = None
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
    compile_model: bool = True
    compile_cache_dir: str | None = field(
        default_factory=lambda: os.environ.get("RFD3_COMPILE_CACHE_DIR")
    )
    dense_attention_backend: str = "vanilla"
    inference_kernel_backend: str = "auto"
    inference_cuda_graph: bool = False

    # Other:
    num_nodes: int = 1
    devices_per_node: int = 1
    verbose: bool = False
    seed: Optional[int] = None

    # For use as mapping:
    def keys(self):
        return self.__dataclass_fields__.keys()

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class RFD3Output:
    atom_array: AtomArray
    metadata: dict
    example_id: str
    denoised_trajectory_stack: Optional[AtomArrayStack] = None
    noisy_trajectory_stack: Optional[AtomArrayStack] = None

    def dump(
        self,
        out_dir,
        verbose=True,
    ):
        base_path = os.path.join(out_dir, self.example_id)
        base_path = Path(base_path).absolute()
        to_cif_file(
            self.atom_array,
            base_path,
            file_type="cif.gz",
            include_entity_poly=False,
            extra_fields=SAVED_CONDITIONING_ANNOTATIONS,
        )
        if self.metadata:
            with open(f"{base_path}.json", "w") as f:
                json.dump(self.metadata, f, indent=4)

        # Trajectory saving
        prefix = str(base_path)[:-1].rstrip("_model_")
        suffix = str(base_path)[-1]
        if self.denoised_trajectory_stack is not None:
            to_cif_file(
                self.denoised_trajectory_stack,
                "_denoised_model_".join([prefix, suffix]),
                file_type="cif.gz",
                include_entity_poly=False,
            )

        if self.noisy_trajectory_stack is not None:
            to_cif_file(
                self.noisy_trajectory_stack,
                "_noisy_model_".join([prefix, suffix]),
                file_type="cif.gz",
                include_entity_poly=False,
            )

        if verbose:
            ranked_logger.info(f"Outputs for {self.example_id} written to {base_path}.")


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
        compile_model: bool = True,
        compile_cache_dir: str | None = None,
        dense_attention_backend: str = "vanilla",
        inference_kernel_backend: str = "auto",
        inference_cuda_graph: bool = False,
        inference_batch_size: int = 1,
        inference_num_workers: int = 2,
        atom_padding_multiple: int = 128,
        token_padding_multiple: int = 16,
        compile_atom_buckets=DEFAULT_COMPILE_ATOM_BUCKETS,
        compile_token_buckets=DEFAULT_COMPILE_TOKEN_BUCKETS,
        compile_slot_buckets=DEFAULT_COMPILE_SLOT_BUCKETS,
        **kwargs,
    ):
        if min(inference_batch_size, atom_padding_multiple, token_padding_multiple) < 1:
            raise ValueError("Batch size and padding multiples must be positive")
        self.inference_batch_size = inference_batch_size
        if inference_num_workers < 0:
            raise ValueError("inference_num_workers must be nonnegative")
        self.inference_num_workers = inference_num_workers
        self.atom_padding_multiple = atom_padding_multiple
        self.token_padding_multiple = token_padding_multiple
        self.compile_atom_buckets = validate_buckets(
            compile_atom_buckets, "compile_atom_buckets"
        )
        self.compile_token_buckets = validate_buckets(
            compile_token_buckets, "compile_token_buckets"
        )
        self.compile_slot_buckets = validate_buckets(
            compile_slot_buckets, "compile_slot_buckets"
        )
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
        self.inference_sampler_overrides = dict(inference_sampler or {})

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

        self.dense_attention_backend = validate_attention_backend(
            dense_attention_backend
        )
        if inference_kernel_backend not in {
            "auto", "torch", "triton", "triton-transition"
        }:
            raise ValueError(
                "inference_kernel_backend must be auto, torch, triton or triton-transition"
            )
        if (
            inference_kernel_backend in {"triton", "triton-transition"}
            or inference_cuda_graph
        ) and (
            low_memory_mode or os.environ.get("RFD3_LOW_MEMORY_MODE") == "1"
        ):
            raise ValueError(
                "Accelerated inference requires low_memory_mode=false and "
                "RFD3_LOW_MEMORY_MODE unset or 0"
            )
        if inference_cuda_graph and (compile_model or inference_batch_size != 1):
            raise ValueError(
                "inference_cuda_graph requires compile_model=false and "
                "inference_batch_size=1; padded kernels support torch.compile directly"
            )
        self.inference_kernel_backend = inference_kernel_backend
        self.inference_cuda_graph = inference_cuda_graph
        self.compile_model = compile_model
        self.compile_cache_dir = compile_cache_dir
        self.low_memory_mode = low_memory_mode
        self.compiled_ = False

    def initialize(self):
        first_initialize = not getattr(self, "initialized_", False)
        if self.compile_model and torch._dynamo.config.disable:
            raise ValueError(
                "compile_model=true requires torch.compile to be enabled; "
                "set TORCH_COMPILE_DISABLE=0 before importing PyTorch"
            )
        cfg = super().initialize()
        if first_initialize and self.inference_num_workers:
            from omegaconf import OmegaConf

            transform = next(iter(cfg.datasets.val.values())).dataset.transform
            self._inference_transform_config = OmegaConf.to_container(
                OmegaConf.merge(transform, self.transform_overrides), resolve=True
            )
        backend = getattr(self, "inference_kernel_backend", "torch")
        cuda_graph = getattr(self, "inference_cuda_graph", False)
        if backend == "auto":
            from rfd3.inference.runtime import select_kernel_backend

            backend = select_kernel_backend(
                device=next(self.trainer.state["model"].parameters()).device,
                compile_model=self.compile_model,
                inference_batch_size=self.inference_batch_size,
                low_memory_mode=self.low_memory_mode,
            )
        self.resolved_kernel_backend = backend
        if backend in {"triton", "triton-transition"} or cuda_graph:
            if next(self.trainer.state["model"].parameters()).device.type != "cuda":
                raise ValueError("Accelerated inference requires CUDA")
        if backend in {"triton", "triton-transition"}:
            if not self.compile_model:
                from rfd3.inference.runtime import configure_compile_cache

                configure_compile_cache(self.compile_cache_dir)
            import triton  # noqa: F401 — fail early if the optional runtime is absent

        from rfd3.model.layers.layer_utils import Transition

        for module in self.trainer.state["model"].modules():
            module_backend = backend
            if backend == "triton-transition":
                module_backend = "triton" if isinstance(module, Transition) else "torch"
            module.inference_kernel_backend = module_backend
            module.inference_cuda_graph = cuda_graph
            if hasattr(module, "dense_attention_backend"):
                module.dense_attention_backend = self.dense_attention_backend
        if self.compile_model and not self.compiled_:
            self._compile_diffusion_submodules()
            self.compiled_ = True
        if first_initialize:
            ranked_logger.info(
                f"RFD3 acceleration: policy={self.inference_kernel_backend}, "
                f"kernels={backend}, compile_model={self.compile_model}, "
                f"token_and_atom_cuda_graph={cuda_graph}, "
                f"low_memory_mode={self.low_memory_mode}"
            )
        return cfg

    def _compile_diffusion_submodules(self) -> None:
        """Compile the padded denoiser, including at B=1; fallbacks stay eager."""
        diffusion_module = self._rfd3_net().diffusion_module
        from rfd3.inference.runtime import (
            configure_compile_cache,
            dynamic_diffusion_batch,
        )

        options = configure_compile_cache(getattr(self, "compile_cache_dir", None))
        ranked_logger.info(
            "Inductor disk cache: "
            f"{os.environ.get('TORCHINDUCTOR_CACHE_DIR', 'PyTorch default')}"
        )
        device_type = next(diffusion_module.parameters()).device.type
        if device_type == "cuda":
            # Unbounded fan-out around sorted neighbors produced an 18-output
            # Triton kernel with pathological compilation on the full model.
            options["max_fusion_unique_io_buffers"] = 16
        if device_type == "mps":
            # Metal limits kernel buffer arguments. Full checkpoint graphs can
            # otherwise fuse >31 inputs, even when small model tests compile.
            # Leave one argument for Inductor's error buffer; scope to this compile.
            options["max_fusion_unique_io_buffers"] = 30
        diffusion_module.forward_batched = torch.compile(
            diffusion_module.forward_batched,
            dynamic=False,
            fullgraph=True,
            options=options,
        )
        if device_type == "cuda":
            diffusion_module.forward_batched = dynamic_diffusion_batch(
                diffusion_module.forward_batched
            )
        ranked_logger.info(
            "Compiling the padded denoiser with atom/token/slot buckets "
            f"{self.compile_atom_buckets}/{self.compile_token_buckets}/{self.compile_slot_buckets}; "
            "legacy fallbacks remain eager."
        )

    # The base `run` is positional (`inputs, *_`); this engine deliberately exposes a
    # richer keyword-only API, so the override is intentionally LSP-incompatible.
    def run(  # type: ignore[override]
        self,
        *,
        inputs: str | PathLike | AtomArray | DesignInputSpecification,
        n_batches: int | None = None,
        out_dir: str | PathLike | None = None,
    ) -> dict[str, list[RFD3Output]] | None:
        self._set_out_dir(out_dir)
        inputs = self._canonicalize_inputs(inputs)
        design_specifications = self._multiply_specifications(
            inputs=inputs,
            n_batches=n_batches,
        )
        if len(design_specifications) == 0:
            ranked_logger.info("No design specifications to run. Skipping.")
            return None
        ensure_inference_sampler_matches_design_spec(
            design_specifications, self.inference_sampler_overrides
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

    def _run_multi(self, specs) -> None | Dict[str, List[RFD3Output]]:
        if self.inference_batch_size > 1 or self.compile_model:
            return self._run_multi_batched(specs)
        # ==============================================================================
        # Prepare pipeline and inference loader
        # ==============================================================================
        loader = assemble_distributed_inference_loader_from_json(
            # Passed directly to ContigJSONDataset
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
            example_id = pipeline_output["example_id"]

            # Run model
            output_list = self._model_forward(pipeline_output)
            if self.out_dir:
                for output in output_list:
                    output.dump(out_dir=self.out_dir)
            else:
                outputs[example_id] = output_list
        return outputs

    def _model_forward(self, pipeline_output) -> List[RFD3Output]:
        # Wraps around the trainer validation step to create atom arrays for saving.
        t0 = time.time()
        with torch.no_grad():
            pipeline_output = self.trainer.fabric.to_device(pipeline_output)
            output_val = self.trainer.validation_step(
                batch=pipeline_output,
                batch_idx=0,
                compute_metrics=False,
            )
        t_end = time.time()
        ranked_logger.info(f"Finished inference batch in {t_end - t0:.2f} seconds.")
        return self._format_model_output(pipeline_output, output_val)

    def _format_model_output(self, pipeline_output, output_val) -> List[RFD3Output]:
        # Add additional information to prediction metadata
        if self.dump_trajectories:
            X_noisy_L_traj = torch.stack(
                output_val["network_output"]["X_noisy_L_traj"]
            ).transpose(0, 1)  # [D, N_steps, L, 3]
            X_denoised_L_traj = torch.stack(
                output_val["network_output"]["X_denoised_L_traj"]
            ).transpose(0, 1)  # [D, N_steps, L, 3]

        outputs = []
        for idx in range(len(output_val["predicted_atom_array_stack"])):
            if self.dump_prediction_metadata_json:
                ckpt = Path(self.ckpt_path)
                if ckpt.is_symlink():
                    ckpt = ckpt.resolve(strict=True)  # follow symlink to target
                output_val["prediction_metadata"][idx]["ckpt_path"] = str(ckpt)
                output_val["prediction_metadata"][idx]["seed"] = self.seed

            # Append to outputs
            if self.dump_trajectories:
                X_denoised_L_traj_i = _reshape_trajectory(
                    X_denoised_L_traj[idx], self.align_trajectory_structures
                )
                X_noisy_L_traj_i = _reshape_trajectory(X_noisy_L_traj[idx], False)
                denoised_trajectory_stack = (
                    build_stack_from_atom_array_and_batched_coords(
                        X_denoised_L_traj_i, pipeline_output["atom_array"]
                    )
                )
                noisy_trajectory_stack = build_stack_from_atom_array_and_batched_coords(
                    X_noisy_L_traj_i, pipeline_output["atom_array"]
                )
            else:
                denoised_trajectory_stack = None
                noisy_trajectory_stack = None

            outputs.append(
                RFD3Output(
                    example_id=f"{pipeline_output['example_id']}_model_{idx}",
                    atom_array=output_val["predicted_atom_array_stack"][idx],
                    metadata=output_val["prediction_metadata"][idx]
                    if self.dump_prediction_metadata_json
                    else {},
                    denoised_trajectory_stack=denoised_trajectory_stack,
                    noisy_trajectory_stack=noisy_trajectory_stack,
                )
            )

        return outputs

    def _rfd3_net(self):
        net = self.trainer.state["model"]
        for _ in range(8):
            if hasattr(net, "diffusion_module"):
                return net
            for attr in ("_forward_module", "module", "shadow", "model"):
                if hasattr(net, attr):
                    net = getattr(net, attr)
                    break
            else:
                break
        raise ValueError("Cannot locate the RFD3 network for padded inference")

    def _batch_fallback_reason(self, example):
        net = self._rfd3_net()
        if net.token_initializer.atom_transformer is not None:
            return "static random atom attention uses the legacy initializer"
        if len(example["feats"]["asym_id"].unique()) > 3:
            return "more than three chains use the legacy randomized neighbor policy"
        return None

    def _compile_buckets(self):
        if not self.compile_model:
            return {}
        return dict(
            atom_buckets=self.compile_atom_buckets,
            token_buckets=self.compile_token_buckets,
            slot_buckets=self.compile_slot_buckets,
        )

    def _batch_capacities(self, examples):
        buckets = self._compile_buckets()
        atoms = max(len(e["feats"]["atom_to_token_map"]) for e in examples)
        tokens = max(len(e["feats"]["restype"]) for e in examples)
        capacities = dict(
            atom_capacity=padding_capacity(
                atoms,
                self.atom_padding_multiple,
                buckets.get("atom_buckets"),
                axis="atom length",
            ),
            token_capacity=padding_capacity(
                tokens,
                self.token_padding_multiple,
                buckets.get("token_buckets"),
                axis="token length",
            ),
            batch_capacity=self.inference_batch_size,
        )
        if self.compile_model:
            slots = max(
                int(torch.bincount(e["feats"]["atom_to_token_map"].long()).max())
                for e in examples
            )
            capacities["slot_capacity"] = padding_capacity(
                slots, buckets=buckets["slot_buckets"], axis="atoms per token"
            )
        return capacities

    def _model_forward_batch(self, examples) -> Dict[str, List[RFD3Output]]:
        from rfd3.inference.batching import (
            cfg_reference_examples,
            collate_examples,
            unpad_output,
        )
        from rfd3.model.batched_sampler import ExampleRandomness

        net = self._rfd3_net()
        capacities = self._batch_capacities(examples)
        batch = collate_examples(examples, **capacities)
        network_input = {"f": batch["f"]}
        if net.use_classifier_free_guidance:
            refs = cfg_reference_examples(examples, net.cfg_features)
            # Share A capacity with the conditional batch to limit graph variants.
            ref_batch = collate_examples(
                refs,
                **{**capacities, "slot_capacity": batch["f"]["atom_slots"].shape[-1]},
            )
            network_input["f_ref"] = ref_batch["f"]
        network_input = self.trainer.fabric.to_device(network_input)
        network_input["capture_trajectories"] = self.dump_trajectories
        seed = (
            self.seed
            if self.seed is not None
            else int(torch.randint(0, 2**62, ()).item())
        )
        network_input["noise_source"] = ExampleRandomness(examples, seed)
        with torch.no_grad():
            output = self.trainer.state["model"].forward(
                input=network_input,
                coord_atom_lvl_to_be_noised=self.trainer.fabric.to_device(
                    batch["coords"]
                ),
            )
            # Structure construction needs CPU coordinates. Transfer the batch
            # once, then validate each real example on CPU instead of reading
            # a CUDA boolean and copying coordinates separately for every row.
            output["X_L"] = output["X_L"].cpu()
            results = {}
            for row, example in enumerate(examples):
                single = unpad_output(output, row, example)
                if not torch.isfinite(single["X_L"]).all():
                    raise ValueError(
                        f"Nonfinite coordinates for {example['example_id']}"
                    )
                structures, metadata = self.trainer._build_predicted_atom_array_stack(
                    single, example
                )
                results[example["example_id"]] = self._format_model_output(
                    example,
                    dict(
                        network_output=single,
                        predicted_atom_array_stack=structures,
                        prediction_metadata=metadata,
                    ),
                )
        return results

    def _run_multi_batched(self, specs):
        from rfd3.inference.batching import iter_compatible_batches
        from rfd3.inference.datasets import ContigJsonDataset
        from rfd3.inference.loading import iter_inference_examples

        dataset = ContigJsonDataset(
            data=specs,
            transform=self.pipeline,
            name="inference-dataset",
            cif_parser_args=None,
            subset_to_keys=None,
            eval_every_n=1,
        )
        fabric = self.trainer.fabric

        examples = iter_inference_examples(
            dataset,
            seed=self.seed,
            rank=fabric.global_rank,
            world_size=fabric.world_size,
            num_workers=getattr(self, "inference_num_workers", 0),
            transform_config=getattr(self, "_inference_transform_config", None),
            batch_size=self.inference_batch_size,
            pin_memory=getattr(getattr(fabric, "device", None), "type", None) == "cuda",
        )

        outputs = {}

        # Fallback rows are emitted separately before feature-schema collation.
        def supported():
            for ex in examples:
                reason = self._batch_fallback_reason(ex)
                if reason:
                    ranked_logger.warning(
                        f"Legacy inference for {ex['example_id']}: {reason}."
                    )
                    save({ex["example_id"]: self._model_forward(ex)})
                else:
                    yield ex

        def save(result):
            if self.out_dir:
                for values in result.values():
                    for output in values:
                        output.dump(out_dir=self.out_dir)
            else:
                outputs.update(result)

        for group in iter_compatible_batches(
            supported(),
            self.inference_batch_size,
            self.atom_padding_multiple,
            self.token_padding_multiple,
            **self._compile_buckets(),
        ):
            save(self._model_forward_batch(group))
        return {
            dataset.idx_to_id(i): outputs[dataset.idx_to_id(i)]
            for i in range(len(dataset))
            if dataset.idx_to_id(i) in outputs
        }

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
    ) -> Dict[str, dict | DesignInputSpecification]:
        # Find existing example IDS in output directory
        if self.out_dir is not None:
            existing_example_ids_ = set(
                extract_example_id_from_path(path, CIF_LIKE_EXTENSIONS)
                for path in find_files_with_extension(self.out_dir, CIF_LIKE_EXTENSIONS)
            )
            existing_example_ids = set(
                [
                    "_model_".join(eid.split("_model_")[:-1])
                    for eid in existing_example_ids_
                ]
            )
            ranked_logger.info(
                f"Found {len(existing_example_ids)} existing example IDs in the output directory ({len(existing_example_ids_)} total)."
            )

        # Based on inputs, construct the specifications to loop through
        design_specifications = {}
        for prefix, example_spec in inputs.items():
            # Record task name in the specification
            if isinstance(example_spec, DesignInputSpecification):
                example_spec.extra = example_spec.extra or {}
                example_spec.extra["task_name"] = prefix
            else:
                if "extra" not in example_spec:
                    example_spec["extra"] = {}
                example_spec["extra"]["task_name"] = prefix

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
        return [None]
    if isinstance(inputs, str):
        # str.split yields list[str]; widen to the union return type (list is invariant).
        return cast(list[str | None], inputs.split(","))
    if not isinstance(inputs, list):
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

    # ... Determine prefix of sample to create
    all_specs = {}
    for input in inputs:
        if input is not None and (input.endswith(".json") or input.endswith(".yaml")):
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

        elif input is not None:
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


def _reshape_trajectory(traj, align_structures: bool):
    traj = [traj[i] for i in range(len(traj))]  # make list of arrays
    max_frames = 100
    if len(traj) > max_frames:
        selected_indices = torch.linspace(0, len(traj) - 1, max_frames).long().tolist()
        traj = [traj[i] for i in selected_indices]
    if align_structures:
        # ... align the trajectories on the last prediction
        for step in range(len(traj) - 1):
            traj[step] = weighted_rigid_align(
                X_L=traj[-1][None],
                X_gt_L=traj[step][None],
            ).squeeze(0)
    traj = traj[::-1]  # reverse to go from noised -> denoised

    traj = torch.stack(traj).cpu().numpy()
    return traj
