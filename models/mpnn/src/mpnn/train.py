#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/mpnn_exec.sh" "$0" "$@"'

import sys
from pathlib import Path

import pandas as pd
import torch
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.ml.datasets.pandas_dataset import PandasDataset, StructuralDatasetWrapper
from atomworks.ml.datasets.parsers.default_metadata_row_parsers import GenericDFParser
from atomworks.ml.samplers import (
    DistributedMixedSampler,
    calculate_weights_for_pdb_dataset_df,
)
from omegaconf import DictConfig
from torch.utils.data import DataLoader, WeightedRandomSampler

from foundry.callbacks.metrics_logging import StoreValidationMetricsInDFCallback
from foundry.utils.datasets import wrap_dataset_and_sampler_with_fallbacks
from mpnn.collate.feature_collator import TokenBudgetAwareFeatureCollator
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline
from mpnn.samplers.samplers import PaddedTokenBudgetBatchSampler
from mpnn.trainers.mpnn import MPNNTrainer

model_type = sys.argv[1]

if model_type == "protein_mpnn":
    batch_size = 10000
    train_date_cutoff = "2021-08-02"
    clip_grad_max_norm = None
    train_structure_noise_default = 0.2
elif model_type == "ligand_mpnn":
    batch_size = 6000
    train_date_cutoff = "2022-12-16"
    clip_grad_max_norm = 1.0
    train_structure_noise_default = 0.1
else:
    raise ValueError(f"Unknown model_type: {model_type}")


def create_noam_scheduler(optimizer, d_model, warmup_steps=4000, factor=2):
    """
    Create a NoamOpt-style scheduler using standard PyTorch components.

    Args:
        optimizer: PyTorch optimizer
        d_model: Model dimension (for scaling)
        warmup_steps: Number of warmup steps
        factor: Scaling factor

    Returns:
        LambdaLR scheduler that implements NoamOpt schedule
    """

    def noam_lambda(step):
        # NoamOpt formula: factor * (d_model ** (-0.5)) * min(step ** (-0.5), step * warmup ** (-1.5))
        base_lr = factor * (d_model ** (-0.5))
        if step == 0:
            return 0.0  # Start with zero learning rate

        # Calculate the schedule component
        schedule = min(step ** (-0.5), step * warmup_steps ** (-1.5))
        return base_lr * schedule

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)


def get_num_tokens(df, idx):
    """
    Extract the number of non-atomized tokens for a given index.

    Args:
        df: DataFrame containing the dataset
        idx: Index to extract token count from

    Returns:
        Number of non-atomized tokens for the sample at idx
    """
    if isinstance(idx, (list, tuple)):
        # If idx is a list/tuple, return the first element's token count
        idx = idx[0]
    return df.iloc[idx]["n_non_atomized_tokens"]


# Common filters for MPNN datasets
MPNN_FILTERS = [
    "resolution < 3.5 and ~method.str.contains('NMR')",
    "n_non_atomized_tokens >= 30",
    "cluster.notnull() and cluster != 'nan'",
    "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
    f"n_non_atomized_tokens <= {batch_size}",
    "n_prot == 1",
]

MPNN_TRAIN_FILTERS = [
    f"deposition_date < '{train_date_cutoff}'",
] + MPNN_FILTERS

# Data loading setup
train_path = "/projects/ml/datahub/dfs/mpnn/splits/2025_07_13/pn_units_df_train.parquet"
val_path = "/projects/ml/datahub/dfs/mpnn/splits/2025_07_13/pn_units_df_val.parquet"

# Load datasets
train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)

# Create different pipelines for training and inference
train_pipeline = build_mpnn_transform_pipeline(
    model_type=model_type,
    is_inference=False,
    minimal_return=True,
    train_structure_noise_default=train_structure_noise_default,
)
inference_pipeline = build_mpnn_transform_pipeline(
    model_type=model_type, is_inference=True, minimal_return=True
)

# Create train dataset with fallback
train_structural_dataset = StructuralDatasetWrapper(
    dataset=PandasDataset(
        data=train_df,
        id_column="example_id",
        name="pn_units_df_train",
        filters=MPNN_TRAIN_FILTERS,
    ),
    dataset_parser=GenericDFParser(
        example_id_colname="example_id",
        path_colname="path",
        assembly_id_colname="assembly_id",
    ),
    transform=train_pipeline,
    cif_parser_args={
        **STANDARD_PARSER_ARGS,
        "load_from_cache": True,
        "save_to_cache": True,
        "cache_dir": "/net/tukwila/akubaney/cifutils/cache",
    },
)

train_fallback_dataset = StructuralDatasetWrapper(
    dataset=PandasDataset(
        data=train_df,
        id_column="example_id",
        name="pn_units_df_train",
        filters=MPNN_TRAIN_FILTERS,
    ),
    dataset_parser=GenericDFParser(
        example_id_colname="example_id",
        path_colname="path",
        assembly_id_colname="assembly_id",
    ),
    transform=train_pipeline,
    cif_parser_args={
        **STANDARD_PARSER_ARGS,
        "load_from_cache": True,
        "save_to_cache": True,
        "cache_dir": "/net/tukwila/akubaney/cifutils/cache",
    },
)

# Calculate weights for train dataset
train_weights = calculate_weights_for_pdb_dataset_df(
    dataset_df=train_structural_dataset.data,
    beta=1.0,  # For chains
    alphas={"a_prot": 1.0, "a_nuc": 0, "a_ligand": 0, "a_loi": 0},
)

# Create train sampler with fallback
train_sampler = DistributedMixedSampler(
    datasets_info=[
        {
            "sampler": WeightedRandomSampler(
                train_weights, len(train_structural_dataset)
            ),
            "dataset": train_structural_dataset,
            "probability": 1.0,
        }
    ],
    num_replicas=1,
    rank=0,
    n_examples_per_epoch=20000,
)

train_fallback_sampler = WeightedRandomSampler(
    train_weights, len(train_structural_dataset)
)

train_dataset_with_fallback, train_sampler_with_fallback = (
    wrap_dataset_and_sampler_with_fallbacks(
        dataset_to_be_wrapped=train_structural_dataset,
        sampler_to_be_wrapped=train_sampler,
        dataset_to_fallback_to=train_fallback_dataset,
        sampler_to_fallback_to=train_fallback_sampler,
        n_fallback_retries=5,
    )
)

batched_train_sampler = PaddedTokenBudgetBatchSampler(
    sampler=train_sampler_with_fallback,
    get_num_tokens=lambda idx: get_num_tokens(train_structural_dataset.data, idx),
    max_tokens_with_padding=batch_size,
    shuffle_batches=True,
)

# Create val dataset with fallback
val_structural_dataset = StructuralDatasetWrapper(
    dataset=PandasDataset(
        data=val_df,
        id_column="example_id",
        name="pn_units_df_val",
        filters=MPNN_FILTERS,
    ),
    dataset_parser=GenericDFParser(
        example_id_colname="example_id",
        path_colname="path",
        assembly_id_colname="assembly_id",
    ),
    transform=inference_pipeline,
    cif_parser_args={
        **STANDARD_PARSER_ARGS,
        "load_from_cache": True,
        "save_to_cache": True,
        "cache_dir": "/net/tukwila/akubaney/cifutils/cache",
    },
)

# Create val sampler with fallback
val_weights = calculate_weights_for_pdb_dataset_df(
    dataset_df=val_structural_dataset.data,
    beta=1.0,  # For chains
    alphas={"a_prot": 1.0, "a_nuc": 0, "a_ligand": 0, "a_loi": 0},
)

val_sampler = DistributedMixedSampler(
    datasets_info=[
        {
            "sampler": WeightedRandomSampler(val_weights, len(val_structural_dataset)),
            "dataset": val_structural_dataset,
            "probability": 1.0,
        }
    ],
    num_replicas=1,
    rank=0,
    n_examples_per_epoch=100,
)

# Create collator
collator = TokenBudgetAwareFeatureCollator(max_tokens_with_padding=batch_size)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset_with_fallback,
    batch_sampler=batched_train_sampler,
    num_workers=12,
    collate_fn=collator,
)

val_loaders = {
    "test_val": DataLoader(
        val_structural_dataset,
        sampler=val_sampler,
        num_workers=12,
        collate_fn=collator,
    )
}

# Create output directory for logs and checkpoints
output_dir = Path(f"./mpnn_output_{model_type}")
output_dir.mkdir(exist_ok=True)

# Create CSV logging callback
csv_callback = StoreValidationMetricsInDFCallback(
    save_dir=output_dir / "val_metrics", metrics_to_save="all"
)

# Create trainer with minimal configuration for testing
trainer = MPNNTrainer(
    model_type=model_type,
    accelerator="gpu",
    devices_per_node=1,
    max_epochs=500,
    output_dir=output_dir,
    callbacks=[csv_callback],
    precision="bf16-mixed",
    clip_grad_max_norm=clip_grad_max_norm,
)

# Create minimal train_cfg for optimizer and scheduler construction
train_cfg = DictConfig(
    {
        "model": {
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "lr": 1.0,  # This will be overridden by the NoamOpt scheduler
                "betas": [0.9, 0.98],  # NoamOpt uses (0.9, 0.98)
                "eps": 1e-9,  # NoamOpt uses 1e-9
                "weight_decay": 0.0,
            },
            "lr_scheduler": {
                "_target_": "__main__.create_noam_scheduler",
                "d_model": 128,  # Adjust based on your model's hidden dimension
                "warmup_steps": 4000,
                "factor": 2,
            },
        }
    }
)

# Initialize trainer state with train_cfg
trainer.initialize_or_update_trainer_state({"train_cfg": train_cfg})

# Launch Fabric (this sets up the distributed environment)
trainer.fabric.launch()

# Construct model
trainer.construct_model()

# Construct optimizer and scheduler
trainer.construct_optimizer()
trainer.construct_scheduler()


class CkptConfig:
    def __init__(self, path, weight_loading_config=None, reset_optimizer=False):
        self.path = path
        self.weight_loading_config = weight_loading_config
        self.reset_optimizer = reset_optimizer


ckpt_dir = output_dir / "ckpt"
if ckpt_dir.exists():
    ckpt_config = CkptConfig(
        path=ckpt_dir, weight_loading_config=None, reset_optimizer=False
    )
else:
    ckpt_config = None

# Run the full training using fit method
print("Starting training...")
trainer.fit(train_loader=train_loader, val_loaders=val_loaders, ckpt_config=ckpt_config)
print("Training completed!")
