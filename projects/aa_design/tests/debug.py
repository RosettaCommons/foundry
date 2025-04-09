#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../scripts/shebang/modelhub_exec.sh" "$0" "$@"'
# JBs debugging file, please create your own and go crazy!
import logging
import os
import time
from pathlib import Path

import hydra
import ipdb
import numpy as np
import pytest
import rootutils
import torch
import tree
from biotite.structure import AtomArray, AtomArrayStack
from cifutils import parse
from cifutils.utils.io_utils import load_any, to_cif_file
from datahub.transforms.center_random_augmentation import CenterRandomAugmentation
from datahub.utils.token import (
    get_token_count,
    get_token_starts,
    spread_token_wise,
)

from modelhub.utils.ddp import set_accelerator_based_on_availability
from modelhub.utils.logging import print_config_tree
from projects.aa_design.inference.input_parsing import (
    create_atom_array_from_design_specification,
)
from projects.aa_design.transforms.masks import Mask
from projects.aa_design.transforms.pipelines import (
    MotifCenterRandomAugmentation,
    build_atom14_base_pipeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Same as train.py
rootutils.setup_root(__file__ + '/../..', indicator=".project-root", pythonpath=True)
_config_path = os.path.join(os.environ.get("PROJECT_PATH", os.environ.get("PROJECT_ROOT", '../..')), "configs")
print(f"Config path: {_config_path}")
print(f"Project root: {os.environ.get('PROJECT_ROOT', '../..')}")


def filter_out(out: AtomArray):
    if len(out.shape) == 2:
        out = out[0]
    out = out[out.element != 'H']
    out = torch.nan_to_num(out)
    return out
    
outdir = Path('/home/jbutch/Projects/HT25/af3/modelhub_refactor/projects/aa_design/tests/outs')

pdb_id = '1qys'
def path_to_input(path, pdb_id=pdb_id):
    # input = parse(path, extra_fields=['is_motif_atom'])
    input = parse(path, add_missing_atoms=False, extra_fields="all")
    if "atom_array" not in input:
        assembly_ids = list(input["assemblies"].keys())
        input["atom_array"] = input["assemblies"][assembly_ids[0]][0]
    input["example_id"] = pdb_id
    input["pdb_id"] = pdb_id
    return input
# def path_to_input(path, pdb_id=pdb_id):
#     input = load_any(path, extra_fields=['is_motif_atom'])
#     return {
#         'atom_array': input[0],
#         'example_id': pdb_id,
#         'pdb_id': pdb_id,
#     }

input = path_to_input(f'/home/jbutch/{pdb_id}.cif')

def forward(example, trainer, model, is_inference=True):

    network_input = trainer._assemble_network_inputs(example)

    # Forward pass
    device='cuda:0'
    def _inmap(path, x):
        if hasattr(x, 'cpu') and path != ('f','msa_stack'):
            return x.to(device)
        else:
            return x
    network_input = tree.map_structure_with_path(_inmap, network_input)
    model.eval() if is_inference else model.train()
    network_output = model.forward(input=network_input, n_cycle=1, coord_atom_lvl_to_be_noised=example['coord_atom_lvl_to_be_noised'].to(device))
    return network_output

def o_to_x(output):

    xyz = output['coord_atom_lvl_to_be_noised']
    x = AtomArrayStack(xyz.shape[0], xyz.shape[1])
    idxs = np.argsort(output['t'].numpy())
    x.coord = xyz[idxs].numpy()
    x.coord = x.coord + output['noise'].numpy()[idxs]

    x.set_annotation("chain_id", ["A"]*xyz.shape[1])
    x.set_annotation('atom_name', [f'C{i}' for i in range(x.shape[-1])])
    x.set_annotation("res_id", output['feats']['atom_to_token_map'])
    x.set_annotation('element', ['C']*x.shape[-1])
    return x


# @pytest.mark.parametrize("is_inference", [
#     True, 
#     # False
# ])
# @pytest.mark.slow
# def test_prior_pipeline_bugs_af3(is_inference):

#     pipe = build_atom14_base_pipeline(
#         is_inference=is_inference,
#         return_atom_array=True,
#         sigma_data = 1.0,
#         crop_size=10,
#         crop_contiguous_probability = 1.0 - 0.001,
#         crop_spatial_probability = 0.001,
#     )
#     pipe.transforms = [t for t in pipe.transforms if not isinstance(t, CenterRandomAugmentation)]
#     t0 = time.time()
#     output = pipe(input)
#     print(f"Time taken to process example: {time.time() - t0}")
    
#     # Write outputs:
#     outdir = '/home/jbutch/Projects/HT25/af3/modelhub_refactor/projects/aa_design/tests/outs'

#     # Write stack:
#     # fout=f"{outdir}/{pdb_id}_atom_array-{is_inference}.cif"
#     # to_cif_file(output['atom_array'], fout, id=f'{pdb_id}_atomarray')
#     # print(f"Wrote to {fout}")

#     xyz = output['coord_atom_lvl_to_be_noised']
#     x = AtomArrayStack(xyz.shape[0], xyz.shape[1])
#     idxs = np.argsort(output['t'].numpy())
#     x.coord = xyz[idxs].numpy()
#     x.coord = x.coord + output['noise'].numpy()[idxs]

#     x.set_annotation("chain_id", ["A"]*xyz.shape[1])
#     x.set_annotation('atom_name', [f'C{i}' for i in range(x.shape[-1])])
#     x.set_annotation("res_id", output['feats']['atom_to_token_map'])
#     x.set_annotation('element', ['C']*x.shape[-1])

#     fout = f"{outdir}/{pdb_id}_processed_x-{is_inference}.cif"
#     to_cif_file(x, fout, id=f'{pdb_id}_modifiedx')
#     print(f"Wrote to {fout}")

#     x = AtomArray(xyz.shape[1])

#     print(output["ground_truth"]["coord_atom_lvl"].shape, output["ground_truth"]["coord_atom_lvl"], output["ground_truth"]["coord_atom_lvl"].sum())
#     print(output["ground_truth"]["mask_atom_lvl"].shape, output["ground_truth"]["mask_atom_lvl"], output["ground_truth"]["mask_atom_lvl"].sum())

#     ipdb.set_trace()

#     if is_inference: ...
#         # assert_no_nans(
#         #     output["ground_truth"]["coord_atom_lvl"],
#         #     msg="Nans!",
#         # )
#         # assert_no_nans(
#         #     output[""]["coord_atom_lvl"],
#         #     msg="Nans!",
#         # )

# @pytest.mark.parametrize("is_inference", [
#     True, 
#     # False
# ])
# @pytest.mark.slow
# def test_pipeline_ligands(is_inference):
#     pdb_id = 'muta'
#     input = path_to_input('/home/jbutch/Projects/HT25/af3/modelhub_refactor/projects/aa_design/benchmarks/enzymes/chorismate_mutase.pdb', pdb_id)

#     pipe = build_atom14_base_pipeline(
#         is_inference=is_inference,
#         return_atom_array=True,
#         sigma_data = 1.0,
#         crop_size=10,
#         crop_contiguous_probability = 1.0 - 0.001,
#         crop_spatial_probability = 0.001,
#         allowed_types=['is_protein', 'is_ligand'],
#     )
#     pipe.transforms = [t for t in pipe.transforms if not isinstance(t, CenterRandomAugmentation)]
    
#     t0 = time.time()
#     output = pipe(input)
#     print(f"Time taken to process example: {time.time() - t0}")
    
#     fout = f"{outdir}/{pdb_id}_processed_x-{is_inference}.cif"
#     x = o_to_x(output)
#     to_cif_file(x, fout, id=f'{pdb_id}_modifiedx')
#     print(f"Wrote to {fout}")
#     print(output["ground_truth"]["coord_atom_lvl"].shape, output["ground_truth"]["coord_atom_lvl"], output["ground_truth"]["coord_atom_lvl"].sum())
#     print(output["ground_truth"]["mask_atom_lvl"].shape, output["ground_truth"]["mask_atom_lvl"], output["ground_truth"]["mask_atom_lvl"].sum())
#     ipdb.set_trace()

@hydra.main(config_path=_config_path, config_name="train", version_base="1.3")
def test_conditional_forward(cfg):
    print_config_tree(cfg, resolve=False)
    is_inference = False

    train_masks = cfg.datasets.global_transform_args.train_masks
    print("Mask arguments", train_masks)

    pipe = build_atom14_base_pipeline(
        is_inference=is_inference,
        return_atom_array=True,
        sigma_data=1.0,
        crop_size=100,
        crop_contiguous_probability = 1.0 - 0.001,
        crop_spatial_probability = 0.001,
        allowed_types=['is_protein', 'is_ligand'],
        central_atom='CB',
        train_masks=train_masks,
        diffusion_batch_size=(2 if is_inference else 32)
    )
    pipe.transforms = [t for t in pipe.transforms if not isinstance(t, (CenterRandomAugmentation, MotifCenterRandomAugmentation))]
    # pdb_id='4i3f'
    # input = path_to_input('/home/jbutch/4i3f.cif', pdb_id)
    pdb_id = '1qys'
    input = path_to_input(f'/home/jbutch/{pdb_id}.cif', pdb_id)

    t0 = time.time()
    example = pipe(input)
    print(f"Time taken to process example: {time.time() - t0}")
    print(example.get('sampled_mask_name'))
    
    aa = example['atom_array']
    t_aa = aa[get_token_starts(aa)]
    import ipdb; ipdb.set_trace()
    
    x = o_to_x(example)
    to_cif_file(x, f"{outdir}/{pdb_id}_processed_conditional_x-{is_inference}.cif", id=f'{pdb_id}_modifiedx')
    to_cif_file(example['atom_array'], f"{outdir}/{pdb_id}_atom_array_conditional-{is_inference}.cif", id=f'{pdb_id}_atomarray')
    
    print("Preparing model")
    model, trainer = prep_forward(cfg)
    if is_inference:
        model.eval()
        trainer.state['model'].eval()
    network_output = forward(example, trainer, model, is_inference=is_inference)

@hydra.main(config_path=_config_path, config_name="train", version_base="1.3")
def main(cfg):
    # print_config_tree(cfg, resolve=False)
    model, trainer = prep_forward(cfg)

    # Create example pipe input
    is_inference = False
    pipe = build_atom14_base_pipeline(
        is_inference=is_inference,
        return_atom_array=True,
        sigma_data = 1.0,
        crop_size=100,
        crop_contiguous_probability = 1.0 - 0.001,
        crop_spatial_probability = 0.001,
    )
    t0 = time.time()
    example = pipe(input)
    print(f"Time taken to process example: {time.time() - t0}")
    
    # Forward pass
    network_output = forward(example, trainer, model)

def prep_forward(cfg):

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        loggers=None,
        callbacks=None,
        _convert_="partial",
        _recursive_=False,
    )
    set_accelerator_based_on_availability(cfg)
    trainer.initialize_or_update_trainer_state({"train_cfg": cfg})
    cfg.trainer.devices_per_node = 1
    cfg.trainer.num_nodes = 1
    try:
        trainer.fabric.launch()
    except Exception as e:
        print(f"Error: {e}")
        print('Switching port')
        os.environ['MASTER_PORT'] = str(1024 + np.random.randint(64512))
        trainer.fabric.launch()
    trainer.construct_model()
    model = trainer.state["model"]

    return model, trainer

def get_args(file=None, name=None):
    o = {}
    if file is None:
        o['input'] = '/home/jbutch/Projects/HT25/af3/modelhub_refactor/projects/aa_design/tests/run_M0024_1nzy_cond0_0-atomized-bb-False.pdb'
        o['contigs'] = '10-10,A20-21,5-5,A25-25,5-5,A30-30,10-10'
        o['contig_atoms'] = "{'A20':'CB,CG', 'A25':'OG1,CG2','A30':'CG,CD'}"
        o['length'] = "10-100"
    else:
        import json
        with open(file, 'r') as f:
            o = json.load(f)
            o = o[name] if name else o
        if 'input' in o:
            o['input'] = os.path.join('/home/jbutch/Projects/HT25/af3/modelhub_refactor/projects/aa_design/benchmarks/', o['input'])
    o['cif_parser_args'] = {'add_missing_atoms': False, 'extra_fields': Mask.required_annotations}
    return o

@hydra.main(config_path=_config_path, config_name="train", version_base="1.3")
def test_inference_prep(cfg):
    print_config_tree(cfg, resolve=False)
    is_inference = True
    pdb_id='custom'

    bench_file = '/home/jbutch/Projects/HT25/af3/modelhub_refactor/projects/aa_design/benchmarks/indexed.json' # 'lip-3'
    bench_file = '/home/jbutch/Projects/HT25/af3/modelhub_refactor/projects/aa_design/benchmarks/unindexed.json'
    args = get_args(bench_file, 'rsv0-1')
    print("USING ARGS:", args)

    # Create atom array before sending through pipeline
    atom_array = create_atom_array_from_design_specification(**args)

    # Save file 
    tmpfile = outdir / 'contig_atom_array.cif'
    to_cif_file(atom_array, tmpfile, id='contig_atom_array', extra_fields=Mask.required_annotations)

    # Reload
    input = path_to_input(tmpfile, pdb_id=pdb_id)
    
    # Send input through pipeline
    pipe = build_atom14_base_pipeline(
        is_inference=is_inference,
        return_atom_array=True,
        sigma_data=1.0,
        crop_size=256,
        crop_contiguous_probability = 1.0 - 0.001,
        crop_spatial_probability = 0.001,
        allowed_types=['is_protein', 'is_ligand'],
        central_atom='CB',
        train_masks=cfg.datasets.global_transform_args.train_masks,
    )
    pipe.transforms = [t for t in pipe.transforms if not isinstance(t, (CenterRandomAugmentation, MotifCenterRandomAugmentation))]

    t0 = time.time()
    example = pipe(input)
    print(f"Time taken to process example: {time.time() - t0}")
    print(example.get('sampled_mask_name'))
    
    x = o_to_x(example)
    to_cif_file(x, f"{outdir}/{pdb_id}-{is_inference}.cif", id=f'{pdb_id}_modifiedx')
    to_cif_file(example['atom_array'], f"{outdir}/{pdb_id}_pipe_out-{is_inference}.cif", id=f'{pdb_id}_atomarray')

    import ipdb; ipdb.set_trace()


# @hydra.main(config_path=_config_path, config_name="train", version_base="1.3")
# def test_indexed_forward(cfg):
    # return

if __name__ == "__main__":
    # test_prior_pipeline_bugs_af3(is_inference=True)
    # test_prior_pipeline_bugs_af3(is_inference=False)
    # test_prior_pipeline_bugs_af3(is_inference=False)
    # test_pipeline_ligands(is_inference=True)
    # test_conditional_forward()
    test_inference_prep()
    # pytest.main(["-v", __file__, "-m not very_slow"])
    # main()

    print("Finished main")