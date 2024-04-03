import sys
sys.path.append('../')

import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from tqdm import tqdm
from rf2aa.data.data_loader import default_dataloader_params, loader_tf_complex, loader_distil_tf, get_train_valid_set, DistilledDataset, loader_pdb, loader_complex, loader_na_complex, loader_fb, loader_dna_rna, loader_sm_compl_assembly_single, loader_sm_compl_assembly, loader_sm, loader_atomize_pdb, loader_atomize_complex
from rf2aa.chemical import load_pdb_ideal_sdf_strings
import random
import numpy as np
import torch
import pickle

class OrderedSampler(torch.utils.data.sampler.Sampler):
    """
    Custom sampler that samples specific indices from a dataset.
    """
    def __init__(self, indices):
        self.indices = indices
        
    def __iter__(self):
        return iter(self.indices)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process crop size and MSA limit.')
    parser.add_argument('--crop-size', type=int, help='Crop size for the data loader', default=256)
    parser.add_argument('--msa-limit', type=int, help='MSA limit for the data loader', default=None)
    parser.add_argument('--output', type=str, help='Output file path for the lengths', default='sample_lengths.pt')
    parser.add_argument('--num-samples', type=int, help='Number of samples for the loop', default=None)
    parser.add_argument('--num-workers', type=int, help='Number of DataLoader workers', default=4)
    return parser.parse_args()

def seed_everything():
    seed = 42 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_dataset(crop_size, msa_limit):
    """
    Build a DistilledDataset with the given crop size and MSA limit overwriting the default parameters.
    """
    
    # Load default parameters
    loader_param = default_dataloader_params
    
    # Override default parameters
    loader_param['CROP'] = crop_size or loader_param.get('CROP')
    loader_param['MSA_LIMIT'] = msa_limit or loader_param.get('MSA_LIMIT')
    
    # Build dataset
    (
        train_ID_dict,
        valid_ID_dict,
        weights_dict,
        train_dict,
        valid_dict,
        homo,
        chid2hash,
        chid2taxid,
        chid2smpartners,
    ) = get_train_valid_set(loader_param)

    # define atomize_pdb train/valid sets, which use the same examples as pdb set
    train_ID_dict['atomize_pdb'] = train_ID_dict['pdb']
    valid_ID_dict['atomize_pdb'] = valid_ID_dict['pdb']
    weights_dict['atomize_pdb'] = weights_dict['pdb']
    train_dict['atomize_pdb'] = train_dict['pdb']
    valid_dict['atomize_pdb'] = valid_dict['pdb']

    # define atomize_pdb train/valid sets, which use the same examples as pdb set
    train_ID_dict['atomize_complex'] = train_ID_dict['compl']
    valid_ID_dict['atomize_complex'] = valid_ID_dict['compl']
    weights_dict['atomize_complex'] = weights_dict['compl']
    train_dict['atomize_complex'] = train_dict['compl']
    valid_dict['atomize_complex'] = valid_dict['compl']

    # Assign loaders to each dataset
    loader_dict = dict(
            pdb = loader_pdb,
            peptide = loader_pdb,
            compl = loader_complex,
            neg_compl = loader_complex,
            na_compl = loader_na_complex,
            neg_na_compl = loader_na_complex,
            distil_tf = loader_distil_tf,
            tf = loader_tf_complex,
            neg_tf = loader_tf_complex,
            fb = loader_fb,
            rna = loader_dna_rna,
            dna = loader_dna_rna,
            sm_compl = loader_sm_compl_assembly_single,
            metal_compl = loader_sm_compl_assembly_single,
            sm_compl_multi = loader_sm_compl_assembly_single,
            sm_compl_covale = loader_sm_compl_assembly_single,
            sm_compl_asmb = loader_sm_compl_assembly,
            sm = loader_sm,
            atomize_pdb = loader_atomize_pdb,
            atomize_complex = loader_atomize_complex,
        )
    
    # Get ligand dictionary. This is used for loading negative examples.
    ligand_dictionary = load_pdb_ideal_sdf_strings(return_only_sdf_strings=True)

    # Build dataset
    train_set = DistilledDataset(
        train_ID_dict,
        train_dict,
        loader_dict,
        homo,
        chid2hash,
        chid2taxid,
        chid2smpartners,
        loader_param,
        native_NA_frac=0.25,
        ligand_dictionary=ligand_dictionary
    )
    
    return train_set

def main(crop_size, msa_limit, output_file, num_samples, num_workers):
    print(f"Building dataset with CROP_SIZE={crop_size} and MSA_LIMIT={msa_limit}...")
    # Setup the dataset
    train_set = build_dataset(crop_size, msa_limit)
    
    # Datasets where we will assume `LEN_EXIST` returns the appropriate length
    fixed_length_datasets = ['pdb', 'fb']
    
    # Datasets where we pass the sample through the DataLoader to measure the length
    variable_length_datasets = ["compl", "neg_compl", "na_compl", "neg_na_compl", "distil_tf","tf","neg_tf","rna","dna", "sm_compl", "metal_compl", "sm_compl_multi", "sm_compl_covale", "sm_compl_asmb", "sm", "atomize_pdb", "atomize_complex"]

    # Assert that the monomer and variable dataset lists do not overlap
    assert len(set(fixed_length_datasets) & set(variable_length_datasets)) == 0, "Monomer and variable datasets should not overlap"
    
    # Create a tensor of zeros of the same length as the dataset
    lengths = torch.zeros(len(train_set))
    
    offset = 0
    indices_to_process = []
    for key, index_list in train_set.index_dict.items():
        # Add offset to every value in list
        adjusted_index_list = [x + offset for x in index_list]
        
        # Either add index to list to process later or calculate length directly
        if key in variable_length_datasets:
            indices_to_process.extend(adjusted_index_list)
        elif key in fixed_length_datasets:
            # Calculate lengths and remove duplicates
            if key in ['pdb', 'fb']:
                train_set.dataset_dict[key]['LENGTH'] = train_set.dataset_dict[key]["SEQUENCE"].apply(len)
            else:
                train_set.dataset_dict[key]['LENGTH'] = train_set.dataset_dict[key]["LEN_EXIST"]
            fixed_lengths = train_set.dataset_dict[key].drop_duplicates(subset=['CLUSTER'])

            # Create a dictionary for mapping cluster ID to length
            cluster_length_map = dict(zip(fixed_lengths['CLUSTER'], fixed_lengths['LENGTH']))

            # Map the cluster ID to the index in the dataset
            ids = train_set.ID_dict[key]
            fixed_lengths_processed = [cluster_length_map.get(id) for id in ids]
            
            # Replace the indices within lengths corresponding to adjusted_index_list with the fixed lengths
            lengths[adjusted_index_list] = torch.tensor(fixed_lengths_processed, dtype=torch.float) 
            
        offset += len(index_list)

    sampler = None
    if num_samples:
        print(f"Sampling {num_samples} examples...")
        np.random.shuffle(indices_to_process)
        indices_to_process = indices_to_process[:num_samples]
        sampler = OrderedSampler(indices_to_process)
    else:
        num_samples = len(indices_to_process)
        sampler = OrderedSampler(indices_to_process)

    # Create the training loader
    dataloader_kwargs = {
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "batch_size": 1,
    }
    
    print("num_workers:", dataloader_kwargs["num_workers"])

    train_loader = data.DataLoader(train_set, sampler=sampler, **dataloader_kwargs)

    print(f"Looping through {num_samples} batches.")
    print("First ten indices:", indices_to_process[:10])
    print("Last ten indices:", indices_to_process[-10:])
    for batch_idx, batch in tqdm(enumerate(train_loader), total=num_samples, desc="Processing batches"):
        try:
            # Get the N_residues from the first tensor, which is the last dimension
            length = batch[0].shape[-1]
            
            # Store the length in a slot corresponding to the absolute index of the example
            lengths[indices_to_process[batch_idx]] = length
        except Exception as e:
            print(f"An error occurred while processing batch {batch_idx}: {e}")
        
        # Break if limiting examples
        if batch_idx >= num_samples - 1:
            break

    # Save length tensor to pickle
    print(f"Saving lengths as a tensor to {output_file}...")
    torch.save(lengths, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    seed_everything()
    main(args.crop_size, args.msa_limit, args.output, args.num_samples, args.num_workers)
