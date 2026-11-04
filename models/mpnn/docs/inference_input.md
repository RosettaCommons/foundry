# Inference Input Options

Inference (protein design with an already-trained model) can be controlled either through command line parameters, or through a JSON config file.

To help in converting existing command lines which use the standalone ProteinMPNN and LigandMPNN programs, see [](conversion).

Below, where options take "complex" data structures, these are generally JSON-formatted. (When passed on the command line, surround JSON-like data in single quotes to prevent your shell from intepreting it as a shell command.)

## JSON configuration

For ease of batch usage, a JSON configuration file can be specified, which contains all the settings for the run. 

    --config_json CONFIG.json

If `--config_json` is specified, no other command line parameters will be parsed. See [](config_json) for the format description.

Currently `--config_json` is the only way to specify running multiple inputs in a single invocation.

## Required options

Model type (one of `protein_mpnn`, `ligand_mpnn` or `soluble_mpnn`).

    --model_type TYPE_mpnn

Input structure file to design (either CIF or PDB). 
Note that as MPNN uses atomworks for input, it follows the convention that CIF files are read in as the annotated biounit, rather than the asymetric unit.
(That is, atomworks will add/remove chains to your input CIF on read-in if the asymmetric unit doesn't match the biounit.) PDB files will be read in as-is, even if there is an asymetric unit/biounit mismatch.

    --structure_path INPUT.cif

Currently there isn't any support for multiple structure input on the command line (Use `--config_json` instead.)

Output directory (will be created if it doesn't exist)

    --out_directory DIR

## Commonly used options

Control the number of output designs (both options default to 1):

    --batch_size BATCH_SIZE
    --number_of_batches NUMBER_OF_BATCHES

Control the randomization used (different seeds will yield different designs for the same inputs):

    --seed SEED

Control which residues/chains are designed. These options values can either be comma separated or a JSON-formatted list. Note that these cannot be combined -- only one is allowed per run:

    --fixed_chains '["A","B"]'
    --designed_chains "A,B"
    --fixed_residues "A35,B40,C52"
    --designed_residues '["A35","B40","C52"]'

Skip particular residues in design. (The default is just "UNK" - it is recommended to include this in your list.)

    --omit '["CYS","GLY","UNK"]'

## Output control

Change the label on the output files (default is to base it on the input file name):

    --name NAME

Turn off FASTA writing:

    --write_fasta False

Turn off CIF structure output:

    --write_structures False

## Advanced Options


### Controlling sampling

To control the variability of sampling (higher temperature yield more variable results, lower temperatures less variable but higher confidence)
Default is 0.1

    --temperature TEMPERATURE

This can be specified on a per-residue level:

    --temperature_per_residue '{"A35": 0.2}'

To add a small amount of noise to the structure prior to running the prediction (increases variability)

    --structure_noise NOISE_IN_ANG

By default, the design process ignores the input identities at all designed positions. To include that information in the decoding process: 

    --initialize_sequence_embedding_with_ground_truth True

### Controlling amino acid usage

To omit certain residues from design for the entire protein (will not affect non-designed positions).

    --omit '["CYS","GLY","UNK"]'

This can be also controlled on an per-residue level. (If specified, `--omit` is ignored.):

    --omit_per_residue '{"A35":["ALA","GLY","UNK"], "B23":["CYS","GLY","UNK"]}'

To downweight/upweight certain amino acid identities:

    --bias '{"LEU": 1.0, "GLY": -0.5, "CYS": -2.0}'

Higher numbers mean the amino acid will be more frequent in the output. More negative numbers reduce the frequency.
Amino acid identities which aren't specified get a value of 0. 

Per-position biases can be specified (positions not specified default to 0):

    --bias_per_residue '{"A35": {"ALA": -2.0}}'

### Symmetry

To specify that certain positions should get the same identity during design (e.g. they're symmetric or pseudo-symmetric positions), 
specify the tied groups with `--symmetry_residues`

    --symmetry_residues  '[["A35","B35"],["A40","B40","C40"]]'

This is a list-of-lists of tied residues. The residues in each internal list will all share a residue identity.

Control the weighting of the groups of `--symmetry_residues` (must match the structure):

    --symmetry_residues_weights '[[1.0, 1.0], [1.0, 0.5, -0.5]]'

If entire chains should be the same, `--homo_oligomer_chains` may be more convienient.

    --homo_oligomer_chains '[["A","B"],["C","D","E","F"]]'

If `--homo_oligomer_chains` is set, `--symmetry_residues` and `--symmetry_residues_weights` are ignored.
