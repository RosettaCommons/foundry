# Advanced Enzyme Design with RFdiffusion3

## Table of Contents
<!-- TODO: Add table of contents -->

(adv_enzyme_tutorial_intro)=
## Introduction
In this tutorial, you will learn how to design a *de novo* enzyme by generating novel protein backbones that scaffold a pre-defined active site using [RFdiffusion3 (RFD3)](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2). More specifically, you will design a *de novo* metalloprotease for a system comprised of a phosphonamidate [transition-state analog](#adv_enzyme_tutorial_transition_state_analog_def), zinc [cofactor](#adv_enzyme_tutorial_cofactor_def), and six [catalytic residues](#adv_enzyme_tutorial_catalytic_residue_def) shown below.

```{figure}
<!-- TODO insert image of initial/final (?) system here -->
```

General procedure:
1. Crop an initial system to form a [theozyme](#adv_enzyme_tutorial_theozyme_def)
1. Determine which configuration options to use for your designs and create an input JSON/YAML file
1. Run RFD3 on your own computing systems
1. Analyze the outputs
1. Determine the impacts of adding hydrogen bond and relative accessible surface area (RASA) conditioning

(adv_enzyme_tutorial_getting_started)=
## Before We Get Started...
This tutorial does not cover installing RFD3. If you do not already have RFD3 installed on your system see the installation see the [Getting Started section in the RFD3 README](https://github.com/RosettaCommons/foundry/tree/production/models/rfd3/docs#getting-started) and our guide for [Installing RFdiffusion3 on Unix Systems](./RFdiffusion3_installation_tutorial.md). 

You will also need a molecular visualization tool for manipulating our starting structure. Here we will use [PyMOL](https://www.pymol.org/), however other visualization tools, such as [UCSF Chimera](https://www.cgl.ucsf.edu/chimera/), can be used instead. 

RFD3 is optimized to run on GPUs, specifically NVIDIA GPUs. It is recommend that you have at least 16 GB of GPU memory when running inference for this tutorial. For your own projects, you may need more or less depending on the size of the system you are working with. 

(adv_enzyme_tutorial_prereqs)=
## Prerequisites
- RFdiffusion3 installed and working
- Familiarity with [command line](https://www.freecodecamp.org/news/command-line-for-beginners/)
- Protein visualization software, here we will use [PyMOL](https://www.pymol.org/)

```{note}
PyMOL is not necessary to complete this tutorial, the steps shown here can be replicated using other protein visualization tools. 
```

**Recommended:**
- Familiarity with a text editor (e.g. [emacs](https://www.gnu.org/savannah-checkouts/gnu/emacs/emacs.html), [vim](https://www.vim.org/), [sublime](https://www.sublimetext.com/), etc.)
- Interactive development environment (e.g. [VS Code](https://code.visualstudio.com), [Cursor](https://cursor.com/home))

## Setup 
<!-- TODO: Add setup details depending on input and output file requirements -->

## Designing Metalloproteases
### Creating a Theozyme
Structural inputs to RFdiffusion3 typically either come from a structure-prediction model (e.g. [AlfaFold3](https://www.nature.com/articles/s41586-024-07487-w)) or from an experimental structure reported in resources like the [RCSB Protein Data Bank (PDB)](https://www.rcsb.org/)

However, for enzyme design our goal is to stabilize the **transition state** of the reaction involving our ligand and the key [catalytic residues](#adv_enzyme_tutorial_catalytic_residue_def) it interacts with. If the PDB contains a structure with a bound transition-state analog (a mimic of the transition state), this structure will be most useful for your enzyme design projects.

For the metallohydrolase we are designing in this tutorial, we will start with a phosphoester. These are known for being a transition state analog for ester- and amide-cleaving metallohydrolases due to their tetrahedral geometry and localized negative charge. A careful search of the [RCSB Protein Data Bank (RSCB PDB)](https://www.rcsb.org/) leads us to [astacin (1QJI)](https://www.rcsb.org/structure/1QJI), a phosphonamidate transition-state analog. <!-- Not sure if this line is actually relevant, it is the only time zinc is mentioned in the introduction:  In the specific case of a zinc protease, peptide substrates bearing a **phosphonamidate** at the cleavage site are especially effective transition-state analogs.--> <!-- TODO: figure out if it is appropriate to include figure1 here, it's for a zinc reaction mechanism, and I'm not sure if it's actually necessary/relevant for what we are trying to accomplish here. -->

Now we need to determine which residues in this protein are important for stabilizing the transition state. For this type of catalytic reaction, it is known that the three histidine residues (H92, H96, and H102 in 1QJI) that surround the zinc ion are crucial for [chelation](#adv_enzyme_tutorial_chelation_def). The glutamic acid residue whose side chain interacts with the zinc ion (E93) is also known to serve as the general base for this hydrolysis reaction. The [article](https://www.nature.com/articles/nsb0896-671) that published the 1QJI structure also reveals that Y149 and M147 may ne necessary for this reaction: Y149 stabilizes the oxyanion that is formed during the reaction and M147 maybe important for conserving the motif that sits below the active site.

So for this example, we will be using 6 catalytic residues to create our theozyme along with the ligand and the zinc ion: H92, E93, H96, H102, M147, and Y149. You can see these residues highlighted in the structure below: 

```{figure} ../.assets/adv_enzyme_design_tutorial/catalytic_residues.png
:width: 100%
:alt: 1QJI protein structure with catalytic residues highlighted.
Image of 1QJI with the catalytic residues highlighted in pink, the zinc ion is shown in gray, and the astacin ligand is in orange. 
```

Now that we've identified the catalytic residues, we can crop our structure to only include these residues, the zinc ion, and the ligand (labeled 'PKF' in the structure from the PDB), resulting in the theozyme shown below: 
```{figure} ../.assets/adv_enzyme_design_tutorial/theozyme.png
:width: 80%
:alt: Theozyme structure: catalytic residues, zinc ion, and astacin lingand.

The structure we will use as input to RFD3. The catalytic residues are pink, the zinc ion is shown in gray, and the astacin ligand is orange. 
```

For how to crop and save structures in PyMOL, see the 'Motif Preparation' section [Intermediate Enzyme Design Tutorial](./intermediate_enzyme_design_tutorial.md#intermediate_enzyme_motif_prep). You can compare your result to the [`theozyme.pdb`](./advanced_enzyme_tutorial_files/theozyme.pdb) file provided in the RFD3 documentation. 

Make sure to save this structure as a **PDB** for for use in the next section.

### Adding an ORI token
[ORI (origin) tokens](#adv_enzyme_tutorial_ori_token_def) allow you to specify where the center of mass of the *designed* portion of your protein should approximately be. It can be used to have greater control over the interactions between the designed and input portions of your final structure. It can be particularly important for enzyme design as it can be used to guide the approximate orientation of how the generated protein should bind the ligand. <!-- TODO: add image here if Seth gives you the files you need. -->

For our metalloprotease designs, we can start by assuming that the ORI token should be placed near the zinc atom since it should be relatively buried in the enzyme structure. You could just determine the coordinates of the zinc atom and use these for the ORI token input. However, for this tutorial let's say that we know we want the ORI token to actually sit slightly below the zinc atom. We could determine the coordinates of this ourselves, but often times it is helpful to determine the placement of our ORI token visually. 

First, let's add a pseudoatom to our PyMOL session with our theozyme structure. We don't know where to place this pseudoatom, and it'd be best to place it close to our structure so that we don't need to hunt for it in our PyMOL workspace. Let's figure out the placement of our zinc atom and then place our pseudoatom near it. To determine the coordinates of the zinc atom in PyMOL, select the zinc atom and run the following in the command prompt: 
```bash
iterate_state 1, sele, print(name, x, y, z)
```
You might see it print the coordinates several times. 

Now let's add the pseudoatom via the PyMOL command prompt: 
```bash
cmd.pseudoatom(object="ORI", pos=[17.79, 24.45, 22.30], elem="ORI", name="ORI", vdw=1.5, hetatm=True, chain='z', segi='z', resn="ORI"); cmd.show("sphere", "ORI");
```
This command:
1. Sets the name of the object that appears in the right-hand sidebar in PyMOL.
1. Sets the position of the pseudoatom, here we have simply added 1Å in the z direction from the zinc atom coordinates to not have the object completely overlap with the zinc atom.
1. Adds labels to the token.
1. Setts the Van der Waals radius of the atom - this will be useful for visualizing the token.
1. Adds it to a new 'z' chain and segment.
1. Sets this new object to appear as a sphere. 

Your PyMOL window should now look something like: 
```{figure} ../.assets/adv_enzyme_design_tutorial/ori_1.png
:width: 100%
:alt: Theozyme (catalytic residues, zinc ion, ligand) with pseudoatom to represent ORI token. 

Initial placement of the ORI token (white sphere). The ligand is shown in orange, the catalytic residues in pink, and the zinc ion in gray. 
```

Now we can move the token around by hand by using the right-hand menu. Select the A(ction) menu for the ORI object and select **drag coordinates**. 
```{figure}
:width: 100%
:alt: Action menu for the ORI object.

Right-hand panel in PyMOL with the action menu for the ORI object open. The 'drag coordinates' option is highlighted in white. 
```
You should now be able to move your pseudoatom by holding shift while clicking on then dragging the pseudoatom with the middle button on your mouse. This may vary depending on your PyMOL version and OS. Your final ORI location should lead to a structure that looks approximately like: 

```{figure} ../.assets/adv_enzyme_design_tutorial/ori_2.png
:width: 100%
:alt: Final location of ORI token with theozyme structure. 

Final location of the ORI token relative to the theozyme structure. The ORI token is shown in white, the catalytic residues in pink, the zinc ion in gray, and the ligand in orange. 
```

Once you have the ORI token in place, you can follow the previous instructions to have PyMOL print out its coordinates. Here our ORI token coordinates ended up being [17.49, 23.97, 19.17] You will need to know these for setting up your configuration file.

### Preparing the Configuration File
The main inputs to RFdiffusion3 are a structure file (optional) and a JSON/YAML file (required) that specifies and constraints you want to apply to the diffusion process. Here we will be using the JSON file format, however the same options can be used with the YAML format. 

```{important}
We will only be discussing the options relevant to this tutorial example. For a list of all constraints you can apply to RFdiffusion3, see the [Input Specification Fields list](../input.md#inputspecification-fields). 
```

Open a new file called `metalloprotease_rfd3_input.json` in a text editor of your choice. Add the following and ensure all the brackets are closed: 
```json
{
    "test1": {
        "input": "<path/to/input/theozyme.pdb",
        "ligand": "ZN,PKF",
        "length": "120-150",
        "unindex": "A92,A93,A96,A102,A147,A149",
        "select_fixed_atoms": {
            "A92": "NE2,CD2,CG,CB,ND1,CE1",
            "A93": "OE1,OE2,CD,CG",
            "A96": "NE2,CD2,CG,CB,ND1,CE1",
            "A102": "NE2,CD2,CG,CB,ND1,CE1",
            "A147": "SD,CE,CG",
            "A149": "OH,CZ,CE1,CD1,CE2,CG,CB"
        },
        "ori_token": [
            17.349,
            23.971,
            19.174
        ]
    }
}
```

```{important}
Make sure to replace `<path/to/input/>` with the actual path to your `theozyme.pdb` file. This path can be relative to where you are saving your JSON file or absolute. 
```

```{note}
All of the options used here have been discussed in the [introductory](enzyme_design_tutorial.md) and [intermediate](intermediate_enzyme_design_tutorial.md) enzyme design tutorials. We will only discuss the choices unique to this tutorial here, please refer to them and the [Input Specification documentation](../input.md#inputspecification-fields) for more information. 

For examples of the JSON format, see the [Examples](../index.rst#examples).
```

Here we specify both the zinc atom (ZN) and the astacin molecule (PKF) as ligands because we want RFD3 to be aware of them as they are designing the protein structure. Note that the names given need to match what is in your PDB/CIF file. 

The length setting is telling RFdiffusion3 to design a protein that is between 120 and 150 amino acids in length, inclusive of the input residues. We are not using a `contig` setting here because we are using other ways to specify the designed or input portions of our structure. 

We are `unindex`-ing all of the input residues here because their index in the final protein structure is not important, only their orientation with respect to each other, the ligand, and the zinc [cofactor](#adv_enzyme_tutorial_cofactor_def) matters. 

Which brings us to the `select_fixed_atoms` setting. You can see the atom names for each residue in the PDB/CIF file for the input structure and/or view them on PyMOL using its labeling functionality. You'll notice that all of the atoms specified here are from the side chains of the residues. For this design problem, we want to keep the backbone position flexible to avoid overconstraining the designed protein. Any atoms not listed are free to be moved by default. Only the positions of the side chain atoms relative to the ligand and cofactor are important for enzymatic activity. 

```{important}
Never specify hydrogen atoms in your constraints for RFdiffusion3. RFD3 strips all hydrogen atoms from the input (residues, ligands, cofactors, etc.) during preprocessing.
```

<!-- TODO: add image of the atoms that need to be held fixed. -->

Last, but not least, we specify the ORI token we discussed at the end of the last section. Feel free to try various ORI token locations and see how they impact your results. 

### Running RFdiffusion3

Once the input structure and JSON/YAML file have been prepared we can run RFD3. The simplest possible command to do so is
```bash
rfd3 design out_dir=</path/to/outputs/> inputs=</path/to/inputs/>metalloprotease_rfd3_input.json
```
This will generate 8 designs. If the output directory that you specify does not exist, RFD3 will create it.

However, for this tutorial we recommend changing some of the default settings for how RFD3 runs: 
```bash
rfd3 design \
    out_dir=<path/to/outputs/> \
    inputs=<path/to/inputs/>metalloprotease_rfd3_input.json \
    skip_existing=False \
    dump_trajectories=True \
    prevalidate_inputs=True \
    diffusion_batch_size=4 \
    n_batches=6 \
    inference_sampler.use_classifier_free_guidance=True \
    inference_sampler.s_jitter_origin=1.5 \
    seed=42
```
Here is a brief description of what each of these additional settings are changing: 
- `skip_existing=False` tells RFD3 to overwrite any existing files in the output directory that would have the same name as those that would be created. This is `True` by default to avoid re-running a calculation you already have results for. 
- `dump_trajectories=True` will generate two trajectory files (`noisy` and `denoised`) that will show different aspects of the diffusion process. See the [small molecule binder design tutorial](binder_design_tutorial.html#step-3-running-rfd3) for a discussion of these files. Note that these files can take up a bit of space on your machine.
- `prevalidate_inputs=True` will check your JSON/YAML file for any formatting issues.
- `diffusion_batch_size=4` changes the diffusion batch size from 8 to 4, lowering the default batch size can help if you run into GPU memory errors. 
- `n_batches=6` run 6 batches of RFD3 instead of 1, for a total of 24 designs. The total number of designs you want to generate will depend on your design needs.
- `inference_sampler.use_classifier_free_guidance=True` turns on classifier free guidance so that the model learns to predict the denoised structure with and without some of the conditioning features. It results in better adherence to your input constraints at the cost of lower diversity in the designed structures. 
- `inference_sampler.s_jitter_origin=1.5` adds a small amount of positional noise to the ['motif'](#adv_enzyme_tutorial_motif_def), the fixed portions of the structure that were given to RFD3 as input. Useful for exploring how the protein scaffold can orient around the active site. 
- `seed=42` sets the seed to increase the reproducibility of results between RFD3 calculations. As of the publication of this tutorial, setting the seed **does not** result in fully deterministic results – they will still be slightly different between runs. 

Feel free to reduce the number of batches or batch size if you have limited GPU resources.

You can learn more about these settings and other possible options [here](../input.md#cli-arguments). 

### Analyzing the Outputs
You can find a set of example outputs <!-- TODO --> with the tutorial files. These files will not completely match what you produce. 

You should see 4 types of files for each design (96 files total) in your output directory. Each design should have:
- `<prefix>_model_<N>.cif.gz`: The final structure of the given design. 
- `<prefix>_model_<N>.json`: A JSON file containing quality metrics, index mapping, the full input specification, and sampler parameters for the design. 
- `<prefix>_denoised_model_<N>.cif.gz`: This trajectory files shows what the diffusion network thinks the final clean structure will be at each timestep. The input motif is not held fixed in this view. Can be used to see what the model ‘learned’ at each step as it is easier to watch the secondary structure emerge during the diffusion process.
- `<prefix>_noisy_model_<N>.cif.gz`: A trajectory that shows how the diffusion process actually progressed while the input motifs are held fixed. Can be used to verify motif integrity.

Let's go through the outputs associated with one of the designs (all files can be found here <!-- TODO: link files -->) to show some of the ways one might analyze the outputs from RFD3. 

#### Final structure
First, let's open the `.cif.gz` file in PyMOL. If you are using the provided tutorial files your structure should look like this: 
<!-- TODO: insert figure -->

Here is what we are looking for: 
1. **Overall fold.** Does the protein form a compact, well-folded structure? Is the active site located in a cleft or pocket, as you would expect for an enzyme?
1. **Catalytic geometry.** Are the fixed atoms (H92/E93/H96/H102/Y149/M147) in their expected positions relative to the Zn(II) ion and the astacin ligand? Do the key interactions look reasonable? (You can find the new residue numbers for the [catalytic residues](#adv_enzyme_tutorial_catalytic_residue_def) in the `diffused_index_map` section of the design's JSON output file.)
1. **Backbone connectivity.** Does the backbone trace smoothly through the structure without obvious clashes or unnatural loops?
1. **Active site accessibility.** Is the ligand reasonably accessible from the protein surface? A completely buried ligand may be problematic depending on the application.

```{tip}
When viewing multiple designs in PyMOL, you can use the `alignto` to roughly align all of the structures. Keep in mind that the residue numbers will vary between each design.
```

#### Output JSON
Next let's inspect the JSON files. The first section of the JSON file includes the `diffused_index_map` which shows where any input residues have ended up in your design. The indices on the left of the colon are from your input structure, the right are where these residues are in your final design. 

The next section is the `metrics` section that includes many values that are automatically calculated by RFD3, only a few of which we will discuss here. You can see more details about all of these values in the [Output Metrics documentation](../output.md).

For this type of enzyme design problem, you will likely care about:
- `insertion.rmsd`: measures how well the unindexed motif was placed into the generated backbone.
- `n_chainbreaks`: the number of chainbreaks in your system, here we want none. 
- `n_clashing.interrresidue_clashes_w_sidechains`: the number of clashes between sidechains.
- `n_clashing.ligand_clashes`: number of clashses between the design and the ligand
- `non_loop_fraction`: fraction of residues in a recognizable secondary structure rather than in a loop.

#### Specification and Inference Sampler
These two sections will allow you to recreate this inference calculation. `specification` is a copy of your input JSON/YAML settings with some extra settings that are inherent to your input.`inference_sampler` is a record of all of the sampler hyperparameters that were set during the diffusion process.

### Filtering Script
While looking at these files is instructive, it is impossible to do for the tens, hundreds, or even thousands of designs you might generate for your research projects. You can instead write a simple python script to filter these designs based on the various metrics that you care about. 

Here's an example of a simple filtering script: 
```python

import json, glob

# sort the files by name and print a header
jsons = sorted(glob.glob("outputs/*_model_*.json"))
   print(f"{'File':<45} {'RMSD':>6} {'Join':>6} {'Breaks':>6} {'Clashes':>7} {'SS%':>5} {'Nres':>5}")
   print("-" * 85)

# print the below metrics for each file
for path in jsons:
    with open(path) as f:
       d = json.load(f)
    m = d["metrics"]
    name = path.split("/")[-1].replace("rfd3__1qji__ZnProtease_HEHHMY_", "")
    print(f"{name:<45} "
        f"{m['insertion_rmsd']:6.3f} "
        f"{m['join_point_rmsd']:6.3f} "
        f"{m['n_chainbreaks']:6d} "
        f"{m['n_clashing.interresidue_clashes_w_sidechain']:7d} "
        f"{m['non_loop_fraction']:5.2f} "
        f"{m['num_residues']:5d}")

# filter based on hard-coded cutoffs and values
for path in jsons:
    with open(path) as f:
        d = json.load(f)
    m = d["metrics"]
    if (m["insertion_rmsd"] < 0.5
        and m["n_chainbreaks"] == 0
        and m["n_clashing.interresidue_clashes_w_sidechain"] == 0
        and m["n_clashing.ligand_clashes"] == 0
        and m["non_loop_fraction"] > 0.6):
        print(f"PASS: {path}")  
```

It is still **highly recommended** that you look at any of your passing designs in PyMOL after a quantitative filter. Some will still have issues that will not be captured by the metrics. For example, for this type of enzyme design problem, we will want relatively compact structures. 

## Advanced Input Specifications
The process discussed thus far in the tutorial shows the input specifications that will be generally useful for any enzyme design task. Here we will look at the addition of two more categories of input specification: hydrogen bond conditioning and relative accssible surface area (RASA) conditioning.

### Hydrogen Bond Conditioning
```{important}
You must have [HBPLUS](https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/) installed on your system as described in the [RFD3 README](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/README.md#install-hbplus-for-training-with-hydrogen-bond-conditioning) to run RFD3 with hydrogen bond conditioning.
```
The `select_hbond_donor` and `select_hbond_acceptor` inut specification options allow you to tell RFD3 which specific atoms in your input structure should form be used as hydrogen bond donors and acceptors, respectively, in your final design. The addition of these parameters can help with the binding specificity of your designed enzyme structures. 

You specify the atoms to use for the hydrogen bond donor/acceptor via a dictionary. If you would like to see the impacts of adding hydrogen bond conditioning to the design task described in this tutorial add the following to your JSON file: 
```json
"select_hbond_acceptor": {
    "PKF": "O4,O6,O7"
},
"select_hbond_donor": {
    "PKF": "N2,N4,N20"
}
```

You can find an example output from the addition of these diffusion constarints here. <!-- TODO: add link to files -->

### RASA Conditioning
Relative accesible surface area (RASA) conditioning allows you to tell RFD3 how exposed to the solvent or buried in the designed structure you want portions of your input structure to be in your final design. This is particularly useful in enzyme design as you will want certain parts of the subtrate to be enclused by the protein while others should extend outside of the active site cleft.

There are three input parameters that are used to control RASA conditioning in your design: 
- `select_buried` — atoms that should be **surrounded by protein** (low solvent accessibility)
- `select_exposed` — atoms that should be **accessible to solvent** (high solvent accessibility)
- `select_partially_buried` — atoms with **intermediate** solvent accessibility (less commonly used)

The specific atoms you want buried, exposed, or partially buried are specified in the same format as for `select_fixed_atoms`. Here is an example of what these settings could look like for the design example discussed in this tutorial: 
```json
"select_buried": {
    "ZN": "ZN",
    "PKF": "O5,P1,O6,C28,C29,C33,C36"
},
"select_exposed": {
    "PKF": "N5,C19,O2,C2,O8,C23,C24,C31,O3"
}
``` 
The buried were chosen because we want to bury the zinc ions and the atoms near the reactive center of PKF. These are direvely involved in the catalytic mechanism so they should be enclosed by the protein. The atoms on the peptide tail of PKF, meanwhile, should be exposed as they would naturally protrude from the binding cleft in a real enzyme-subtrate complex.

You can find an example output from the addition of these diffusion constarints here. <!-- TODO: add link to files -->

You can find an example output from a design that uses both hydrogen bonding and RASA conditioning here. <!--TODO: add link to files-->

(adv_enzyme_tutorial_conclusion)=
## Conclusion
You have now set up an RFD3 calculation and successfully designed enzymes based around a theozyme created from a known structure. While the options discussed here are particularly useful in enzyme design projects, RFD3 has many more that you can explore by looking at {doc}`../input`.

(adv_enzyme_tutorial_glossary)=
## Glossary

(adv_enzyme_tutorial_catalytic_residue_def)=
### Catalytic Residue
Catalytic residues are amino acids known to be cruicial for the enzymatic activity of a given reaction. They can either directly interact with the molecule the enzyme interacts with, or indirectly support the stability of the transition state of the ligand. 

(adv_enzyme_tutorial_chelation_def)=
### Chelation
Chelation describes a type of interaction between ligands and metal atoms that form a ring structure. 

(adv_enzyme_tutorial_cofactor_def)=
### Cofactor
A cofactor is a non-protein molecule that binds to an enzyme to help it function.

(adv_enzyme_tutorial_motif_def)=
### Motif
The input structure to RFD3 that designs are generated around. 

(adv_enzyme_tutorial_ori_token_def)=
### ORI Token
The ORI token is the user-specified center of mass of the struture designed by RFD3. It gives the user some control over the interactions between the designed and input portions of the final structure.

(adv_enzyme_tutorial_theozyme_def)=
### Theozyme
A theozyme is a small structure formed from the transition state structure of a ligand and any catalytically important residues, atoms, etc. that are necessary to achieve a given catalytic reaction. It is the input structure for enzyme design calculations and can be created from known enzyme structures or via quantum mechanical methods. 

(adv_enzyme_tutorial_transition_state_analog_def)=
### Transition State Analog
A transition state analog is a compound that resembles the transition state of a substrate molecule in an enzyme-catalyzed reaction


(adv_enzyme_tutorial_refs)=
## Resources and References
- [RFdiffusion3 preprint](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2)
- The procedure described here follows the approach discussed in [Kim, D. et al. (2025)](https://www.nature.com/articles/s41586-025-09746-w) and [Chen, A. et al. (2025)](https://www.nature.com/articles/s41586-025-09746-w)
- [Astacin structure](https://doi.org/10.1038/nsb0896-671)

