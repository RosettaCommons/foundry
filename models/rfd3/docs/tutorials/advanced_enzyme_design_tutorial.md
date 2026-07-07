# Advanced Enzyme Design with RFdiffusion3

## Table of Contents
<!-- TODO: Add table of contents -->

(adv_enzyme_tutorial_intro)=
## Introduction
In this tutorial, you will learn how to design a *de novo* enzyme by generating novel protein backbones that scaffold a pre-defined active site using [RFdiffusion3 (RFD3)](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2). More specifically, you will design a *de novo* metalloprotease for a system comprised of a phosphonamidate transition-state analog, zinc cofactor, and six catalytic residues shown below.

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

However, for enzyme design our goal is to stabilize the **transition state** of the reaction involving our ligand and the key catalytic residues it interacts with. If the PDB contains a structure with a bound transition-state analog (a mimic of the transition state), this structure will be most useful for your enzyme design projects.

For the metallohydrolase we are designing in this tutorial, we will start with a phosphoester. These are known for being a transition state analog for ester- and amide-cleaving metallohydrolases due to their tetrahedral geometry and localized negative charge. A careful search of the [RCSB Protein Data Bank (RSCB PDB)](https://www.rcsb.org/) leads us to [astacin (1QJI)](https://www.rcsb.org/structure/1QJI), a phosphonamidate transition-state analog. <!-- Not sure if this line is actually relevant, it is the only time zinc is mentioned in the introduction:  In the specific case of a zinc protease, peptide substrates bearing a **phosphonamidate** at the cleavage site are especially effective transition-state analogs.--> <!-- TODO: figure out if it is appropriate to include figure1 here, it's for a zinc reaction mechanism, and I'm not sure if it's actually necessary/relevant for what we are trying to accomplish here. -->

Now we need to determine which residues in this protein are important for stabilizing the transition state. For this type of catalytic reaction, it is known that the three histidine residues (H92, H96, and H102 in 1QJI) that surround the zinc ion are crucial for chelation. The glutamic acid residue whose side chain interacts with the zinc ion (E93) is also known to serve as the general base for this hydrolysis reaction. The [article](https://www.nature.com/articles/nsb0896-671) that published the 1QJI structure also reveals that Y149 and M147 may ne necessary for this reaction: Y149 stabilizes the oxyanion that is formed during the reaction and M147 maybe important for conserving the motif that sits below the active site.

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

Make sure to save this structure as a CIF or PDB for for use in the next section.

### Adding an ORI token
ORI (origin) tokens allow you to specify where the center of mass of the *designed* portion of your protein should approximately be. It can be used to have greater control over the interactions between the designed and input portions of your final structure. It can be particularly important for enzyme design as it can be used to guide the approximate orientation of how the generated protein should bind the ligand. <!-- TODO: add image here if Seth gives you the files you need. -->

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
:width:100%
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

We are `unindex`-ing all of the input residues here because their index in the final protein structure is not important, only their orientation with respect to each other, the ligand, and the zinc cofactor matters. 

Which brings us to the `select_fixed_atoms` setting. You can see the atom names for each residue in the PDB/CIF file for the input structure and/or view them on PyMOL using its labeling functionality. You'll notice that all of the atoms specified here are from the side chains of the residues. For this design problem, we want to keep the backbone position flexible to avoid overconstraining the designed protein. Any atoms not listed are free to be moved by default. Only the positions of the side chain atoms relative to the ligand and cofactor are important for enzymatic activity. 

```{important}
Never specify hydrogen atoms in your constraints for RFdiffusion3. RFD3 strips all hydrogen atoms from the input (residues, ligands, cofactors, etc.) during preprocessing.
```

<!-- TODO: add image of the atoms that need to be held fixed. -->

Last, but not least, we specify the ORI token we discussed at the end of the last section. Feel free to try various ORI locations and see how they impact your results. 




(adv_enzyme_tutorial_glossary)=
## Glossary

(adv_enzyme_tutorial_theozyme_def)=
### Theozyme
<!-- TODO: Add definition -->
### ORI Token
<!-- TODO: Add definition-->

(adv_enzyme_tutorial_refs)=
## Resources and References
- [RFdiffusion3 preprint](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2)
- The procedure described here follows the approach discussed in [Kim, D. et al. (2025)](https://www.nature.com/articles/s41586-025-09746-w) and [Chen, A. et al. (2025)](https://www.nature.com/articles/s41586-025-09746-w)
- Astacin structure: Grams, F. et al. (1996). Structure of astacin with a transition-state analogueinhibitor. *Nature Structural Biology* **3**, 671-675. DOI: https://doi.org/10.1038/nsb0896-671
- 

