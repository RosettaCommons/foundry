# Advanced Enzyme Design with RFdiffusion3

## Table of Contents

(adv_enzyme_tutorial_intro)=
## Introduction
In this tutorial, you will learn how to design enzymes for a system comprised of a phosphonamidate transition-state analog, zinc cofactor, and six catalytic residues using RFdiffusion3 (RFD3). You will:
1. Crop an initial system to form a [theozyme](#adv_enzyme_tutorial_theozyme_def) using [PyMOL](https://www.pymol.org/)
1. Determine which configuration options to use for your designs and create an input JSON/YAML file
1. Run RFD3 on your own computing systems
1. Analyze the outputs
1. Determine the impacts of adding hydrogen bond and relative accessible surface area (RASA) conditioning

```{note}
[PyMOL](https://www.pymol.org/) is not necessary to complete this tutorial, the steps shown here can be replicated using other protein visualization tools. 
```

(adv_enzyme_tutorial_getting_started)=
## Before We Get Started...
This tutorial does not cover installing RFD3. If you do not already have RFD3 installed on your system see the installation see the [Getting Started section in the RFD3 README](https://github.com/RosettaCommons/foundry/tree/production/models/rfd3/docs#getting-started) and our guide for [Installing RFdiffusion3 on Unix Systems](./RFdiffusion3_installation_tutorial.md). 

(adv_enzyme_tutorial_prereqs)=
## Prerequisites
- RFdiffusion3

(adv_enzyme_tutorial_glossary)=
## Glossary

(adv_enzyme_tutorial_theozyme_def)=
### Theozyme
<!-- TODO: Add definition -->