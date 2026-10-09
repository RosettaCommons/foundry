# Common Issues

## Contents

## Ligand Code Conflicts with the CCD
### Issue Description
The [chemical component dictionary (CCD)](https://www.wwpdb.org/data/ccd) has assigned codes to all small molecule and residues that appear in the [PDB](https://www.rcsb.org/). RFD3 (via [AtomWorks](https://github.com/RosettaCommons/atomworks)) can match these codes to their conformational structues in the CCD. However, and especially for enzyme design problems, the ligand structure you are using may be a transition state structure and will purposefully not match what is in the CCD. If this happens you will see several warning messages before the inference run crashes: 
```{bash}
WARNING:atomworks.ml:Atom ZN1 not found in conformer for residue TSA ...
WARNING:atomworks.ml:Atom P1 not found in conformer for residue TSA ...
WARNING:atomworks.ml:Atom N1 not found in conformer for residue TSA ...
... (many more "Atom not found" warnings) ...

ValueError: Transforms failed at stage `CreateDesignReferenceFeatures`:
could not broadcast input array from shape (16,3) into shape (53,3)
```

The shape mismatch error is how you know this particular issue is occuring. 

### Solution
You need to give your ligand a name that does not match any in the CCD. We recommend adding a colon or semicolon to the three-letter code, for example `L:G`. These characters render the code invalid for the CCD, so RFD3 will not try to match it to a known structure and instead use the structure from your PDB directly. You will need to update this label in both your input PDB/CIF file and your input JSON/YAML file. 