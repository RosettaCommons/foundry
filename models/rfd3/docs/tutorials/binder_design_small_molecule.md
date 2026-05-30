# Designing Binders for Targets with Non-Protein Molecules

## CD28: Cleaning the Starting Structure
- keeping chain C and the NAG molecules around it
    - remove chains A, B, D, E, F
    - remove the metal ions
    - remove GOL
    - remove IMD
    - remove unresolved residues
- renaming so that the main structure is chain B and all the NAG molecules come after it in their own chains, each numbered 1:
    - select chain C
    - deselect any NAG molecules that come with it
    - `alter sele, chain='B'` and `alter sele, segi='B'`
    - change AA/C/204 to chain D residue 1: `alter sele, chain='D'` ` alter sele, resi='1'`, `a;ter sele, segi='D'`
    - repeat for BA/C/205 to be on chain E, 
    - The G NAG 1 is actually already fine
    - The second G NAG is actually still that way in the PDB that they uploaded?? Oh it's because they're bonded together
    - The X/C/NAG201 goes to chain F
    - The Y/C/NAG202 goes to chain C
    - sort
- No renumbering of chain B was necessary
- The gap in this chain (between residues 18 and 20) was replaced with asparagine

## PDF1 
- fetch 8s1x
- remove everything that isn't the protein structure and the ligand (BB2)
    - `remove resn K`
    - `remove resn ZN`
    - `remove resn PO4`
- remove the unresolved residue in chain A (A1)
- remove chain B
- renumber chain A so that it starts at 1: `sele chain A`, `alter (sele),resi=str(int(resi)-1)`
- relabel the ligand so it's chain C