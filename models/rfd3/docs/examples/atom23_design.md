# RNA / DNA Design in RFdiffusion3

This guide describes extensions to RFdiffusion3 for nucleic acid and hybrid RNA–protein design, including:

- RNA/DNA-aware contigs (`R` / `D` suffix)
- Ligand-conditioned aptamer design
- Secondary structure (SS) conditioning
- Base-pair constraints (region- and position-level)
- Partial structure fixing and unindexing

---

## 1. Contig Syntax for RNA/DNA

Contigs now support nucleic acid specification:

- `R` → RNA segment  
- `D` → DNA segment  
- No suffix → protein (default)  

### Example

```json
{
    "contig": "40-50R,/0,10-20D,/0,80-110"
}
```
This corresponds to: 40–50 nt RNA, chain break, 10–20 nt DNA, chain break, 80–110 aa protein

Multipolymer Design

```json

{
    "multipolymer": {
        "contig": "40-50R,/0,10-20D,/0,80-110",
        "length": "130-180",
        "input": "../input_pdbs/AMP.pdb"
    }
}
```

## 2. Secondary Structure Conditioning
### 2.1 Dot-Bracket Notation (Global)
```json
{
    "W05": {
        "ss_dbn": ".(((((((((((((((((((..[[[[[[.)))))(((....)))(((....)))))))))))))))))((((((..]]]]]].)))))).",
        "select_fixed_atoms": false,
        "contig": "90-90R",
        "length": "90-90",
        "input": "../input_pdbs/AMP.pdb"
    }
}
```
`ss_dbn` specifies full RNA secondary structure

Will be applied to the first L tokens, where L is the length of `ss_dbn`.

### 2.2 Dictionary-Based SS Input

Specify secondary structure for subsections:
``` json
{
    "ss_dbn_dict": {
        "A6-25": "(((..)))....(((..)))",
        "B1-20": "((((..))))...((...))"
    }
}
```
Used in:
``` json
{
    "dict_input_ss": {
        "ss_dbn_dict": {
            "A6-25": "(((..)))....(((..)))",
            "B1-20": "((((..))))...((...))"
        },
        "contig": "30-30R,/0,30-30R",
        "length": "60-60",
        "input": "../input_pdbs/AMP.pdb"
    }
}
```
## 3. Base Pair region Conditioning
### 3.1 Paired Regions

Define paired and loop regions:
```json
{
    "paired_region_list": ["A20-25,B10-15"],
    "loop_region_list": ["A10-19","B20-30"]
}
```
Enforces pairing and loop propensity between residue ranges during sampling

Used in:
```json
{
    "paired_region_input_ss": {
        "paired_region_list": ["A20-25,B10-15"],
        "loop_region_list": ["A10-19","B20-30"],
        "contig": "50-50R,/0,50-50R",
        "length": "100-100",
        "input": "../input_pdbs/AMP.pdb"
    }
}
```

### 3.2 Explicit Base Pair Positions

Fine-grained base pairing control:

```json
{
    "paired_position_list": [
        "A3,B3","A5,B5","A7,B7","A9,B9","A11,B11",
        "A13,B13","A15,B15","A17,B17","A19,B19"
    ]
}
```
Used in:
```json
{
    "paired_position_input_ss": {
        "paired_position_list": [
            "A3,B3","A5,B5","A7,B7","A9,B9","A11,B11",
            "A13,B13","A15,B15","A17,B17","A19,B19"
        ],
        "contig": "20-20R,/0,20-20R",
        "length": "40-40",
        "input": "../input_pdbs/AMP.pdb"
    }
}
```
### Note: Most of the above jsons is not actually reading the `input` field. Kept as a dummy for the `inference3_engine`.

## 4. Ligand-Conditioned Aptamer Design

Supports small molecule binding RNA design.

AMP Aptamer Example
```json
{
    "AMP_aptamer": {
        "input": "../input_pdbs/AMP.pdb",
        "ligand": "AMP",
        "contig": "40-50R",
        "length": "40-50",
        "ori_jitter": 1,
        "select_buried": {"AMP": "ALL"},
        "select_hbond_acceptor": {
            "AMP": "N7,O4',O1P,O2P,O3P,N3,N1"
        },
        "select_hbond_donor": {
            "AMP": "N6,O3',O2'"
        }
    }
}
```
Key Options

`ligand`: ligand name in the input PDB

`select_buried`: enforce burial of ligand atoms

`select_hbond_acceptor` / `select_hbond_donor`: suggest Hbond interaction atoms

`ori_jitter`: small random perturbation of ori token (from ligand COM)


## 5. Hybrid RNA–Protein Design with Constraints
### RNase P Active Site Example

```json
{
    "unindexed_rnasep": {
        "input": "../input_pdbs/rnase_p_3q1q_active_site_small.pdb",
        "contig": "50-80R,/0,100-120,/0,C1-4,C79-86",
        "length": "162-212",
        "ligand": "MG,PO4",
        "unindex": "B49,B50,B51,B52,B321,/0,A56-58,/0",
        "select_fixed_atoms": {
            "B49": "ALL",
            "B50": "ALL",
            "B51": "ALL",
            "B52": "ALL",
            "B321": "ALL",
            "A56-58": "ALL",
            "C1-4": "ALL",
            "C79-86": "ALL"
        }
    }
}
```
Key Features

Mixed RNA + protein + fixed fragments

`unindex`: removes residues from positional indexing

`select_fixed_atoms`: freezes specified atoms

Ligands (MG, PO4) included in design context

Useful for catalytic residues or structural motifs

## 7. Summary of New Features

R / D suffix → RNA / DNA specification in contigs

`ss_dbn` → global secondary structure constraint (optional)

`ss_dbn_dict` → local secondary structure constraints (optional)

`paired_region_list` → helix-level pairing constraints (optional)

`paired_position_list` → base-level pairing constraints (optional)

ligand + selection options → aptamer design

`unindex` → remove residues from indexing

`select_fixed_atoms` → freeze structural elements


---

