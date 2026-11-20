# N-to-C macrocycle design

Use `cyclic_chains` to request cyclic residue relative positional encoding for
one complete de novo canonical peptide. This follows the approach of
[RFpeptides](https://doi.org/10.1038/s41589-025-01929-w) and the cyclic encoding
implemented in RF3. RFD3 retains its existing weights and sampler.

## Monomers

[macrocycle_monomer.json](macrocycle_monomer.json) contains separate 10- and
12-residue designs. Run from the repository root:

```bash
rfd3 design inputs=models/rfd3/docs/examples/macrocycle_monomer.json \
    out_dir=outputs/macrocycle_monomers ckpt_path=/path/to/rfd3_latest.ckpt \
    diffusion_batch_size=1 n_batches=1
```

This requests one backbone per input key. Increase `n_batches` or
`diffusion_batch_size` using the usual inference controls for larger runs.
You can also specify a sampled range, for example `"length": "10-12"`.

## Protein binders

[macrocycle_binder.json](macrocycle_binder.json) requests a 12-residue peptide
against the insulin receptor using the existing example structure and a hotspot:

```bash
rfd3 design inputs=models/rfd3/docs/examples/macrocycle_binder.json \
    out_dir=outputs/macrocycle_binders ckpt_path=/path/to/rfd3_latest.ckpt \
    diffusion_batch_size=1 n_batches=1
```

The contig `12,/0,E6-155` builds peptide chain `A` and target chain `B`, so
`"cyclic_chains": ["A"]` selects the peptide. With target-first contig
`E6-155,/0,12`, select `["B"]`. These are assembled chain IDs; `select_hotspots`
still uses source-file IDs such as `E64`. A target's unrelated cofactor does not
prevent cyclic encoding of the peptide.

## Behavior and limits

Omitting `cyclic_chains`, using `null`, or using `[]` keeps linear encoding.
The option changes only residue offsets within the selected peptide; target and
interchain encodings retain their existing behavior. Half-ring ties in even-length
peptides retain the original signed offset, as in RF3.

The selected chain must be entirely de novo canonical peptide with one token per
residue. Motif-containing rings, multiple cyclic selections, partial diffusion,
and active symmetry are unsupported. Unrelated components follow existing input
rules. See [the input reference](../input.md#cyclic-peptides).

Cyclic encoding does not enforce geometric closure or add a terminal C–N bond
to model inputs or written CIF files. The request is saved in normal output JSON.
Evaluate closure and scientific quality separately.
