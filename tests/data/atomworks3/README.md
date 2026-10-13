# AtomWorks 2.2.1 model-feature references

The two compressed NumPy archives were generated from public Foundry PR #397
revision `754d1caae5a74b35a3bfcf68169e459474b7dbc4`, before the AtomWorks 3 port,
using AtomWorks 2.2.1, Biotite 1.4.0, Torch 2.11.0, and seed 123 for Python,
NumPy, and Torch. Each stores all 41 entries of the preprocessing `feats` dict,
with tensors converted to CPU NumPy arrays. No object pickles are stored.

The exact specifications and pipeline arguments are in
`test_model_features_match_atomworks221_reference` in `tests/test_atomworks3.py`.
RFD3 uses four protein residues. RFD3NA uses four RNA, four DNA, and four protein
residues, with the repository's AMP input fixture. Conformer generation is
disabled and the ground-truth conformer policy is selected to isolate changes in
preprocessing from RDKit conformer generation.

To regenerate, execute that specification/pipeline on the reference revision in
the reference environment and save `actual['feats']` with `np.savez_compressed`,
converting tensors with `value.cpu().numpy()`. Do not regenerate from the port
under test. Raw AtomArray bond graphs are intentionally excluded; actual model
token-bond features are included. See `docs/source/atomworks3_port.md` for known
AtomWorks 3 semantic differences and checkpoint validation.
