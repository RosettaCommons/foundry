# Folding with an MSA

```{important}
RF3 supports `.a3m` and `.fasta` files as input MSA formats; `.a3m` is recommended. We do not at the moment support pre-paired MSAs (we will pair on-the-fly) or on-the-fly MSA computation, but both are on the roadmap. 

Please [create a new Issue](https://github.com/RosettaCommons/foundry/issues/new/choose) if these limitations are critical for your project and we can prioritize accordingly.
```

Here's an example for using RF3 to generate possible folds with MSA information:
```{note}
The JSON file below is provided in `models/rf3/docs/examples/3en2_from_json_with_msa_.json`. The paths in the file assume that you will run these examples from the main `foundry` directory.
```
```json
{
    "name": "3en2_from_json_with_msa",
    "components": [
        {
            "seq": "AINRLQLVATLVEREV(MSE)RYTPAGVPIVNCLLSYSGQA(MSE)EAQAARQVEFSIEALGAGK(MSE)ASVLDRIAPGTVLECVGFLARKHRSSKALVFHISGLEHHHHHH",
            "chain_id": "A",
            "msa_path": "models/rf3/docs/examples/msas/3en2_A.a3m.gz"
        },
        {
            "seq": "AINRLQLVATLVEREV(MSE)RYTPAGVPIVNCLLSYSGQA(MSE)EAQAARQVEFSIEALGAGK(MSE)ASVLDRIAPGTVLECVGFLARKHRSSKALVFHISGLEHHHHHH",
            "chain_id": "B",
            "msa_path": "models/rf3/docs/examples/msas/3en2_A.a3m.gz"
        }
    ]
}
```

