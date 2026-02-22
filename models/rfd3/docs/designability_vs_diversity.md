# Designability vs. Diversity

When using RFdiffusion3 there is a balance between designability and diversity of generated structures. Increasing the diversity of the designs will lead to a greater number of novel folds, however, there will also be a larger portion of structures that have low confidence scores when refolded. 

Whether you are struggling to produce designable structures or you are looking to increase the diversity of the folds you see, here are a few settings to try changing: 
- **Low temperature sampling:**
    
    One can increase `inference_sampler.step_scale` and decrease `inference_sampler.gamma_0` to decrease the sampling space that RFdiffusion3 has access to, similar to what lowering the temperature does in physics-based design methods. These settings directly change how the RFdiffusion3 inference engine works, so these options are specified in the CLI, and are not options you specify in your input JSON or YAML file.
    
    Here are what these settings do:
    - `inference_sampler.step_scale`: Changing this value (default 1.5) changes the diffusion step size, or how much you go towards the most probable result. Increasing this setting will increase the designability of the output structures, as these are more probable, but will also decrease the diversity of the produced structures. 
    - `inference_sampler.gamma_0`: Changing this value (default 0.6) will change how much randomness there is at the beginning of an inference run. Decreasing this setting will increase the designability of the output structures as the reduced randomness will lead RFdiffusion3 to higher-probability structures. Increase this quantity to increase the diversity of designed structures.
- **`is_non_loopy` setting:**

    The `is_non_loopy` setting is a constraint on the designs RFdiffusion3 produces, which makes it a setting provided in a JSON/YAML file. If `True` it biases the model away from forming structures with many regions without a defined secondary structure. This will slightly decrease the diversity of structures that RFdiffusion3 produces while increasing the designability. 

Here are a few plots showing the impacts of these settings in protein-protein interface design tasks: 
```{note}
For the purposes of the plots below:
* `Low temperature` means a `step_scale` of 3 and a `gamma_0` of 0.2. 
* Pass rates are refolding pass rates, the number of backbones that pass after four attempts at designing the sequence using MPNN-based methods.
* 'Cluster' refers to `foldseek-based clusters <https://www.nature.com/articles/s41587-023-01773-0>`_, and the cluster pass rate is the number of clusters represented among the passing designs divided by the total number of designed backbones.

```

```{figure} ./.assets/400bb_rfd3_inference_settings_designability.png
:width: 800px

Impacts of using low temperature settings (inf) and the `is_non_loopy` constraint on the outputs of RFdiffusion3. 
```

</br>

---

```{figure} ./.assets/400bb_rfd3_inference_settings_diversity.png
:width: 800px

Diversity of folds in structures designed by RFD3 when using low temperature sampling and the `is_non_loopy` setting.
```

</br>

---

```{figure} ./.assets/400bb_rfd3_inference_settings_secondary_structure.png
:width: 800px

Compares the amount of alpha helices and beta sheets in structures designed by RFD3 when the low temperature sampling and `is_non_loopy` settings are used. The removal of the `is_non_loopy` setting results in a large reduction in α-helices and a small increase in the number of ß-sheets.
```