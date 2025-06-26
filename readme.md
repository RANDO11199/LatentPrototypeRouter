# Latent Prototype Routing for Mixture-of-Experts (LPR)

## Installation

### To Replicate the Results in the Paper
We modified components of **Mixtral**, **Qwen3Moe**, and **DeepseekV3** to fit the LPR framework. Some basic building blocks such as `DecoderLayer` and `SparseMoeBlock` are not yet supported by the HuggingFace Transformers library. To reproduce our results:

1. Copy all files from `/ModifiedTransformsLibFile` into your local `transformers` library installation.  
   *Note:* These changes only adjust the import structure — the original behavior and logic of the library remain unchanged.

2. Download the **FineWeb** and **C4 validation** datasets, and place them into the `data` directory. The expected structure is as follows:
   ```text
   /data/c4val/c4-validation.*-of-00008.json.gz    # C4 validation set
   /data/fineweb/sample/100BT/*.parquet         # Fineweb sample-100BT data
   ```

3. Run `train.sh` to train the model and `eval_all.sh` to evaluate the model.

### To Use the Latent Prototype Router
If you only want to incorporate the router into your own PyTorch project:

1. Copy `router.py` and `Metrics.py` from `/model_zoo` into your project.
2. Import `TokenDistributionRouter` and the modified config modules, as well as `SparseMoeBlock`, `DecoderLayer`, `Model`, and `CausalLM` classes to enable the router and output the regularization losses.
3. See `model_zoo/Qwen3Moe.py` for a working example of this integration. Essentially, most of the modifications aim at outputting the LPR regularization loss; the gating module can simply be replaced with the new router.

## Result

Current results of models listed below are trained on Fineweb 100BT subset for 1 billion tokens.

| **Method**                 | **Test Loss** ↓ | **GINI** ↓ | **Min-Max** ↑     |
|----------------------------|------------------|-------------|--------------------|
| Mixtral-0.6B (128-8)       | **3.683**        | 0.635       | 3.33 × 10⁻⁶        |
| Mixtral-LPR-0.6B (w/o init)| 3.747           | **0.047**   | **0.649**           |
| DeepSeekV3-0.6B (128-8)    | **3.673**        | 0.790       | 6.41 × 10⁻⁹        |
| DeepSeekMoe-LPR (w/o init) | 3.720           | **0.036**   | **0.724**           |
| Qwen3Moe-0.6B (128-8)      | **3.666**        | 0.707       | 1.27 × 10⁻¹⁶       |
| Qwen3Moe-LPR (w/ init)     | 3.685           | 0.057       | 0.597              |
| Qwen3Moe-LPR (w/o init)    | 3.697           | **0.039**   | **0.696**           |


## License
This project is licensed under the [Apache License 2.0](./LICENSE), which allows both commercial and non-commercial use.  
Any derivative works or products must retain proper attribution to the original authors.
