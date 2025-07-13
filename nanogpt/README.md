## Training NanoGPT with enforced weight norms

Setup from [Modded NanoGPT](https://github.com/KellerJordan/modded-nanogpt):

```
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
# downloads only the first 800M training tokens to save time
python data/cached_fineweb10B.py 8
./run.sh
```

Some training parameters you can play with:
1. `Hyperparameters` class on line 522:
    - `w_max` controls the maximum spectral norm of weight matrices
    - `emb_w_max` controls the RMS norm of embedding matrix columns
    - `lm_head_w_max` controls the spectral norm of the unembedding matrix
2. Optimizer init on lines 631-650 (set lr for each part of the model)

The default setting is a very conservative `w_max = 1`, which reaches 21\% validation accuracy after 6200 steps. Try `w_max = 8`.

The GPT architecture follows Modula parameterization, which means:
    * Residual connections are defined like `x + 1/L * block(x)`, where `L` is the number of layers
    * Attention is defined like `softmax(QK^T / d)`, using `1/d` scaling rather than `1/sqrt(d)`

The Modula parameterization is responsible for a significant part of lowering the Lipschitz constant. The other part is weight norm constraints.

The default method used is spectral capping (`train_spectral_cap.py`). You can try spectral normalization from [Miyato et al. 2018](https://arxiv.org/abs/1802.05957) (`train_spectral_normalize.py`). The relevant change is around line 260 in the Muon update step. The `w_max` hyperparameter enforces a max spectral norm in either case, but using a different method.