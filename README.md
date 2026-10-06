Spatial cross-attention router from [StarWM](https://arxiv.org/abs/2609.30667), with the stop-gradient split between where-to-attend and what-to-extract.

```
pip install -e .
python demo.py
python -m pytest
```

The router is eq. 5: keys are spatial tokens plus a 2D sinusoidal encoding, values append normalized coordinates, and N queries softmax over the H×W locations. Inverse dynamics and the logistic contrastive loss (eq. 7) read flatten(At Ft), so they train the queries. Reconstruction composes a foreground stream and a residual stream with stop_gradient(At) and the coverage mask (eq. 8), so it does not train the queries.

Dropped: DreamerV3 RSSM, the policy, DMC, distractor video, and multi-head attention. The encoder is a stride-4 convolution on 16×16 synthetic frames (a moving square), C=8, N=2. The residual stream is a 4-d Gaussian bottleneck, not the paper's auxiliary VAE. The pixel head stitches non-overlapping 4×4 patches. This does not reproduce the DMC numbers.
