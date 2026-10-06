Attention routing with stop-gradient barriers for a self-supervised world-model latent, from https://arxiv.org/abs/2609.30667.

Run the tests:

    JAX_PLATFORMS=cpu python -m pytest -q

Run the demo:

    JAX_PLATFORMS=cpu python demo.py

No DreamerV3, no DeepMind Control, no video backgrounds, no returns. The encoder is a small conv on 16x16 synthetic frames. The claim checked here is the routing math and the stop-gradient isolation, not the paper's DMC scores.

The router is cross-attention over time: a learned query scores each latent, softmax yields alpha, and z_att is the weighted value projection. Reconstruction decodes stop_gradient(z_att) into an 8x8 patch, so that loss does not train the router. Inverse dynamics and InfoNCE read z_att and stop_gradient of the decoder output, so they do not train the content decoder. The encoder is a linear strided conv on the synthetic frames, trained by reconstruction through a straight-through residual. It is not a Dreamer encoder.
