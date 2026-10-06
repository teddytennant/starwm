"""Twenty CPU steps on synthetic frames. Prints losses and a spatial attention peak."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import optax

from starwm.losses import init_params, train_step
from starwm.routing import H, W, attention
from starwm.synthetic import encode, make_batch


def main():
    batch = make_batch(0, batch=4)
    params = init_params(jax.random.PRNGKey(0))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(1)
    last = None
    for step in range(20):
        params, opt_state, aux = train_step(
            params, opt_state, batch, optimizer, jax.random.fold_in(key, step)
        )
        last = aux
        print(
            f"step {step} total={float(aux['total']):.6f} "
            f"recon={float(aux['recon']):.6f} "
            f"inverse={float(aux['inverse']):.6f} "
            f"contrastive={float(aux['contrastive']):.6f} "
            f"kl={float(aux['kl']):.6f}"
        )
    features = encode(params["encoder"], batch["video"])
    attn = attention(params["router"], features)
    peak = int(jnp_argmax(attn[0, 0, 0]))
    print(f"query0 frame0 peak token {peak} of {H * W} (cell {peak // W},{peak % W})")
    print(f"final total {float(last['total']):.6f}")


def jnp_argmax(x):
    import jax.numpy as jnp

    return jnp.argmax(x)


if __name__ == "__main__":
    main()
