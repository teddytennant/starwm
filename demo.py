"""Five CPU steps on one synthetic episode. Prints losses and attention."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import optax

from starwm.losses import init_params, train_step
from starwm.routing import route
from starwm.synthetic import downsample_video, encode, make_episode


def main():
    video, actions = make_episode(0)
    batch = {
        "video": video[None, ...],
        "actions": actions[None, ...],
        "target": downsample_video(video[None, ...]),
    }
    params = init_params(jax.random.PRNGKey(0))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    for step in range(5):
        params, opt_state, aux = train_step(params, opt_state, batch, optimizer)
        print(
            f"step {step} total={float(aux['total']):.6f} "
            f"recon={float(aux['recon']):.6f} "
            f"inverse={float(aux['inverse']):.6f} "
            f"contrastive={float(aux['contrastive']):.6f}"
        )
    z = encode(params["encoder"], batch["video"])
    _z_att, alpha = route(params["router"], z)
    weights = [f"{float(w):.4f}" for w in alpha[0]]
    print("attention", " ".join(weights))


if __name__ == "__main__":
    main()
