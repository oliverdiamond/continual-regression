from typing import Any, Optional

import optax


def adam(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 0.0,
    *,
    nesterov: bool = False
):
    # Note: the GVF paper doesn't use AdamW when implementing weight decay.
    return optax.chain(
        optax.add_decayed_weights(weight_decay),
        optax.adam(
            learning_rate,
            b1,
            b2,
            eps,
            eps_root,
            mu_dtype,
            nesterov=nesterov
        ),
    )
