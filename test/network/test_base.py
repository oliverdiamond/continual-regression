import jax
import jax.numpy as jnp
from flax import nnx

from src.network.base import BaseNetwork


def test_number_of_weights():
    model = BaseNetwork(rngs=nnx.Rngs(0))
    state = nnx.state(model)
    # https://github.com/jax-ml/jax/discussions/6153#discussioncomment-7619878
    number_of_weights = sum(x.size for x in jax.tree_util.tree_leaves(state))
    assert number_of_weights == 722_945


def test_call():
    model = BaseNetwork(rngs=nnx.Rngs(0))
    x = jnp.ones((1, 384))
    y = model(x)
    assert y.shape == (1, 1)
