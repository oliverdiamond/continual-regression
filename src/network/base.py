import jax.numpy as jnp
from flax import nnx


class BaseNetwork(nnx.Module):
    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        num_input_unit: int = 384,
        num_hidden_layer: int = 2,
        num_hidden_units: int = 512,
        num_output_unit: int = 1,
    ):
        self.linear_input = nnx.Linear(num_input_unit, num_hidden_units, rngs=rngs)
        self.linear_hiddens = [
            nnx.Linear(num_hidden_units, num_hidden_units, rngs=rngs) for _ in range(num_hidden_layer)
        ]
        self.linear_output = nnx.Linear(num_hidden_units, num_output_unit, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear_input(x))
        for linear_hidden in self.linear_hiddens:
            x = nnx.relu(linear_hidden(x))
        x = self.linear_output(x)
        x = jnp.squeeze(x)
        return x
