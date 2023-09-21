import pandas as pd
from flax import linen as nn  # Linen API
import jax
import jax.numpy as jnp
import  matplotlib.pyplot as plt
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax
import tensorflow_datasets as tfds  # TFDS for MNIST
import tensorflow as tf             # TensorFlow operations
# Common loss functions and optimizers
data = pd.DataFrame('./average.csv')
class MLP(nn.Module):                    # create a Flax Module dataclass
  out_dims: int

  @nn.compact
  def __call__(self, x):
    x = x.reshape((1, -1))
    x = nn.Dense(128)(x)                 # create inline Flax Module submodules
    x = nn.relu(x)
    x = nn.Dense(self.out_dims)(x)       # shape inference
    return x

model = MLP(out_dims=10)                 # instantiate the MLP model
key = jax.random.PRNGKey(17)
x = jnp.empty((4, 28, 28, 1))            # generate random data
variables = model.init(jax.random.PRNGKey(42), x)# initialize the weights
y = model.apply(variables, x)
plt.plot(x, y, "r-", label="real")

plt.legend(loc="upper left")
plt.grid(True)
plt.show()
# make forward pass



