import jax
import optax
from typing import Any, Callable, Sequence
import jax.numpy as jnp
from jax import random
import flax
from flax import linen as nn
from  flax import serialization

# We create one dense layer instance (taking 'features' parameter as input)
model = nn.Dense(features=5)
# key = random.PRNGKey(17)
key1, key2 = random.split(random.PRNGKey(17))

x = random.normal(key1, (10,)) # Dummy input data
params = model.init(key2, x) # Initialization call
print(jax.tree_util.tree_map(lambda x: x.shape, params)) # Checking output shapes
model.apply(params, x)# 模型每次执行前向传播时需要使用apply方法传参数
# Set problem dimensions.
n_samples = 20
x_dim = 10
y_dim = 5

# Generate random ground truth W and b.
key = random.PRNGKey(42)
k1, k2 = random.split(key)
# print(k1)
W = random.normal(k1, (x_dim, y_dim))
b = random.normal(k2, (y_dim,))
# Store the parameters in a FrozenDict pytree.
#通常，在定义模型和训练之前，会使用类似的方式创建和冻结参数 避免参数在训练过程被修改保持稳定性
true_params = flax.core.freeze({'params': {'bias': b, 'kernel': W}})

# Generate samples with additional noise.
key_sample, key_noise = random.split(k1)
x_samples = random.normal(key_sample, (n_samples, x_dim))
y_samples = jnp.dot(x_samples, W) + b + 0.1 * random.normal(key_noise,(n_samples, y_dim))
print('x shape:', x_samples.shape, '; y shape:', y_samples.shape)

# Same as JAX version but using model.apply().
@jax.jit
def mse(params, x_batched, y_batched):
  # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
        pred = model.apply(params, x)
        #inner计算了误差向量的内积（点积），也就是两个向量的逐元素乘积之和。这个操作得到了误差向量的平方和
        return jnp.inner(y-pred, y-pred) / 2.0
  # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched,y_batched), axis=0)

learning_rate = 0.3  # Gradient step size.
print('Loss for "true" W,b: ', mse(true_params, x_samples, y_samples))
loss_grad_fn = jax.value_and_grad(mse)

@jax.jit
def update_params(params, learning_rate, grads):
    params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, params, grads)
    return params

for i in range(101):
    # Perform one gradient update.
    loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
    params = update_params(params, learning_rate, grads)
    if i % 10 == 0:
        print(f'Loss step {i}: ', loss_val)

tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(mse)

for i in range(101):
  loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  if i % 10 == 0:
    print('Loss step {}: '.format(i), loss_val)

bytes_output = serialization.to_bytes(params)
dict_output = serialization.to_state_dict(params)
print('Dict output')
print(dict_output)
print('Bytes output')
print(bytes_output)

serialization.from_bytes(params, bytes_output)

class ExplicitMLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x
key = random.PRNGKey(17)
key1, key2 = random.split(key, 2)
x = random.uniform(key1, (4,4))

model = ExplicitMLP(features=[3,4,5])
params = model.init(key2, x)
y = model.apply(params, x)
print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, flax.core.unfreeze(params)))
print('output:\n', y)

try:
    y = model(x) # Returns an error
except AttributeError as e:
    print(e)

class SimpleMLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f'layers_{i}')(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
      # providing a name is optional though!
      # the default autonames would be "Dense_0", "Dense_1", ...
        return x
key = random.PRNGKey(17)
key1, key2 = random.split(key, 2)
x = random.uniform(key1, (4,4))

model = SimpleMLP(features=[3,4,5])
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, flax.core.unfreeze(params)))
print('output:\n', y)

