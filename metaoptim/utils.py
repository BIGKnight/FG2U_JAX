import jax
from jax import random
import jax.numpy as jnp


def random_split_like_tree(rng_key, target=None, tree_structure=None):
    if tree_structure is None:
        tree_structure = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, tree_structure.num_leaves)

    return jax.tree_util.tree_unflatten(tree_structure, keys)


def tree_random_like(key, target, dist='normal'):
    keys_tree = random_split_like_tree(key, target)
    if dist in ['normal', 'gaussian']:
        random_vec_func = jax.random.normal
    elif dist in ['rademacher']:
        random_vec_func = lambda k, shape, dtype: random.randint(k, shape, 0, 2).astype(dtype) * 2.0 - 1.0
    else:
        raise ValueError('Unknown dist type for random vector')
    return jax.tree_map(
        lambda l, k: random_vec_func(k, l.shape, l.dtype),
        target,
        keys_tree,
    )

def trees_random_like(key, target, n, dist='normal'):
    keys = random.split(key, n)
    return jnp.array([tree_random_like(k, target, dist) for k in keys])

if __name__ == '__main__':
    target = [jax.numpy.array([1., 2.]),
              [jax.numpy.array([1., 2.]), jax.numpy.array([1., 2., 3.])],
              {'k': jax.numpy.array([1., 2.])}]
    key = jax.random.PRNGKey(0)
    print(trees_random_like(key, target, 3, dist='rademacher'))
