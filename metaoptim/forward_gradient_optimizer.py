import jax
import jax.numpy as jnp
import optax
from functools import partial
from jax import random

from jax.tree_util import Partial
from einops import rearrange
from .utils import trees_random_like

DEFAULT_EPS = 1e-4
from jax.example_libraries.optimizers import l2_norm


@partial(jax.jit, static_argnums=4)
def loss_func(params, images, labels, model_fwd, weight_decay=0):
    logits = model_fwd(params, images, True)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
    l2_penalty = weight_decay * l2_norm(params) ** 2
    loss += l2_penalty
    return loss, logits

@jax.jit
def acc(params, images, labels, model_fwd):
    logits = model_fwd(params, images, True)
    target_class = jnp.argmax(labels, axis=1)
    predicted_class = jnp.argmax(logits, axis=1)
    return jnp.mean(predicted_class == target_class)

def meta_loss_func(params, meta_batch, model_fwd):
    return loss_func(params, *meta_batch, model_fwd)[0]

def inner_loss_func(params, meta_params, meta_labels, model_fwd, weight_decay):
    return loss_func(params, meta_params, meta_labels, model_fwd, weight_decay)[0]

@partial(jax.jit, static_argnums=6)
def inner_step(params, opt_state, meta_params, meta_labels, model_fwd, opt_update, weight_decay):
    aux, grads = jax.value_and_grad(loss_func, argnums=0, has_aux=True)(
        params, meta_params, meta_labels, model_fwd, weight_decay
    )
    loss, logits = aux
    updates, new_opt_state = opt_update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss, logits

def body_func(i, args):
    (in_params, in_opt_state, meta_params, meta_labels, model_fwd, opt_update, weight_decay, mini_batch_indexes) = args
    random_indexes = mini_batch_indexes[i]
    mini_params = meta_params[random_indexes]
    mini_labels = meta_labels[random_indexes]
    in_params, in_opt_state, _, _ = inner_step(
        in_params, in_opt_state, mini_params, mini_labels, model_fwd, opt_update, weight_decay
    )
    return in_params, in_opt_state, meta_params, meta_labels, model_fwd, opt_update, weight_decay, mini_batch_indexes

def forward(params, opt_state, meta_params, meta_labels, meta_batch, num_inner_steps, model_fwd, opt_update, weight_decay, mini_batch_indexes):
    in_params, in_opt_state = params, opt_state
    in_params, _, _, _, _, _, _, _ = jax.lax.fori_loop(
        0, num_inner_steps, 
        body_func, 
        (in_params, in_opt_state, meta_params, meta_labels, model_fwd, opt_update, weight_decay, mini_batch_indexes)
    )
    meta_loss = meta_loss_func(in_params, meta_batch, model_fwd)
    return meta_loss

def call_jvp_fn(tangent, params, opt_state, meta_params, meta_labels, meta_batch, model_fwd, opt_update, num_inner_steps, weight_decay, mini_batch_indexes):
    def call_jvp(f, primal, tangent): return jax.jvp(f, (primal,), (tangent,))[1]
    fwd_fn = lambda mps: forward(params, opt_state, mps, meta_labels, meta_batch, num_inner_steps, model_fwd, opt_update, weight_decay, mini_batch_indexes)
    ws = jax.vmap(call_jvp, in_axes=(None, None, 0))(fwd_fn, meta_params, tangent)
    grd = jax.vmap(lambda w, v: w * v, in_axes=(0, 0))(ws, tangent)
    return grd, ws

def call_fun(f, *args, **kwargs):
    return f(*args, **kwargs)

@jax.jit
def meta_l1_norm(meta_params):
    return jnp.sum(jnp.abs(meta_params))

def neumman_series(alpha, K, Matrix):
    shape = Matrix.shape
    assert (len(shape) == 2) & (shape[0] == shape[1]), "Matrix must be square matrix"
    estimated_inverse = jnp.zeros_like(Matrix)
    Idt_matrix = jnp.eye(shape[0])
    estimated_inverse = jnp.eye(shape[0])
    for i in range(K):
        estimated_inverse = Idt_matrix + estimated_inverse @ (Idt_matrix - alpha * Matrix)
    return alpha * estimated_inverse

def coordinate_random_select(rng, vector_size, num_select):
    rng, key = random.split(rng)
    selected_indexes = jax.random.permutation(jax.random.PRNGKey(key), vector_size)[:num_select]
    return selected_indexes, rng


def hypergrad_approx(
        key, 
        num_vec_total, num_vec_vmap,
        params, opt_state,
        num_inner_steps,
        meta_params, meta_labels, meta_batch,
        model_fwd, opt_update,
        weight_decay,
        inner_batch_size,
        vec_dist,
        rank,
    ):
    vmap_times, leftover = divmod(num_vec_total, num_vec_vmap)
    vmap_times = vmap_times + bool(leftover)
    hypergrad_approx = jax.tree_map(lambda x: jnp.zeros_like(x), meta_params)
    mini_batch_indexes = jax.random.choice(key, len(meta_params), shape=(num_inner_steps, inner_batch_size), replace=True)

    if selected_indexes is not None:
        selected_indexes = selected_indexes.reshape(-1) # 320
        vector_size = jnp.prod(jnp.array(meta_params.shape))
        vs_basis_tmp = jnp.array(selected_indexes[:, None] == jnp.arange(vector_size), dtype=meta_params.dtype) # one-hot e1, e2, e3, ... [, vector_size]
        vs_basis = vs_basis_tmp.reshape(vmap_times, -1, *meta_params.shape)

    vs_list = []
    signal_dir_d_list = []
    for idx in range(vmap_times):
        key, key1 = jax.random.split(key, 2)
        if num_vec_vmap % rank != 0: raise ValueError('num_vec_vmap must be divisible by rank')
        num_vec_vmap_per_device = num_vec_vmap // rank

        vs_ori = trees_random_like(key1, meta_params, num_vec_vmap, dist=vec_dist) if selected_indexes is None else vs_basis[idx]
        vs = rearrange(vs_ori, '(r n) ... -> r n ...', r=rank, n=num_vec_vmap_per_device)

        calls_f = Partial(
            call_jvp_fn, 
            params=params, opt_state=opt_state, 
            meta_params=meta_params,
            meta_labels=meta_labels, meta_batch=meta_batch, 
            model_fwd=model_fwd, opt_update=opt_update,
            num_inner_steps=num_inner_steps, 
            weight_decay=weight_decay,
            mini_batch_indexes=mini_batch_indexes,
        )
        grd, signal_dir_d = jax.pmap(call_fun, in_axes=(None, 0))(calls_f, vs)
        g_approx = rearrange(grd, 'r n ... -> (r n) ...', r=rank, n=num_vec_vmap_per_device)
        signal_dir_d = rearrange(signal_dir_d, 'r n ... -> (r n) ...', r=rank, n=num_vec_vmap_per_device)
        vs_list.append(vs_ori)
        signal_dir_d_list.append(signal_dir_d)
        g_approx = jnp.mean(g_approx, axis=0)
        hypergrad_approx = jax.tree_map(lambda x, y: x + 1. / vmap_times * y, hypergrad_approx, g_approx)
    return hypergrad_approx

@jax.jit
def meta_grad_func(params, images, labels, model_fwd):
    grad = jax.grad(lambda params: loss_func(params, images, labels, model_fwd)[0])(params)
    return grad

@jax.jit
def inner_step_1(params, opt_state, meta_params, meta_labels, model_fwd, opt_update, weight_decay, random_indexes):
    mini_params = meta_params[random_indexes]
    mini_labels = meta_labels[random_indexes]
    aux, grads = jax.value_and_grad(loss_func, argnums=0, has_aux=True)(
        params, mini_params, mini_labels, model_fwd, weight_decay
    )
    loss, logits = aux
    updates, new_opt_state = opt_update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss, logits

def grad_func(
        key, 
        inner_params, inner_opt_state, 
        num_inner_steps, 
        meta_params, meta_labels, meta_batch, 
        model_fwd, opt_update,
        weight_decay,
        inner_batch_size
    ):
    params, opt_state = inner_params, inner_opt_state
    history = []
    for j in range(num_inner_steps):
        key, key_ = jax.random.split(key)
        random_indexes = jax.random.permutation(key, len(meta_params))[:inner_batch_size]
        params, opt_state, _, _ = inner_step_1(
            params, opt_state, 
            meta_params, meta_labels, 
            model_fwd, opt_update, 
            weight_decay,
            random_indexes
        )
        history.append(dict(params=params, opt_state=opt_state, random_indexes=random_indexes))

    c = jnp.zeros(meta_params.shape)
    d = meta_grad_func(params, *meta_batch, model_fwd)
    for j in range(2, num_inner_steps + 1):
        rec = history[-j]
        params, opt_state, random_indexes = rec['params'], rec['opt_state'], rec['random_indexes']
        _, vjp_pz_pz = jax.vjp(lambda p: inner_step_1(p, opt_state, meta_params, meta_labels, model_fwd, opt_update, weight_decay, random_indexes)[0], params)
        _, vjp_pz_px = jax.vjp(lambda x: inner_step_1(params, opt_state, x, meta_labels, model_fwd, opt_update, weight_decay, random_indexes)[0], meta_params)
        c = jax.tree_map(lambda x, y: x + y, c, vjp_pz_px(d)[0])
        d = vjp_pz_pz(d)[0]
    return c

def inner_loss_func(params, meta_params, meta_labels, model_fwd, weight_decay):
    return loss_func(params, meta_params, meta_labels, model_fwd, weight_decay)[0]



@partial(jax.jit, static_argnums=[5])
def vjp_func(v, params, meta_params, meta_labels, model_fwd, weight_decay):
    _, vjp_func = jax.vjp(
        lambda x: jax.grad(inner_loss_func, 0)(
            params, x, meta_labels, model_fwd, weight_decay
        ), 
        meta_params
    )
    g = vjp_func(v)[0]
    return g

@jax.jit
def v0_func(params, meta_batch, model_fwd):
    results = jax.grad(meta_loss_func, 0)(params, meta_batch, model_fwd)
    return results

def implicit_func1(a, b, alpha):
    return a - alpha * b

def implicit_func2(params, meta_params, meta_labels, model_fwd, weight_decay, alpha):
    func1 = lambda a, b: implicit_func1(a, b, alpha)
    results = jax.tree_map(
        func1, 
        params, 
        jax.grad(inner_loss_func, 0)(
            params, 
            meta_params, meta_labels, 
            model_fwd, 
            weight_decay,
        )
    )
    return results

def grad_implicit_func(
        rng, 
        inner_params, 
        meta_params, meta_labels, 
        meta_batch, 
        model_fwd, 
        alpha, num_max_steps, 
        principle, 
        weight_decay,
        inner_batch_size
    ):
    params = inner_params
    v0 = v0_func(params, meta_batch, model_fwd)
    v = v0        
    assert len(meta_params) % inner_batch_size == 0
    rng, key = random.split(rng)
    total_bs = len(meta_params) // inner_batch_size
    # mini_batch_indexes = jax.random.choice(key, len(meta_params), shape=(total_bs, inner_batch_size), replace=False) 
    if principle == 'neumann':
        for step in range(num_max_steps):
            func = Partial(
                implicit_func2, 
                meta_params=meta_params, meta_labels=meta_labels, 
                model_fwd=model_fwd,
                weight_decay=weight_decay,
                alpha=alpha
            )
            _, v = jax.jvp(func, (params,), (v,))
            v = jax.tree_map(lambda a, b: a + b, v, v0)
        v = jax.tree_map(lambda x: - alpha * x, v)
    elif principle == 'none':
        v = jax.tree_map(lambda x: -x, v)
    else:
        raise ValueError('Unknown implicit gradient computation principle: {}'.format(principle))
    
    g_list = []
    length = meta_params.shape[0]
    assert length % inner_batch_size == 0
    for i in range(length // inner_batch_size):
        mini_params = meta_params[i * inner_batch_size: (i + 1) * inner_batch_size]
        mini_labels = meta_labels[i * inner_batch_size: (i + 1) * inner_batch_size]
        mini_g = vjp_func(v, params, mini_params, mini_labels, model_fwd, weight_decay)
        g_list.append(mini_g)
    g = jnp.concatenate(g_list, axis=0)
    return g