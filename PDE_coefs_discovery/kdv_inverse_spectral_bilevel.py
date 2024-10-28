import os
import haiku as hk

import jax
import optax
import jax.numpy as jnp

from functools import partial
from jax.experimental.jet import jet
import jax.experimental.ode as jode
from tqdm import tqdm
import argparse
import scipy
import numpy as np
import copy
from utils.interpolate2d import bispline_interp as interp2d

jax.config.update("jax_enable_x64", True)

from torch import nn

def set_seed(seed):
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    return key

def load_data(path):
    keys = ['x', 'tt', 'uu']
    data = scipy.io.loadmat(path)
    xi = [np.real(data[k]).reshape((-1, 1)) for k in keys[:-1]]
    raw = np.real(data[keys[-1]]).T
    shape = np.real(data[keys[-1]]).T.shape
    u = np.real(data[keys[-1]]).T.reshape((-1))
    x = np.concatenate([xx.reshape((-1, 1)) for xx in np.meshgrid(*xi)], axis=1)
    return x[:, 0], x[:, 1], u, shape # x, t, u

class InversePDESolver:
    def __init__(self, args, key):
        self.args = args
        self.key = key
        self.path = '/disk3/yezhen/Experiments/BilevelInverse/Data/KdV.mat'
        self.data_x, self.data_t, self.data_u, self.data_mesh_shape = load_data(self.path)
        # self.training_x, self.training_t, self.training_u = self.sampling_data_points(self.args.number_data_points)
        self.training_x, self.training_t, self.training_u = self.sampling_data_points(
            self.args.number_data_points, 
            self.args.random_or_grid, 
            self.args.noise
        )
        #----- Constructing the grid -----
        L   = 2.
        self.nx  = self.args.N_x
        x   = jnp.linspace(-1., 1 , self.nx+1)
        self.X   = x[:self.nx]
        self.t0  = 0; self.tf = 1
        self.nt  = self.args.N_t
        self.T = jnp.linspace(self.t0, self.tf, self.nt)
        kx1 = jnp.linspace(0, self.nx//2-1,self.nx//2)
        kx2 = jnp.linspace(1, self.nx//2,  self.nx//2)
        kx2 = -1*kx2[::-1]
        self.k  = (2.* jnp.pi/L)*jnp.concatenate((kx1,kx2))

        #----- Initialize the learnbale parameters -----
        self.lamb = {'coefs': {'lamb': jnp.ones(1) * args.lamb_init}}
        self.nu = {'coefs': {'nu': jnp.ones(1) * args.nu_init}}

        lamb_linear_decay_scheduler = optax.linear_schedule(
            init_value=args.lamb_lr, end_value=0,
            transition_steps=args.iterations,
            transition_begin=0
        )
        self.lamb_optimizer = optax.adam(lamb_linear_decay_scheduler)
        self.lamb_opt_state = self.lamb_optimizer.init(self.lamb)

        nu_linear_decay_scheduler = optax.linear_schedule(
            init_value=args.nu_lr, end_value=0,
            transition_steps=args.iterations,
            transition_begin=0
        )
        self.nu_optimizer = optax.adam(nu_linear_decay_scheduler)

        self.nu_opt_state = self.nu_optimizer.init(self.nu)
        self.predict_fn = jax.vmap(self.predict, in_axes=(None, None, None, None, 0, 0, None, None))

        #----- Constructing the grid for quadrature -----
        if self.args.learnable_lamb:
            vi = jnp.linspace(-5, 5, self.args.number_rs)
            vj = jnp.linspace(-5, 5, self.args.number_rs)
            vi, vj = jnp.meshgrid(vi, vj)
            self.v = jnp.hstack([vi.reshape(-1, 1), vj.reshape(-1, 1)]) # (number_rs ** 2, 2)
            sum_square = - jnp.sum(self.v ** 2, axis=1) / 2
            self.sftmax = jax.nn.softmax(sum_square, axis=0) # number_rs
            print(self.sftmax.sum())
        else:
            self.v = jnp.linspace(-5, 5, self.args.number_rs).reshape(-1, 1)
            sum_square = - jnp.sum(self.v ** 2, axis=1) / 2
            self.sftmax = jax.nn.softmax(sum_square, axis=0)

    #----- Change of Variables -----
    @partial(jax.jit, static_argnums=(0,))
    def _uhat2vhat(self, uhat, t, k, delta):
        return jnp.exp(-1j * (k**3) * (delta) * t) * uhat

    @partial(jax.jit, static_argnums=(0,))
    def _vhat2uhat(self, vhat, t, k, delta):
        return jnp.exp(1j * (k**3) * delta * t) * vhat

    @partial(jax.jit, static_argnums=(0,))
    def _vhatprime(self, vhat, t, k, lamb, delta):
        u = jnp.fft.ifft(self._vhat2uhat(vhat, t, k, delta))
        return  -0.5j * k * lamb * self._uhat2vhat(jnp.fft.fft(u**2), t, k, delta)

    def predict(self, vhat0, T, k, lamb, delta):
        V_hat = jode.odeint(
            self._vhatprime, vhat0, T, k, lamb, delta, 
            rtol=1.4e-8, atol=1.4e-8, 
            mxstep=2000, hmax=jnp.inf
        )
        vhat2uhat_fn = jax.vmap(self._vhat2uhat, in_axes=(0, 0, None, None))
        U = jnp.fft.ifft(vhat2uhat_fn(V_hat, T, k, delta), axis=1).real
        return U
        
    # @partial(jax.jit, static_argnums=(0,))
    def step(self, lamb, nu):
        #------ Constructing the grid for rs quadrature -----
        if self.args.learnable_lamb:
            stacked_coefs = jnp.hstack([lamb['coefs']['lamb'], nu['coefs']['nu']]).reshape(1, 2)
            rs_coefs_plus = stacked_coefs + self.v * self.args.epsilon
            rs_coefs_collection = jnp.concatenate([stacked_coefs, rs_coefs_plus], axis=0) # (number_rs ** 2 + 1 , 1)
        else:
            stacked_nu = nu['coefs']['nu'].reshape(1, 1)
            rs_nu_plus = stacked_nu + self.v * self.args.epsilon
            rs_nu_collection = jnp.concatenate([stacked_nu, rs_nu_plus], axis=0) 
            stacked_lamb = lamb['coefs']['lamb'].reshape(1, 1).repeat(self.args.number_rs + 1, axis=0)
            rs_coefs_collection = jnp.concatenate([stacked_lamb, rs_nu_collection], axis=1) # (number_rs + 1 , 1)
        #------ Initial conditions -----
        u0      = jnp.cos(jnp.pi*self.X)
        uhat0   = jnp.fft.fft(u0)
        vhat0 = self._uhat2vhat(uhat0, self.t0, self.k, nu['coefs']['nu'])
        #------ Solving for ODE -----
        rs_preditced_grid = jax.vmap(self.predict, in_axes=(None, None, None, 0, 0))(
            vhat0, self.T, self.k, 
            rs_coefs_collection[:, 0], rs_coefs_collection[:, 1]
        ) # (number_rs ** 2 + 1, N_t, N_x) or (number_rs + 1, N_t, N_x)
        #------ Compute the du/dcoef -----
        sample_size = self.args.number_rs ** 2 if self.args.learnable_lamb else self.args.number_rs
        v_dim = 2 if self.args.learnable_lamb else 1
        self.predicted_grid = rs_preditced_grid[0]
        intep2d_fn = jax.vmap(interp2d, in_axes=(None, None, None, None, 0))
        positive_grids = rs_preditced_grid[1:sample_size + 1]
        pos_intep_u = intep2d_fn(self.training_t, self.training_x, self.T, self.X, positive_grids)
        intep_u = interp2d(self.training_t, self.training_x, self.T, self.X, self.predicted_grid)
        # print(intep_u.shape, pos_intep_u.shape)
        gradients = (pos_intep_u - intep_u.reshape(1, -1)) / (self.args.epsilon)
        rs_gradients = jax.vmap(jnp.matmul, (0, 0))(
            gradients.reshape(sample_size, -1, 1),  
            self.v.reshape(sample_size, 1, v_dim)
        )
        du_dcoef = jnp.sum(rs_gradients * self.sftmax.reshape(-1, 1, 1), axis=0) # # col_points
        #------ Compute the dL/du -----
        def get_loss_data(intep_u, u):
            u, intep_u = u.reshape(-1), intep_u.reshape(-1)
            mse_u = jnp.mean((u - intep_u) ** 2)
            return mse_u
        dL_du = jax.jacrev(get_loss_data, argnums=0)(intep_u, self.training_u).reshape(1, -1)
        #------ Compute the dL/dcoef -----
        dL_dcoef = jnp.matmul(dL_du, du_dcoef)
        #------ Update the learnable parameters -----
        if self.args.learnable_lamb:
            lamb_gradients = {'coefs': {'lamb': dL_dcoef.reshape(2)[0]}}
            nu_gradients = {'coefs': {'nu': dL_dcoef.reshape(2)[1]}}
            lamb_updates, self.lamb_opt_state = self.lamb_optimizer.update(lamb_gradients, self.lamb_opt_state)
            nu_updates, self.nu_opt_state = self.nu_optimizer.update(nu_gradients, self.nu_opt_state)
            lamb = optax.apply_updates(lamb, lamb_updates)
            nu = optax.apply_updates(nu, nu_updates)
            return lamb, nu, (jnp.mean((intep_u - self.training_u)) ** 2)
        else:
            nu_gradients = {'coefs': {'nu': dL_dcoef.reshape(1)[0]}}
            nu_updates, self.nu_opt_state = self.nu_optimizer.update(nu_gradients, self.nu_opt_state)
            nu = optax.apply_updates(nu, nu_updates)
            return nu, (jnp.mean((intep_u - self.training_u)) ** 2)

    def sampling_data_points(self, data_size, random_or_grid='random', noise=0.0):
        if random_or_grid == 'random':
            # idx = np.random.choice(np.arange(len(self.data_x)), data_size, replace=False)
            idx = np.load('caches/kdv_' + str(data_size)+'_random_idx.npy')
            # np.save("random_idx.npy", idx)
        else:
            # grid_size = int(np.sqrt(data_size))
            grid_size = data_size
            t_grid_nums = self.data_mesh_shape[0]
            x_grid_nums = self.data_mesh_shape[1]
            t_points = np.linspace(0, t_grid_nums - 1, grid_size + 2, dtype=int)[1:-1, None]
            x_points = np.linspace(0, x_grid_nums - 1, grid_size + 2, dtype=int)[None, 1:-1]
            idx = (t_points * x_grid_nums + x_points).reshape(-1)
        x, t, u = self.data_x[idx], self.data_t[idx], self.data_u[idx]
        u = u + noise * np.std(u) * np.random.randn(*u.shape)
        return x, t, u
    
    # @jax.jit
    def train(self):
        current_nu = np.inf
        count = 0
        f = open("/disk3/yezhen/Experiments/BilevelInverse/logs/kdv/" + self.args.log_file + ".txt", "a")
        for n in tqdm(range(self.args.iterations)):
            if self.args.learnable_lamb:
                _, self.nu, loss = self.step(self.lamb, self.nu)
            else:
                self.nu, loss = self.step(self.lamb, self.nu)
            
            lamb, nu = self.lamb['coefs']['lamb'][0], self.nu['coefs']['nu'][0]
            interpolated_U = interp2d(self.data_t, self.data_x, self.T, self.X, self.predicted_grid)
            L2 = jnp.sum((interpolated_U - self.data_u)**2) ** 0.5 / jnp.sum(self.data_u ** 2)**0.5
            str = 'epoch %d, loss: %e, L2: %e, lamb_est: %e, delta2_est: %e' % (
                    n, 
                    loss, 
                    L2,
                    lamb, 
                    nu
                )
            print(str)
            f.write(str + '\n')
            if abs(current_nu - nu) < 1e-7: # if the nu is not changing for 10 times, then break
                count += 1
            
            if n == self.args.iterations - 1 or np.isnan(nu).any() or count > 10:
                nu = self.nu['coefs']['nu'][0]
                f.write('data_num %d, loss: %e, L2: %e, noise: %e, nu_init: %e, nu_est: %e, iter: %d \n' % (
                    self.args.number_data_points, loss, L2, self.args.noise, self.args.nu_init, nu, n + 1
                    )
                )
                f.close()
                break
            current_nu = self.nu['coefs']['nu'][0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # PINN Data
    parser.add_argument('--number_data_points', type=int, default=5, help='number of interior datapoints')

    parser.add_argument('--N_x', type=int, default=128, help='number of spatial grid size')
    parser.add_argument('--N_t', type=int, default=512, help='number of temporal grid size')
    # Optimization params - LBFGS
    parser.add_argument('--lamb_lr', type=float, default=1e-3, help='lamb learning rate')
    parser.add_argument('--nu_lr', type=float, default=1e-3, help='nu learning rate')
    parser.add_argument('--lamb_init', type=float, default=1, help='lamb init')
    parser.add_argument('--nu_init', type=float, default=1e-3, help='nu init')
    parser.add_argument('--number_rs', type=int, default=12, help='learning rate')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='epsilon')
    parser.add_argument('--iterations', type=int, default=500, help='number of lbfgs steps')
    parser.add_argument('--learnable_lamb', default=False, action='store_true')
    parser.add_argument('--log_file', type=str, default="kdv_bilevel", help='log file name')
    parser.add_argument('--noise', type=float, default=0.0, help='noise magnitude of data points')
    parser.add_argument('--random_or_grid', type=str, default='random', help='random or grid sampling')
    # Others
    args = parser.parse_args()
    print(args)
    key = set_seed(args.seed)
    model = InversePDESolver(args, key)
    model.train()