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
from jax_smi import initialise_tracking
initialise_tracking()

jax.config.update("jax_enable_x64", True)

from torch import nn

def set_seed(seed):
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    return key

def load_data(path):
    data = scipy.io.loadmat(path) # You can download data at https://github.com/maziarraissi/PINNs/tree/master/main/Data
    y = data['t'].flatten()
    x = data['x'].flatten() 
    shape = np.real(data['u']).shape
    u = np.real(data['u'].T)
    y, x = np.meshgrid(y, x) 
    x, y, u = x.reshape(-1), y.reshape(-1), u.reshape(-1)
    dataset = {'x': x, 'y': y, 'u': u}
    return x, y, u, shape

class InversePDESolver:
    def __init__(self, args, key):
        self.args = args
        self.key = key
        self.path = 'Data/Allen_Cahn.mat'
        self.data_x, self.data_t, self.data_u, self.data_mesh_shape = load_data(self.path)
        self.training_x, self.training_t, self.training_u = self.sampling_data_points(
            self.args.number_data_points, 
            self.args.random_or_grid, 
            self.args.noise
        )
        self.lamb = {'coefs': {'lamb': jnp.ones(1) * self.args.lamb_init}}
        self.nu = {'coefs': {'nu': jnp.ones(1) * np.sqrt(self.args.nu_init)}}
        N_x = self.args.N_x
        N_t = self.args.N_t
        dx = 2 / N_x
        dt = 1 / N_t
        self.X = jnp.linspace(-1, 1, N_x)
        self.T = jnp.linspace(0, 1, N_t)
        # u0 = -jnp.sin(jnp.pi * X)
        self.u0 = jnp.cos(jnp.pi * self.X) * (self.X ** 2)
        #Wave number discretization
        self.k = 2*jnp.pi*jnp.fft.fftfreq(N_x, d=dx)

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

    @partial(jax.jit, static_argnums=(0,))
    def _burger_system(self, u,t,k,lamb,nu):
        #Spatial derivative in the Fourier domain
        u_hat = jnp.fft.fft(u)
        u_hat_x = 1j*k*u_hat
        u_hat_xx = -k**2*u_hat
        
        #Switching in the spatial domain
        u_x = jnp.fft.ifft(u_hat_x)
        u_xx = jnp.fft.ifft(u_hat_xx)
        
        #ODE resolution
        u_t = nu ** 2 * u_xx + lamb * (u - u ** 3)
        return u_t.real

    def predict(self, u0, T, k, lamb, nu):
        predicted_grid = jode.odeint(
            self._burger_system, 
            u0, T, k, 
            lamb, nu, 
            rtol=1.4e-8, atol=1.4e-8, 
            mxstep=1000, hmax=jnp.inf
        )
        return predicted_grid
        
    # @partial(jax.jit, static_argnums=(0,))
    def step(self, lamb, nu):
        vi = jnp.linspace(-5, 5, self.args.number_rs)
        vj = jnp.linspace(-5, 5, self.args.number_rs)
        vi, vj = jnp.meshgrid(vi, vj)
        v = jnp.hstack([vi.reshape(-1, 1), vj.reshape(-1, 1)]) # number_rs ** 2, 2
        sum_square = - jnp.sum(v ** 2, axis=1) / 2
        sftmax = jax.nn.softmax(sum_square, axis=0) # number_rs ** 2
        stacked_coefs = jnp.hstack([lamb['coefs']['lamb'], nu['coefs']['nu']]).reshape(1, 2)
        rs_coefs_plus = stacked_coefs + v * self.args.epsilon # (number_rs ** 2, 2)
        rs_coefs_minus = stacked_coefs - v * self.args.epsilon # (number_rs ** 2, 2)
        rs_coefs_collection = jnp.concatenate([stacked_coefs, rs_coefs_plus, rs_coefs_minus], axis=0) # (2 * number_rs ** 2, 2)
        rs_preditced_grid = jax.vmap(self.predict, in_axes=(None, None, None, 0, 0,))(
            self.u0, self.T, self.k, 
            rs_coefs_collection[:, 0], rs_coefs_collection[:, 1]
        ) # (2 * number_rs ** 2 + 1, N_t, N_x)
        self.predicted_grid = rs_preditced_grid[0]
        positive_grids = rs_preditced_grid[1:self.args.number_rs ** 2 + 1]
        negative_grids = rs_preditced_grid[self.args.number_rs ** 2+1:2*self.args.number_rs ** 2+1]
        intep2d_fn = jax.vmap(interp2d, in_axes=(None, None, None, None, 0))
        pos_intep_u = intep2d_fn(self.training_t, self.training_x, self.T, self.X, positive_grids)
        neg_intep_u = intep2d_fn(self.training_t, self.training_x, self.T, self.X, negative_grids)
        gradients = (pos_intep_u - neg_intep_u) / (2 * self.args.epsilon)
        rs_gradients = jax.vmap(jnp.matmul, (0, 0))(
                gradients.reshape(self.args.number_rs ** 2, -1, 1),  
                v.reshape(self.args.number_rs ** 2, 1, 2)
            )
        du_dcoef = jnp.sum(rs_gradients * sftmax.reshape(-1, 1, 1), axis=0) # # col_points
        intep_u = interp2d(self.training_t, self.training_x, self.T, self.X, self.predicted_grid)

        def get_loss_data(intep_u, u):
            mse_u = jnp.mean((u - intep_u) ** 2)
            return mse_u
        
        dL_du = jax.jacrev(get_loss_data, argnums=0)(intep_u, self.training_u).reshape(1, -1)
        dL_dcoef = jnp.matmul(dL_du, du_dcoef)
        lambda_gradients = {'coefs': {'lamb': dL_dcoef.reshape(-1)[0]}}
        nu_gradients = {'coefs': {'nu': dL_dcoef.reshape(-1)[1] * 2 * nu['coefs']['nu'],}}
        lamb_updates, self.lamb_opt_state = self.lamb_optimizer.update(lambda_gradients, self.lamb_opt_state)
        nu_updates, self.nu_opt_state = self.nu_optimizer.update(nu_gradients, self.nu_opt_state)
        lamb = optax.apply_updates(lamb, lamb_updates)
        nu = optax.apply_updates(nu, nu_updates)
        return lamb, nu, (jnp.mean((intep_u - self.training_u)) ** 2)

    # def sampling_training_points(self, data_size, random_or_grid='random'):
    #     if random_or_grid == 'random':
    #         # idx = np.random.choice(np.arange(len(self.data_x)), data_size, replace=False)
    #         idx = np.load('caches/allen_cahn_' + str(data_size)+'_random_idx.npy')
    #     else:
    #         grid_size = int(np.sqrt(data_size))
    #         t_grid_nums = self.data_mesh_shape[0]
    #         x_grid_nums = self.data_mesh_shape[1]
    #         t_points = np.linspace(0, t_grid_nums - 1, grid_size + 2, dtype=int)[1:-1, None]
    #         x_points = np.linspace(0, x_grid_nums - 1, grid_size + 2, dtype=int)[None, 1:-1]
    #         idx = (t_points * x_grid_nums + x_points).reshape(-1)
    #     x, t = self.data_x[idx], self.data_t[idx]
    #     return x, t

    def sampling_data_points(self, data_size, random_or_grid='random', noise=0.0):
        if random_or_grid == 'random':
            # idx = np.random.choice(np.arange(len(self.data_x)), data_size, replace=False)
            # print(data_size, len(idx))
            # np.save("caches/allen_cahn_500_random_idx.npy", idx)
            idx = np.load('caches/allen_cahn_' + str(data_size)+'_random_idx.npy')
        else:
            grid_size = int(np.sqrt(data_size))
            t_grid_nums = self.data_mesh_shape[0]
            x_grid_nums = self.data_mesh_shape[1]
            t_points = np.linspace(0, t_grid_nums - 1, grid_size + 2, dtype=int)[1:-1, None]
            x_points = np.linspace(0, x_grid_nums - 1, grid_size + 2, dtype=int)[None, 1:-1]
            idx = (t_points * x_grid_nums + x_points).reshape(-1)
        x, t, u = self.data_x[idx], self.data_t[idx], self.data_u[idx]
        u = u + noise*np.std(u)*np.random.randn(*u.shape)
        return x, t, u

    # @jax.jit
    def train(self):
        current_nu = np.inf
        count = 0
        f = open("/disk3/yezhen/Experiments/BilevelInverse/logs/allen_cahn/" + self.args.log_file + ".txt", "w")
        for n in tqdm(range(self.args.iterations)):
            _, self.nu, loss = self.step(self.lamb, self.nu)
            lamb, nu = self.lamb['coefs']['lamb'][0], self.nu['coefs']['nu'][0] ** 2
            interpolated_U = interp2d(self.data_t, self.data_x, self.T, self.X, self.predicted_grid)
            current_L2 = jnp.sum((interpolated_U - self.data_u)**2) ** 0.5 / jnp.sum(self.data_u ** 2)**0.5
            str_tmp = 'epoch %d, loss: %e, L2: %e, lambda1_est: %e, delta2_est: %e' % (
                    n, 
                    loss, 
                    current_L2,
                    lamb, 
                    nu
                )
            print(str_tmp)
            f.write(str_tmp + '\n')
            if abs(current_nu - nu) < 1e-7: # if the nu is not changing for 10 times, then break
                count += 1
            # flag = (nu - 1e-3) / (1e-3)
            # if n == self.args.iterations - 1 or np.isnan(nu).any() or abs(flag) < 0.001 or count > 10:
            if n == self.args.iterations - 1 or np.isnan(nu).any() or count > 10:
                # f = open("/home/yezhen/Experiments/BilevelInverse/logs/allen_cahn/"+ self.args.log_file + ".txt", "a")
                f.write('grid_size %d, loss: %e, L2: %e, noise: %e, nu_init: %e, nu_est: %e, iters: %d \n' % (
                    self.args.number_data_points, loss, current_L2, self.args.noise, self.args.nu_init, nu, self.args.iterations
                    )
                )
                f.close()
                break
            current_nu = nu
            # print('epoch %d, lamb_est: %e, nu_est: %e' % (n, lamb, nu))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # PINN Data
    parser.add_argument('--number_data_points', type=int, default=4, help='number of interior datapoints')
    parser.add_argument('--N_x', type=int, default=512, help='number of spatial grid size')
    parser.add_argument('--N_t', type=int, default=1024, help='number of temporal grid size')
    # Optimization params - LBFGS
    parser.add_argument('--lamb_lr', type=float, default=1e-1, help='lamb learning rate')
    parser.add_argument('--nu_lr', type=float, default=1e-4, help='nu learning rate')
    parser.add_argument('--lamb_init', type=float, default=5, help='lamb initial value')
    parser.add_argument('--nu_init', type=float, default=0, help='nu initial value')
    parser.add_argument('--number_rs', type=int, default=12, help='learning rate')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='epsilon')
    parser.add_argument('--iterations', type=int, default=1000, help='number of lbfgs steps')
    parser.add_argument('--noise', type=float, default=0.0, help='noise level')
    parser.add_argument('--random_or_grid', type=str, default='random', help='random or grid sampling')
    parser.add_argument('--log_file', type=str, default="allen_cahn", help='log file name')
    # Others
    args = parser.parse_args()
    print(args)
    key = set_seed(args.seed)
    model = InversePDESolver(args, key)
    model.train()