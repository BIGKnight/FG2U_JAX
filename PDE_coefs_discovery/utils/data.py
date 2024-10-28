import scipy.io
import numpy as np
import torch
import warnings
from copy import deepcopy


def load_data(path, keys=None):
    if keys is None:
        # default: 2D
        keys = ['x', 'tt', 'uu']
    data = scipy.io.loadmat(path)
    xi = [np.real(data[k]).reshape((-1, 1)) for k in keys[:-1]]
    raw = np.real(data[keys[-1]]).T
    u = np.real(data[keys[-1]]).T.reshape((-1, 1))
    x = np.concatenate([xx.reshape((-1, 1)) for xx in np.meshgrid(*xi)], axis=1)
    data = np.concatenate([x, u], axis=1)
    domain = [[np.min(x[:, i]), np.max(x[:, i])] for i in range(x.shape[-1])]
    return data, domain, raw


class DataGenerator(object):
    def __init__(self, pde, device):
        self.pde = pde
        self.device = device
        self.data = pde.data if hasattr(pde, 'data') else None
        self.domain = np.array(self.pde.domain)

    def border2idx(self, border, data=None):
        """
                        b[1][0]      b[1][1]
                ------------------------------------
                |           |           |           |
                |           |           |           |
        b[0][0] ------------------------------------
                |           |***********|           |
                |           |***********|           |
                |           |***********|           |
        b[0][1] ------------------------------------
                |           |           |           |
                |           |           |           |
                |           |           |           |
                ------------------------------------
        :param border: relative border
        :param data:
        :return:
        """
        if data is None:
            data = self.data
        assert data is not None
        borders = np.array(border)
        idx = np.ones(len(data)).astype(np.bool)
        for d in range(len(borders)):
            domain_low, domain_high = self.domain[d]
            border_low, border_high = borders[d]
            if border_high == border_low or border_high == 1:
                border_high += 1e-6
            low = (domain_high - domain_low) * border_low + domain_low
            high = (domain_high - domain_low) * border_high + domain_low
            cond = np.logical_and(data[:, d] < high, data[:, d] >= low)
            idx = np.logical_and(idx, cond)
        return idx

    def slice(self, border, data=None):
        if data is None:
            data = self.data
        assert data is not None
        return data[self.border2idx(border)]

    def gen_data_i(
            self,
            n,
            sigma=0.,
    ):
        if self.pde.analytical_initial_term:
            assert self.pde.input_dim > 1
            assert n < np.inf
            x_i = np.random.random((n, self.pde.input_dim - 1))
            x_i = np.concatenate((x_i, (np.zeros(n)[:, None])), axis=1)
            x_i = self.to_domain(x_i)
            x_i = torch.from_numpy(x_i)
            u_i = self.pde.initial_term(x_i)
        else:
            assert self.data is not None
            border = np.array([[0., 1.] for _ in range(len(self.domain))])
            border[-1][1] = 0.
            data_i = self.slice(border)
            if n <= len(data_i):
                idx = np.random.choice(np.arange(len(data_i)), n, replace=False)
                x_i, u_i = data_i[idx, :-1], data_i[idx, -1:]
            else:
                warnings.warn('data_i: Only {} data points available, while {} are required.'.format(len(data_i), n))
                x_i, u_i = data_i[:, :-1], data_i[:, -1:]
            x_i, u_i = torch.from_numpy(x_i), torch.from_numpy(u_i)

        noise = torch.randn(u_i.shape) * sigma
        u_i.data = u_i.data + noise

        return x_i.double().to(self.device), u_i.double().to(self.device)

    def gen_data_b(
            self,
            n,
            sigma=0.,
            with_init=True,
    ):
        k = len(self.domain)
        if with_init:
            k = k - 1
        if self.pde.analytical_boundary_term:
            assert n < np.inf
            x_b = np.random.random((n, len(self.domain)))
            for i in range(len(x_b)):
                x_b[i][np.random.randint(k)] = 0. if np.random.randint(2) else 1.
            x_b = self.to_domain(x_b)
            x_b = torch.from_numpy(x_b)
            u_b = self.pde.boundary_term(x_b)
        else:
            assert self.data is not None
            idx = np.zeros(len(self.data)).astype(np.bool)
            for i in range(k):
                border = np.array([[0., 1.] for _ in range(len(self.domain))])
                border[i][1] = 0.
                idx = np.logical_or(idx, self.border2idx(border))
                border = np.array([[0., 1.] for _ in range(len(self.domain))])
                border[i][0] = 1.
                idx = np.logical_or(idx, self.border2idx(border))
            data_b = self.data[idx]

            if n <= len(data_b):
                idx = np.random.choice(np.arange(len(data_b)), n, replace=False)
                x_b, u_b = data_b[idx, :-1], data_b[idx, -1:]
            else:
                warnings.warn('data_b: Only {} data points available, while {} are required.'.format(len(data_b), n))
                x_b, u_b = data_b[:, :-1], data_b[:, -1:]
            x_b, u_b = torch.from_numpy(x_b), torch.from_numpy(u_b)

        noise = torch.randn(u_b.shape) * sigma
        u_b.data = u_b.data + noise

        return x_b.double().to(self.device), u_b.double().to(self.device)

    def gen_data_b_sym(
            self,
            n,
    ):
        # TODO: Update
        t = np.random.random(n)
        x_1 = np.concatenate([np.zeros_like(t)[:, None], t[:, None]], 1)
        x_2 = np.concatenate([np.ones_like(t)[:, None], t[:, None]], 1)
        x_1 = torch.from_numpy(self.to_domain(x_1))
        x_2 = torch.from_numpy(self.to_domain(x_2))

        return x_1.double().to(self.device), x_2.double().to(self.device)

    def gen_data_u(
            self,
            n,
            sigma=0.,
    ):
        if self.pde.analytical_solution:
            assert n < np.inf
            x = np.random.random((n, len(self.domain)))
            x = self.to_domain(x)
            x = torch.from_numpy(x)
            u = self.pde.u(x)
        else:
            assert self.data is not None
            if n <= len(self.data):
                idx = np.random.choice(np.arange(len(self.data)), n, replace=False)
                x, u = self.data[idx, :-1], self.data[idx, -1:]
            else:
                warnings.warn(
                    'data_u: Only {} data points available, while {} are required.'.format(len(self.data), n))
                x, u = self.data[:, :-1], self.data[:, -1:]
            x, u = torch.from_numpy(x), torch.from_numpy(u)

        noise = torch.randn(u.shape) * sigma
        u.data = u.data + noise

        return x.double().to(self.device), u.double().to(self.device)

    def gen_data_f(
            self,
            n,
            sigma=0.,
    ):
        x_f = np.random.random((n, len(self.domain)))
        x_f = self.to_domain(x_f)
        x_f = torch.from_numpy(x_f)
        f = self.pde.force_term(x_f)
        noise = torch.randn(f.shape) * sigma
        f.data = f.data + noise

        return x_f.double().to(self.device), f.double().to(self.device)

    def to_domain(self, x):
        l, u = np.array(self.domain)[:, 0], np.array(self.domain)[:, 1]
        return x * (u - l) + l

    def gen_pinn_data(self, n_dict, sigma_dict):
        data = {}
        if n_dict.get('i', 0) > 0:
            x_i, u_i = self.gen_data_i(n=n_dict['i'], sigma=sigma_dict.get('i', 0.))
            data['i'] = [x_i, u_i]
        if n_dict.get('b', 0) > 0:
            x_b, u_b = self.gen_data_b(n=n_dict['b'], sigma=sigma_dict.get('b', 0.), with_init=(n_dict.get('i', 0) > 0))
            data['b'] = [x_b, u_b]
        if n_dict.get('u', 0) > 0:
            x, u = self.gen_data_u(n=n_dict['u'], sigma=sigma_dict.get('u', 0.))
            data['u'] = [x, u]
        if n_dict.get('f', 0) > 0:
            x_f, f = self.gen_data_f(n=n_dict['f'], sigma=sigma_dict.get('f', 0.))
            data['f'] = [x_f, f]
        if n_dict.get('b_sym', 0) > 0:
            x_1, x_2 = self.gen_data_b_sym(n=n_dict['b_sym'])
            data['b_sym'] = x_1, x_2
        x_t, u_t = self.gen_data_u(n=n_dict['t'], sigma=0.)
        data_t = [x_t, u_t]

        return data, data_t
    def gen_kdv_data(self, n_dict, sigma_dict):
        data = {}
        if n_dict.get('i', 0) > 0:
            x_i, u_i = self.gen_data_i(n=n_dict['i'], sigma=sigma_dict.get('i', 0.))
            data['i'] = [x_i, u_i]
        if n_dict.get('b', 0) > 0:
            x_b, u_b = self.gen_data_b(n=n_dict['b'], sigma=sigma_dict.get('b', 0.), with_init=(n_dict.get('i', 0) > 0))
            data['b'] = [x_b, u_b]
        if n_dict.get('u', 0) > 0:
            orig_data = scipy.io.loadmat(self.pde.path)
            N0 = int(n_dict['u']/2)
            N1 = n_dict['u'] - N0
            idx_t0 = int(0.2 * orig_data['uu'].shape[1])
            idx_t1 = int(0.8 * orig_data['uu'].shape[1])
            idx_x = np.random.choice(orig_data['uu'].shape[0], N0, replace=False)
            x0 = orig_data['x'][:,idx_x].reshape((N0,1))
            t0 = np.ones((N0,1),'float64') * orig_data['tt'][:,idx_t0]
            u0 = orig_data['uu'][idx_x,idx_t0].reshape((N0,1))
            X0 = np.concatenate((x0,t0),axis=1)
            idx_x = np.random.choice(orig_data['uu'].shape[0], N1, replace=False)
            x1 = orig_data['x'][:,idx_x].reshape((N1,1))
            t1 = np.ones((N1,1),'float64') * orig_data['tt'][:,idx_t1]
            u1 = orig_data['uu'][idx_x,idx_t1].reshape((N1,1))
            X1 = np.concatenate((x1,t1),axis=1)
            X = np.concatenate((X0,X1),axis=0)
            u = np.concatenate((u0,u1),axis=0)
            X, u = torch.from_numpy(X), torch.from_numpy(u)
            X = X.double().to(self.device)
            u = u.double().to(self.device)
            data['u'] = [X, u]
        if n_dict.get('f', 0) > 0:
            x_f, f = self.gen_data_f(n=n_dict['f'], sigma=sigma_dict.get('f', 0.))
            data['f'] = [x_f, f]
        if n_dict.get('b_sym', 0) > 0:
            x_1, x_2 = self.gen_data_b_sym(n=n_dict['b_sym'])
            data['b_sym'] = x_1, x_2
        return data

    def gen_trans_mtx(self, n_grid_x, n_grid_y):
        l, u = np.array(self.domain)[:, 0], np.array(self.domain)[:, 1]
        interval = (u - l) / [n_grid_x, n_grid_y]
        trans_mtx = []
        for i in range(n_grid_x):
            for j in range(n_grid_y):
                anchor = ([i, j] * interval) + l
                trans_mtx.append(np.array([[interval[0], 0, 0], [0, interval[1], 0], [anchor[0], anchor[1], 1]]))
        return np.array(trans_mtx)


class XPINNDataGenerator(object):
    def __init__(self, pde, borders, device):
        self.pde = pde
        self.device = device
        self.data_generator = DataGenerator(pde, device)
        self.data = pde.data if hasattr(pde, 'data') else None
        assert self.data is not None
        self.domain = np.array(self.pde.domain)
        self.borders = borders
        self.partition_shape = [len(i) - 1 for i in self.borders]
        self.n_subdomains = np.prod(self.partition_shape)

        self.subdomain_data_generators = {}
        for n in range(self.n_subdomains):
            loc = self.n2loc(n)
            subpde = deepcopy(self.pde)
            subpde.data = self.data_generator.slice(
                border=[[self.borders[i][j], self.borders[i][j + 1]]
                        for i, j in enumerate(loc)])
            subpde.domain = [[np.min(subpde.data[:, i]), np.max(subpde.data[:, i])] for i in
                             range(subpde.data.shape[-1] - 1)]
            self.subdomain_data_generators[str(n)] = DataGenerator(subpde, device)

    def generate_interface_data(self, id1, id2, n, sigma=0.):
        id1, id2 = min(id1, id2), max(id1, id2)
        loc1, loc2 = self.n2loc(id1), self.n2loc(id2)
        if np.sum((loc1 - loc2) ** 2) != 1:
            return None, None
        d = np.argmax((loc1 - loc2) ** 2)
        assert loc1[d] == loc2[d] - 1
        v = self.borders[d][loc2[d]]
        x = np.random.random((n, len(self.domain)))
        x[:, d] = v
        x = self.data_generator.to_domain(x)
        x = torch.from_numpy(x)
        f = self.pde.force_term(x)
        noise = torch.randn(f.shape) * sigma
        f.data = f.data + noise

        return x.double().to(self.device), f.double().to(self.device)

    def filter(self, subdomain_id, data):
        x, y = data
        data_np = np.concatenate([x.detach().cpu().numpy(), y.detach().cpu().numpy()], 1)
        loc = self.n2loc(subdomain_id)
        idx_np = self.data_generator.border2idx(
            border=[[self.borders[i][j], self.borders[i][j + 1]] for i, j in enumerate(loc)], data=data_np)
        idx = torch.from_numpy(idx_np).to(self.device)
        return x[idx], y[idx]

    def n2loc(self, n):
        loc = []
        for i in range(1, len(self.partition_shape) + 1):
            k = self.partition_shape[-i]
            loc = [n % k] + loc
            n = n // k
        return np.array(loc)

    def loc2n(self, loc):
        k = 1
        n = 0
        for i in range(1, len(self.partition_shape) + 1):
            n += k * loc[-i]
            k *= self.partition_shape[-i]
        return n

    def gen_xpinn_data(self, n_dict, sigma_dict):
        pinn_data, pinn_data_t = self.data_generator.gen_pinn_data(n_dict, sigma_dict)
        xpinn_data = {'if': {}}
        xpinn_data_t = {}
        for i in range(self.n_subdomains):
            xpinn_data[str(i)] = {}
            for k in ['i', 'b', 'u', 'f']:
                if pinn_data.get(k) is not None:
                    x, y = self.filter(subdomain_id=i, data=pinn_data[k])
                    if len(x) > 0:
                        xpinn_data[str(i)][k] = [x, y]
            for j in range(i+1, self.n_subdomains):
                x, f = self.generate_interface_data(id1=i, id2=j, n=n_dict['if'], sigma=sigma_dict.get('if', 0.))
                if x is not None:
                    xpinn_data['if']['{}-{}'.format(i, j)] = [x, f]
            x, u = self.subdomain_data_generators[str(i)].gen_data_u(n=np.inf, sigma=0.)
            xpinn_data_t[str(i)] = [x, u]
        return xpinn_data, xpinn_data_t


if __name__ == '__main__':
    from pde import ConvectionDiffusion
    from matplotlib import pyplot as plt

    pde = ConvectionDiffusion(path='../Data/convection_diffusion.mat')
    data_generator = DataGenerator(pde=pde, device='cpu')
    data, data_t = data_generator.gen_pinn_data(
        n_dict={'i': np.inf, 'b': 0, 'f': 1000, 'u': 0, 'b_sym': 100}, sigma_dict={})
    for k in data.keys():
        x, y = data[k]
        print(len(x))
        plt.plot(x[:, 1], x[:, 0], 'o', markersize=3, label=k, clip_on=False)
        if k == 'b_sym':
            plt.plot(y[:, 1], y[:, 0], 'o', markersize=3, label=k, clip_on=False)
    plt.legend()
    plt.show()

    # from matplotlib import pyplot as plt
    # from pde import Burgers

    # pde = Burgers(path='../Data/burgers.mat')
    # xpinn_data_generator = XPINNDataGenerator(pde, borders=[[0., 0.4, 1.0], [0., 1.0]], device='cpu')
    # plt.imshow(pde.u.T, interpolation='nearest', cmap='seismic',
    #            extent=list(pde.domain[1]) + list(pde.domain[0]), origin='lower', aspect='auto')
    # data, data_t = xpinn_data_generator.gen_xpinn_data(n_dict={'i': 100, 'b': 100, 'f': 1000, 'u': np.inf, 'if': 100}, sigma_dict={})
    # # data, data_t = xpinn_data_generator.data_generator.gen_pinn_data(n_dict={'i': 100, 'b': 100, 'f': 1000, 'u': 0, 'if': 100}, sigma_dict={})
    # for k in data_t.keys():
    #     for kk in data[k].keys():
    #         if kk == 'u':
    #             x = data[k][kk][0]
    #             print(len(x))
    #             plt.plot(x[:, 1], x[:, 0], 'o', markersize=3, label=k + '-' + kk, clip_on=False)
    # plt.legend()
    # plt.show()
