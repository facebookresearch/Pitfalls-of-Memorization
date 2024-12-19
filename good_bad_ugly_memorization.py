# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from torch.nn.functional import cross_entropy, softmax


def get_data(seed, split, N=1000, n=20,
             x_noise_dim=30, x_noise_std=0.0, y_noise_std=0.0):
    np.random.seed(seed + {'tr': 0, 'va': 1, 'te': 2}[split])
    torch.manual_seed(seed + {'tr': 0, 'va': 1, 'te': 2}[split])

    x_sweep = torch.linspace(-1.5, 1.5, N).unsqueeze(1)
    y_sweep = torch.tanh(x_sweep) + 0.5 * torch.tanh(-5 * x_sweep)
    x_sweep = torch.cat([x_sweep, torch.zeros(N, x_noise_dim)], 1)

    i = np.linspace(0, N-1, n).astype('int')
    x_sweep[i, 1:] = x_noise_std * torch.randn(n, x_noise_dim)
    x = x_sweep[i]
    y = y_sweep[i] + y_noise_std * torch.randn(n, 1)

    return x, y, x_sweep, y_sweep


class Net(torch.nn.Module):
    def __init__(self, dims, act=nn.Identity):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [nn.Linear(dims[d], dims[d + 1]), act()]
        self.layers = torch.nn.Sequential(*(self.layers[:-1]))

    def forward(self, x):
        return self.layers(x)

names = [r'Good memorization - $\sigma_{\mathbf{\epsilon}}=10^{-4}$',
         r'Bad memorization - $\sigma_{\mathbf{\epsilon}}=10^{-3}$',
         r'Ugly memorization - $\sigma_{\mathbf{\epsilon}}=0.0$']

y_hat_sweep = {}
for (x_noise_std, name) in zip([0.0001, 0.001, 0.0], names):

    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    x_tr, y_tr, x_sweep, y_sweep = get_data(
        seed, 'tr', x_noise_dim=30,
        x_noise_std=x_noise_std, y_noise_std=0.08)

    net = Net([x_tr.shape[1], 300, 300, 1], nn.Tanh)
    opt = Adam(net.parameters(), lr=1e-3) 

    epoch = 0
    while True:
        y_hat = net(x_tr)
        opt.zero_grad()
        loss = (y_hat - y_tr).pow(2).mean()
        loss.backward()
        opt.step()
        epoch += 1
        if epoch % 100 == 0:
            print(epoch, loss.item())
        if loss.item() < 1e-6 or epoch > 1e5:
            break
    y_hat_sweep[name] = net(x_sweep)

print('plotting ...')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.figure(figsize=(12, 4))
for i, name in enumerate(names):
    ax = plt.subplot(1, 3, i + 1)
    ax.scatter(x_tr[:, 0], y_tr[:, 0], alpha=0.95,
               c='#4B79BD', edgecolor='k', linewidth=1, s=30, zorder=2,
               label='(Noisy) Training examples' if i == 0 else None)
    ax.plot(x_sweep[:, 0], y_sweep[:, 0].data.numpy(),
            c='k', ls='--', alpha=0.5, lw=2,
            label=r'True function $f(x^*)$' if i == 0 else None)
    ax.plot(x_sweep[:, 0], y_hat_sweep[name][:, 0].data.numpy(),
            c='#4B79BD', alpha=0.7, lw=2,
            label=r'Predicted function $g(x) = g([x^*, \mathbf{\epsilon}])$' if i == 0 else None)
    ax.set_title(name, fontsize=14)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-0.5, 0.5])
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.axvline(x=0, color="white", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=0, color="white", linestyle="--", linewidth=1, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlabel(r'$x_y$ (core feature)')    
    ax.set_ylabel(r'true and prediction functions', fontsize=10)
    if i == 0:
        ax.legend(frameon=False, fontsize=9.5, loc='upper left')

plt.tight_layout()
plt.savefig('good_bad_ugly.png', dpi=300)
