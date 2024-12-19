# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import SGD
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from sklearn.datasets import make_moons
from torch.nn.functional import cross_entropy, softmax
from matplotlib.colors import LinearSegmentedColormap


def plot(results, net, path=None, title=''):

    print('plotting ...')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'

    cmap_background = LinearSegmentedColormap.from_list(
        'smooth_transition', ['#5CA0D4', 'white', '#F6B66A'], N=100)

    _, (ax_1, ax_2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 3]}, figsize=(6 * 0.8, 10 * 0.8))

    # Color schemes for the accuracy plot
    ax_1.plot(results['epochs'], results['maj_tr'], label='Majority Train', c='k', lw=2.5, alpha=0.75)
    ax_1.plot(results['epochs'], results['min_tr'], label='Minority Train', c='k', ls='--', lw=2.5, alpha=0.75)
    ax_1.plot(results['epochs'], results['maj_va'], label='Majority Test', c='#DF3000', lw=2, alpha=0.75)
    ax_1.plot(results['epochs'], results['min_va'], label='Minority Test', c='#DF3000', ls='--', lw=2, alpha=0.75)

    ax_1.set_xlabel('Epochs', fontsize=13)
    ax_1.set_ylabel('Accuracy', fontsize=13)
    ax_1.set_title(title, fontsize=16)
    ax_1.legend(frameon=False, fontsize=12)
    ax_1.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.6)

    # Decision Boundary Plot
    x_tr, y_tr = results['x_tr'], results['y_tr']
    xlim = np.array([-2, 2])
    ylim = np.array([-10, 10])
    d1, d2 = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 100),
        np.linspace(ylim[0], ylim[1], 100))
    hp = np.stack((d1.flatten(), d2.flatten()), axis=1)
    hp = torch.cat((torch.FloatTensor(hp), torch.zeros(10000, x_tr.shape[1] - 2)), 1)
    out = net(hp)
    out = out[:, 1:] - out[:, 0:1]
    out = torch.sigmoid(10 * out.reshape(100, 100))
    out = out.data.numpy()
    hp_x = hp[:, 0].reshape(100, 100)
    hp_y = hp[:, 1].reshape(100, 100)

    ax_2.contourf(
        hp_x, hp_y, out, np.linspace(0, 1, 20), cmap=cmap_background,
        alpha=0.8, vmin=0, vmax=1, extend="both")
    ax_2.contour(hp_x, hp_y, out, [0.5], antialiased=True, linewidths=1.0, colors="white")

    ax_2.scatter(x_tr[:, 0], x_tr[:, 1], c='white', s=60, edgecolor='none', zorder=1)
    ax_2.scatter(x_tr[:, 0], x_tr[:, 1],
                 c=np.array(['#3A7BC2', '#EE944F'])[y_tr],
                 s=50, edgecolor='white', linewidth=0.5, zorder=2, alpha=0.8)

    ax_2.axhline(y=0, ls="--", lw=1, color="white", alpha=0.5)
    ax_2.axvline(x=0, ls="--", lw=1, color="white", alpha=0.5)
    ax_2.set_xlim(np.array([-2, 2]))
    ax_2.set_ylim(np.array([-10, 10]))
    ax_2.grid(color="white", linestyle="--", lw=0.5, alpha=0.5)
    ax_2.set_xlabel(r'$x_y$', fontsize=16, labelpad=0)
    ax_2.set_ylabel(r'$x_a$', fontsize=16, labelpad=-5, rotation=0)

    ax_1.set_xlim(np.array([-10, 1010]))
    ax_1.set_ylim(np.array([-0.01, 1.01]))

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=300)
    plt.close()


def problem_1(seed, split, noise_std=1.0):
    n_samples = 1000
    dim = 1800
    maj_perc = {'tr': 0.9, 'va': 0.5, 'te': 0.5}[split]

    seed_ = {'tr': 0, 'va': 1, 'te': 2}[split] + seed
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    y = torch.zeros(n_samples, 1).bernoulli_(0.5)
    x1 = (torch.randn(n_samples, 1) * 0.12 - 1) * (2 * y - 1)
    x2 = (torch.randn(n_samples, 1) * 0.12 - 1) * (2 * y - 1)
    x2[torch.randperm(len(x2))[:int((1 - maj_perc) * len(x2))]] *= -1
    noise = noise_std * torch.randn(n_samples, dim - 2)
    x = torch.cat((x1 * 1, x2 * 5, noise), -1)
    m = x[:, :2].prod(1).gt(0).long()
    return x, y.squeeze().long(), m


class Net(torch.nn.Module):
    def __init__(self, dims, act=nn.Identity):
        super().__init__()
        layers = []
        for d in range(len(dims) - 1):
            layers += [nn.Linear(dims[d], dims[d + 1], bias=False), act()]
        self.backbone = torch.nn.Sequential(*(layers[:-1]))

    def forward(self, x):
        return self.backbone(x)


def run_erm(noise_std, lr=1e-2):
    epochs = 1000
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    x_tr, y_tr, m_tr = problem_1(seed, 'tr', noise_std=noise_std)
    x_va, y_va, m_va = problem_1(seed, 'va', noise_std=noise_std)
    net = Net([x_tr.shape[1], 2])
    opt = SGD(net.parameters(), lr=lr, momentum=0.0)
    results = {'x_tr': x_tr, 'y_tr': y_tr,
               'epochs': [], 'loss': [],
               'maj_tr': [], 'min_tr': [],
               'maj_va': [], 'min_va': [],
               'tmp': []}

    for epoch in tqdm(range(epochs)):

        results['epochs'] += [epoch + 1]
        y_hat_va = net(x_va)
        results['maj_va'] += [y_hat_va.argmax(1).eq(y_va)[m_va.eq(1)].float().mean().item()]
        results['min_va'] += [y_hat_va.argmax(1).eq(y_va)[m_va.eq(0)].float().mean().item()]
        y_hat_tr = net(x_tr)
        results['maj_tr'] += [y_hat_tr.argmax(1).eq(y_tr)[m_tr.eq(1)].float().mean().item()]
        results['min_tr'] += [y_hat_tr.argmax(1).eq(y_tr)[m_tr.eq(0)].float().mean().item()]

        opt.zero_grad()
        loss = cross_entropy(net(x_tr), y_tr)
        loss.backward()
        results['loss'] += [loss.item()]
        opt.step()

    return results, net


def run_mat(noise_std, lr=1e-2, temp=1):
    epochs = 1000
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    x_tr, y_tr, m_tr = problem_1(seed, 'tr', noise_std=noise_std)
    x_va, y_va, m_va = problem_1(seed, 'va', noise_std=noise_std)
    net_a = Net([x_tr.shape[1], 2])
    net_b = Net([x_tr.shape[1], 2])
    net_c = Net([x_tr.shape[1], 2])
    opt_a = SGD(net_a.parameters(), lr=lr, momentum=0.0)
    opt_b = SGD(net_b.parameters(), lr=lr, momentum=0.0)
    opt_c = SGD(net_c.parameters(), lr=lr, momentum=0.0)
    inds_a = torch.randint(2, (len(x_tr), 1))
    results = {'x_tr': x_tr, 'y_tr': y_tr,
               'epochs': [], 'loss_hi': [], 'loss_c': [],
               'maj_tr': [], 'min_tr': [],
               'maj_va': [], 'min_va': [],
               'tmp': []}
    
    def get_shift(y, ho, temp):
        # 1st dim: x, 2nd dim: yho
        p_yho_given_x = (ho / temp).softmax(1)
        # 1st dim: yho, 2nd dim: y
        p_y_yho = torch.cat([
            p_yho_given_x[y.eq(y_i)].sum(0).unsqueeze(1) / len(y)
            for y_i in y.unique()], 1)
        p_y_given_yho = p_y_yho / p_y_yho.sum(1, keepdim=True)
        # calibrated p_yho_given_x
        return torch.log(torch.mm(p_yho_given_x, p_y_given_yho) + 1e-6).detach()

    y_dyn, nc = y_tr.clone(), len(y_tr.unique())
    for epoch in tqdm(range(epochs)):
        a = net_a(x_tr)
        b = net_b(x_tr)
        hi = a * inds_a + b * (1 - inds_a)
        ho = a * (1 - inds_a) + b * inds_a
        opt_a.zero_grad()
        opt_b.zero_grad()
        loss_hi = cross_entropy(hi, y_dyn)
        loss_hi.backward()
        opt_a.step()
        opt_b.step()
        p_ho, y_ho = ho.softmax(dim=1).detach().max(1)
        is_flip = torch.bernoulli((p_ho - 1 / nc) * nc / (nc - 1)).long()
        y_dyn = is_flip * y_ho + (1 - is_flip) * y_dyn

    shift = get_shift(y_tr, ho, temp=temp).detach()

    for epoch in tqdm(range(epochs)):

        results['epochs'] += [epoch + 1]
        y_hat_va = net_c(x_va)
        results['maj_va'] += [y_hat_va.argmax(1).eq(y_va)[m_va.eq(1)].float().mean().item()]
        results['min_va'] += [y_hat_va.argmax(1).eq(y_va)[m_va.eq(0)].float().mean().item()]
        y_hat_tr = net_c(x_tr)
        results['maj_tr'] += [y_hat_tr.argmax(1).eq(y_tr)[m_tr.eq(1)].float().mean().item()]
        results['min_tr'] += [y_hat_tr.argmax(1).eq(y_tr)[m_tr.eq(0)].float().mean().item()]

        c = net_c(x_tr)
        opt_c.zero_grad()
        loss_c = cross_entropy(c + shift, y_tr)
        loss_c.backward()
        results['loss_c'] += [loss_c.mean().item()]
        opt_c.step()

    return results, net_c


results, net = run_erm(0.0, lr=3e-2)
plot(results, net, 'ERM_0.png', title=r'ERM - $\sigma_{\epsilon}$ = 0 (no memorization)')

results, net = run_erm(0.5)
plot(results, net, f'ERM_1.png', title=r'ERM - $\sigma_{\epsilon}$ = 1')

results, net = run_mat(0.5, temp=0.1)
plot(results, net, f'MAT_1.png', title=r'MAT - $\sigma_{\epsilon}$ = 1')
