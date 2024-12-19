# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import numpy as np
import torch
from XRM.utils import process_json_files, sort_and_remove_empty
from XRM.algorithms import ERM, GroupDRO, IRM

def get_algorithm(hparams, net, optim):

    # 'RWG' and 'SUBG' are both ERM but they differ in how they balance batches
    if hparams['algorithm_name'] in ['ERM', 'RWG', 'SUBG']:
        return ERM(hparams, net, optim)
    elif hparams['algorithm_name'] == 'GroupDRO':
        return GroupDRO(hparams, net, optim)
    elif hparams['algorithm_name'] == 'IRM':
        return IRM(hparams, net, optim)
    elif hparams['algorithm_name'] == 'MAT':
        return MAT(hparams, net, optim)


def get_xrm_results_path(phase_1_dir, dataset_name, seed=0):
    xrm_dir = f'{phase_1_dir}/XRM/{dataset_name}/group_labels_no'
    pattern = os.path.join(xrm_dir, f'results_hpcomb_*_seed{seed}.json')
    print(pattern)
    json_files = glob.glob(pattern)
    assert len(json_files) > 0
    all_values = process_json_files(
        pattern, json_files, True, 'flip_rate')
    all_values = sort_and_remove_empty(all_values)

    criterion_values = all_values['flip_rate']
    best_hparams_comb = int(all_values['hp_comb'][
        np.argmax(criterion_values)])
    path = pattern.replace('*', str(best_hparams_comb))
    path = path.replace('.json', '.pt')
    path = path.replace('results_hpcomb', 'inferred_hpcomb')
    print(path)
    return path


class MAT(ERM):
    def __init__(self, hparams, net, optim):
        super(MAT, self).__init__(hparams, net, optim)
        pt = torch.load(get_xrm_results_path(hparams["phase_1_dir"],
                        hparams["dataset_name"]),
                        weights_only=False)
        self.va_m_hat = pt['va']
        self.xrm_ho = pt['tr_ho']

    def get_shift(self, y, ho):
        # 1st dim: x, 2nd dim: yho
        p_yho_given_x = (ho / self.hparams['temp']).softmax(1)
        # 1st dim: yho, 2nd dim: y
        p_y_yho = torch.cat([
            p_yho_given_x[y.eq(y_i)].sum(0).unsqueeze(1) / len(y)
            for y_i in y.unique()], 1)
        p_y_given_yho = p_y_yho / p_y_yho.sum(1, keepdim=True)
        # calibrated p_yho_given_x
        return torch.log(torch.mm(p_yho_given_x, p_y_given_yho) + 1e-6).detach()

    def update(self, batch):
        i, x, y, m = batch
        x = x.to(self.device)
        y = y.to(self.device)
        m = m.to(self.device)
        self.optim.zero_grad(set_to_none=True)
        shift = self.get_shift(y, self.xrm_ho[i])
        loss = self.get_loss(self.net(x) + shift, y, m)
        loss.backward()
        self.optim.step()
        if self.optim.lr_scheduler is not None:
            self.optim.lr_scheduler.step()
