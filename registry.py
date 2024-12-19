# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch

def get_hparams(args):

    hparams = {
        'phase': args['phase'],
        'data_path': args['data_path'],
        'dataset_name': args['dataset'],
        'group_labels': args['group_labels'],
        'balanced_batch': False,
        'precompute_features': False,
        'quick_run': args['quick_run'],
        'adam': args['adam'],
        'algorithm_name': args['algorithm'],
        'hparams_comb': args['hparams_comb'],
        'seed': args['seed'],
        'resume': args['resume'],
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'out_dir': args['out_dir'],
        'phase_1_dir': args['phase_1_dir']}

    rs = np.random.RandomState(args['hparams_comb'])
    if "Color" in args['dataset']:
        hparams['net_type'] = 'mlp'
        hparams['lr'] = 10 ** rs.uniform(-5, -3)
        hparams['batch_size'] = int(2 ** rs.uniform(3, 6))
        hparams['weight_decay'] = 10 ** rs.uniform(-6, -3)
    elif args['dataset'] in ["MultiNLI", "CivilComments"]:
        hparams['net_type'] = 'bert'
        hparams['balanced_batch'] = args['dataset'] == "CivilComments"
        hparams['lr'] = 10 ** rs.uniform(-6, -4)
        hparams['batch_size'] = int(2 ** rs.uniform(4, 6))
        hparams['weight_decay'] = 10 ** rs.uniform(-6, -3)
        hparams['last_layer_dropout'] = rs.choice([0., 0.1, 0.5])
    else:
        hparams['net_type'] = 'resnet'
        hparams['precompute_features'] = hparams['phase'] == 1
        if args['dataset'] == "Waterbirds":
            hparams['precompute_features'] = True
        hparams['lr'] = 10 ** rs.uniform(-5, -3)
        hparams['batch_size'] = int(2 ** rs.uniform(5, 7))
        hparams['weight_decay'] = 10 ** rs.uniform(-6, -3)

    if args['algorithm'] == 'GroupDRO':
        hparams['eta'] = 10 ** rs.uniform(-3, -1)

    if args['algorithm'] == 'IRM':
        hparams['lambda'] = 10 ** rs.uniform(-1, 5)
        hparams['penalty_anneal_iters'] = int(10 ** rs.uniform(0, 4))

    hparams['num_step'], hparams['checkpoint_freq'] = {
        'Waterbirds': [1001, 50],
        'CelebA': [10001, 50],
        'MultiNLI': [60001, 250],
        'CivilComments': [60001, 250],
        'MetaShift': [5001, 50],
        'ImagenetBG': [10001, 100],
        'ColorMNIST': [1001, 50],
        'InverseColorMNIST': [1001, 50],
        'MColor': [1001, 50],
        'ColorMNIST_V3': [1001, 50],
        'MultiColorMNIST': [1001, 50]}[args['dataset']]

    hparams['num_m'] = 2
    hparams['num_y'] = {
        'Waterbirds': 2,
        'CelebA': 2,
        'MultiNLI': 3,
        'CivilComments': 2,
        'MetaShift': 2,
        'ImagenetBG': 9,
        'ColorMNIST': 2,
        'InverseColorMNIST': 2,
        'MColor': 2,
        'ColorMNIST_V3': 2,
        'MultiColorMNIST': 10}[args['dataset']]

    if args['algorithm'] == 'MAT':
        rs = np.random.RandomState(args['hparams_comb'])
        hparams['temp'] = rs.choice([0.001, 0.01, 0.1])


    # for debugging only
    if hparams['quick_run']:
        hparams['precompute_features'] = True
        hparams['resume'] = False
        hparams['num_step'] = 10000
        hparams['num_workers'] = 0
        hparams['lr'] = 0.001
        hparams['weight_decay'] = 0
        hparams['batch_size'] = 2000
        hparams['last_layer_dropout'] = 0.1
        hparams['eta'] = 0.04

    ext = f'hpcomb_{args["hparams_comb"]}_seed{args["seed"]}'
    hparams['ckpt_file'] = os.path.join(hparams['out_dir'], f'ckpt_{ext}.pt')
    hparams['results_file'] = os.path.join(hparams['out_dir'], f'results_{ext}.json')

    return hparams
