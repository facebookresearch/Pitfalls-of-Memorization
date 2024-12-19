# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import torch
from XRM.utils import load_checkpoint, save_checkpoint, set_all_seeds, is_degenerate
from XRM.utils import report_stats, Iter, prepare_out_dir, find_best_hparams_comb
from XRM.datasets import get_loaders
from XRM.algorithms import XRM as XRM_algo
from XRM.networks import get_network, get_optim, TwinNets
from registry import get_hparams
from algos import get_algorithm


def phase_1(args):

    assert args['group_labels'] == 'no'  # making sure not to use group info
    set_all_seeds(args['seed'])
    hparams = get_hparams(args)
    prepare_out_dir(hparams)

    loaders = get_loaders(hparams)
    net_a = get_network(hparams)
    net_b = get_network(hparams)
    net = TwinNets(hparams, net_a, net_b)
    optim = get_optim(hparams, net)
    # randomly assings each training example to one of the twin nets
    assigns = torch.zeros(loaders['tr'].n_examples, 1).bernoulli_(0.5)
    algorithm = XRM_algo(hparams, net, optim, loaders['tr'], assigns)
    step_st = load_checkpoint(hparams, algorithm) if args['resume'] else 0

    print('Training starts..')
    tr_loader_iterator = Iter(loaders['tr'])
    for step in range(step_st, hparams['num_step']):
        batch = tr_loader_iterator.next()
        algorithm.update(batch)
        if step % hparams['checkpoint_freq'] == 0:
            report_stats(hparams, algorithm, loaders, step)
            save_checkpoint(hparams, algorithm, step)

    # find cross-mistakes at the end of training
    path = os.path.join(
        hparams['out_dir'], 
        f'inferred_hpcomb_{hparams["hparams_comb"]}'
        f'_seed{hparams["seed"]}.pt')
    inferred_groups, ys = {}, {}
    for split in ['tr', 'va']:
        ig_, ho_, ys_ = algorithm.cross_mistakes(loaders[split], split == 'tr')
        inferred_groups[split] = ig_
        inferred_groups[split + '_ho'] = ho_  # soft held-out preds
        ys[split] = ys_
    if is_degenerate(algorithm, inferred_groups, ys):
        report_stats(hparams, algorithm, loaders, hparams['num_step'])
    else:
        torch.save(inferred_groups, path)
        print('Inferred group labels stored at:', path)


def phase_2(args):
    
    set_all_seeds(args['seed'])
    hparams = get_hparams(args)
    prepare_out_dir(hparams)

    loaders = get_loaders(hparams)
    net = get_network(hparams)
    optim = get_optim(hparams, net)
    algorithm = get_algorithm(hparams, net, optim)
    step_st = load_checkpoint(hparams, algorithm) if args['resume'] else 0

    print('Training starts..')
    tr_loader_iterator = Iter(loaders['tr'])
    for step in range(step_st, hparams['num_step']):
        batch = tr_loader_iterator.next()
        algorithm.update(batch)
        if step % hparams['checkpoint_freq'] == 0:
            report_stats(hparams, algorithm, loaders, step)
            save_checkpoint(hparams, algorithm, step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Phase 1 or 2')
    parser.add_argument('--slurm_partition', type=str, default=None)
    parser.add_argument('--slurm_dir', type=str, default="./submitit")
    parser.add_argument('--constraint', type=str, default='volta32gb')
    parser.add_argument('--phase', type=int, choices=[1, 2], required=True, 
                        help='1: group inference, 2: invariant learning')
    parser.add_argument('--data_path',
                        default='./data')
    parser.add_argument('--datasets', nargs='+', default=['Waterbirds'])
    parser.add_argument('--group_labels', default='yes',
                        help='Choose "yes" for ground-truth'
                             'Choose "no" for none'
                             'or input a .pt file containing group labels'
                             'or input a dir path containing .pt files (best is selected)')
    parser.add_argument('--algorithms', nargs='+', default=['GroupDRO'])
    parser.add_argument('--out_dir', default='./out')
    parser.add_argument('--phase_1_dir', default='./out')
    parser.add_argument('--resume', action='store_true')
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--hparams_comb', type=int)
    group1.add_argument('--num_hparams_combs', type=int)
    group1.add_argument('--best_hparams_comb_selection_metric', type=str,
                        help='Specify metric for determining best hp_comb.',
                        choices=['va_wga', 'va_avg_acc', 'flip_rate'])
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--seed', type=int)
    group2.add_argument('--num_seeds', type=int)
    parser.add_argument('--quick_run', action='store_true',
                        help='Run with a single hparam combination to debug')
    parser.add_argument('--adam', action='store_true')

    args = parser.parse_args()

    def get_hparams_combs(args, out_dir):
        # Determine which hparams_combs to run:
        # the best one wrt a metric, a range, or a specified one.
        if args.best_hparams_comb_selection_metric is not None:
            # HP selection is done at seed=0.
            # the best HP is then ran for multiple seeds
            best_hp = find_best_hparams_comb(
                out_dir, args.best_hparams_comb_selection_metric, seed=0)
            print(f'best_hp for {out_dir} is {best_hp}.')
            return [best_hp]
        elif args.num_hparams_combs is not None:
            return range(args.num_hparams_combs)
        else:
            return [args.hparams_comb]

    def get_seeds(args):
        # Determine which seeds to run:
        # a range or a specified one
        if args.num_seeds is not None:
            return range(args.num_seeds)
        else:
            return [args.seed]

    def get_full_out_dir(args, algorithm, dataset, seed):
        if args.group_labels in ['yes', 'no']:
            folder_name = 'group_labels_' + args.group_labels
            gl = args.group_labels
        elif '.pt' in args.group_labels:
            folder_name = 'group_labels_inferred'
            gl = args.group_labels
        else:
            folder_name = 'group_labels_inferred'
            best_phase_1_hp = find_best_hparams_comb(
                os.path.join(args.group_labels, dataset,
                             'group_labels_no'),
                'flip_rate', seed=0)
            gl = os.path.join(
                args.group_labels, dataset,
                'group_labels_no',
                f'inferred_hpcomb_{best_phase_1_hp}_seed{seed}.pt')
            print('selected .pt file:', gl)
        out_dir = os.path.join(args.out_dir, algorithm, dataset, folder_name)
        return out_dir, gl

    # creating a list of experiments
    args_list = []
    for dataset in args.datasets:
        for algorithm in args.algorithms:
            for seed in get_seeds(args):
                out_dir, gl = get_full_out_dir(args, algorithm, dataset, seed)
                for hparams_comb in get_hparams_combs(args, out_dir):

                    args_list += [{'phase': args.phase,
                                   'data_path': args.data_path,
                                   'dataset': dataset,
                                   'group_labels': gl,
                                   'algorithm': algorithm,
                                   'hparams_comb': hparams_comb,
                                   'seed': seed,
                                   'out_dir': out_dir,
                                   'phase_1_dir': args.phase_1_dir,
                                   'quick_run': args.quick_run,
                                   'adam': args.adam,
                                   'resume': args.resume}]

    run_phase = {1: phase_1, 2: phase_2}[args.phase]
    print(f'About to launch {len(args_list)} jobs..')
    # running locally
    if args.slurm_partition is None:
        for args_ in args_list:
            run_phase(args_)

    # running on cluster
    else:
        import submitit
        executor = submitit.SlurmExecutor(folder=args.slurm_dir)
        executor.update_parameters(
            time=60 * 60,
            gpus_per_node=1,
            array_parallelism=512,
            cpus_per_task=4,
            constraint=args.constraint,
            partition=args.slurm_partition)
        executor.map_array(run_phase, args_list)
