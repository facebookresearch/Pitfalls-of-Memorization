# The Pitfalls of Memorization: When Memorization Hurts Generalization

![License](https://img.shields.io/badge/license-CC--BY--NC-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)

This repository contains the code associated with the paper: \
**[The Pitfalls of Memorization: When Memorization Hurts Generalization](https://arxiv.org/abs/2412.07684)**  
**Authors:** Reza Bayat*, Mohammad Pezeshki*, Elvis Dohmatob, David Lopez-Paz, Pascal Vincent

We explore the interplay between memorization and generalization in neural networks. Includes Memorization-Aware Training (MAT), a novel framework to mitigate the adverse effects of memorization and spurious correlations, alongside theoretical insights, algorithms, and experiments that deepen our understanding of how memorization impacts generalization under distribution shifts.

## The Interpretable Experiment (Figure 1)

```bash
python interpretable_experiment.py
```
![](https://github.com/facebookresearch/Pitfalls-of-Memorization/blob/main/assets/interpretable_experiment.gif?raw=true)

## Memorization: The Good, the Bad, and the Ugly (Figure 3)

```bash
python good_bad_ugly_memorization.py
```
![](https://github.com/facebookresearch/Pitfalls-of-Memorization/blob/main/assets/good_bad_ugly_memorization.gif?raw=true)

## Subpopulation Shift Experiments (Table 1)

Install the required packages and download the datasets:

```bash
pip install -r requirements.txt
python download.py --download --data_path ./data waterbirds celeba civilcomments multinli
export PYTHONPATH=$PYTHONPATH:./XRM
```

We first run XRM and store the held-out predictions for the training set as well as the inferred group labels for the validation set. For more details, checkout the instructions in the [XRM repo](https://github.com/facebookresearch/XRM). As an example, this is how it can be done for the Waterbirds dataset:

```bash
python main.py --phase 1 --datasets Waterbirds --group_labels no --algorithm XRM --out_dir ./phase_1_results --num_hparams_combs 10 --num_seeds 1 --slurm_partition <your_slurm_partition>
```

To run the MAT algorithm:

```bash
python main.py --phase 2 --datasets Waterbirds --group_labels yes --algorithm MAT --out_dir ./phase_2_results --phase_1_dir ./phase_1_results --num_hparams_combs 10 --num_seeds 1 --slurm_partition <your_slurm_partition>
```

To read the results:
- Model selection using the best 'va_wga', i.e., validation worst group accuracy (ground-truth annotations)
```bash
python XRM/read_results.py --dir phase_2_results --datasets Waterbirds --algorithms MAT --group_labels yes --selection_criterion va_wga
```
- Model selection using the best 'va_gi_wga', i.e., validation worst group accuracy (XRM-inferred annotations)
```bash
python XRM/read_results.py --dir phase_2_results --datasets Waterbirds --algorithms MAT --group_labels yes --selection_criterion va_gi_wga
```

## License

This source code is released under the CC-BY-NC license, included [here](LICENSE).

## Citation

If you make use of our work or code, please cite this work :)
```
@article{bayat2024pitfalls,
  title={The Pitfalls of Memorization: When Memorization Hurts Generalization},
  author={Bayat, Reza and Pezeshki, Mohammad and Dohmatob, Elvis and Lopez-Paz, David and Vincent, Pascal},
  journal={arXiv preprint arXiv:2412.07684},
  year={2024}
}
```
