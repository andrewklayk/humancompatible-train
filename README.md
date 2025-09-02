# humancompatible-train: a package for constrained machine learning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Setup](https://github.com/humancompatible/train/actions/workflows/setup.yml/badge.svg)](https://github.com/humancompatible/train/actions/workflows/setup.yml)

The toolkit implements algorithms for constrained training of neural networks based on PyTorch, and inspired by PyTorch's API.
<!-- , as well as a tool to compare stochastic-constrained stochastic optimization algorithms on a _fair learning_ task in the `experiments` folder. -->

## Table of Contents
1. [Basic installation instructions](#basic-installation-instructions)
2. [Using the toolkit](#using-the-toolkit)
3. [Extending the toolkit](#extending-the-toolkit) 
4. [Reproducing the Benchmark](#reproducing-the-benchmark)
5. [License and terms of use](#license-and-terms-of-use)
6. [References](#references)

humancompatible-train is still under active development! If you find bugs or have feature
requests, please file a
[Github issue](https://github.com/humancompatible/train/issues). 

## Installation

Use

```
pip install humancompatible-train
```

The only dependencies of this package are `numpy` and `torch`.

## Using the toolkit

The toolkit implements algorithms for constrained training of neural networks based on PyTorch.

The algorithms follow the `dual_step()` - `step()` framework: taking inspiration from PyTorch, the `double_step` does updates related to the dual parameters and prepares for the primal update (by, e.g., saving constraint gradients), and `step()` updates the primal parameters.

In general, your code using `humancompatible-train` would look something like this:

```
for inputs, labels in dataloader:
  # inference
  outputs = model(inputs)
  # calculate constraints and grads
  for constraint in constraints:
      c_eval = constraint(outputs, labels)
      c_eval.backwards(retain_grad=True)
      # depending on optimizer, update dual parameters / save constraint gradient / both
      optimizer.dual_step(c_eval)
      optimizer.zero_grad()
  # calculate objective
  loss = criterion(outputs,labels)
  loss.backwards()
  optimizer.step()
  optimizer.zero_grad()
```

Our idea is to
1. Deviate minimally from the usual PyTorch workflow
2. Make different stochastic-constrained stochastic optimization algorithms nearly interchangable in the code.

### Code examples

You are invited to check out our new API presented in notebooks in the `examples` folder.

*The legacy API used for the benchmark is presented in `examples/_old_/algorithm_demo.ipynb` and `examples/_old_/constraint_demo.ipynb`.*

## Extending the toolkit

### Adding new code

**To add a new algorithm**, you can subclass the PyTorch ```Optimizer``` class and proceed following the API guideline presented above.

## Reproducing the Benchmark

The code used in [our benchmark paper](https://arxiv.org/abs/2507.04033) is not migrated to the new API yet (WIP).

### Basic installation instructions
The code requires Python version ```3.11```.

1. Create a virtual environment

**bash** (Linux)
```
python3.11 -m venv fairbenchenv
source fairbenchenv/bin/activate
```
**cmd** (Windows)
```
python -m venv fairbenchenv
fairbenchenv\Scripts\activate.bat
```
2. Install from source.
```
git clone https://github.com/humancompatible/train.git
cd train
pip install -r requirements.txt
pip install .
```

If you wish to edit the code of the algorithms, install as an editable package:
```
pip install -e .
```

__Warning__: it is recommended to use Stochastic Ghost with the mkl-accelerated version of the scipy package with Stochastic Ghost; to install it, run

```pip install --force-reinstall -i https://software.repos.intel.com/python/pypi scipy```

after installing requirements.txt; otherwise, the algorithm will run slower. However, this is not supported on MacOS and may fail on some Windows devices.

<!-- Install via pip -->
<!-- ``` -->
<!-- pip install folktables -->
<!-- ``` -->

### Running the algorithms

The benchmark comprises the following algorithms:
- Stochastic Ghost [[2]](#2),
- SSL-ALM [[3]](#3),
- Stochastic Switching Subgradient [[4]](#4).

To reproduce the experiments of the paper, run the following:
``` python
cd experiments
python run_folktables.py data=folktables alg=sslalm
python run_folktables.py data=folktables alg=alm
python run_folktables.py data=folktables alg=ghost
python run_folktables.py data=folktables alg=ssg
python run_folktables.py data=folktables alg=sgd     # baseline, no fairness
python run_folktables.py data=folktables alg=fairret # baseline, fairness with regularizer
```
Each command will start 10 runs of the `alg`, 30 seconds each.
The results will be saved to `experiments/utils/saved_models` and `experiments/utils/exp_results`.
<!-- In the repository, we include the configuration needed to reproduce the experiments in the paper. To do so, go to `experiments` and run `python run_folktables.py data=folktables alg=sslalm`. -->
<!-- Repeat for the other algorithms by changing the `alg` parameter. -->

This repository uses [Hydra](https://hydra.cc/) to manage parameters; see `experiments/conf` for configuration files. 
* To change the parameters of the experiment, such as the number of runs for each algorithm, run time, the dataset used (*note: for now supports only Folktables*) - use `experiment.yaml`. 
* To change the dataset settings - such as file location - or do dataset-specific adjustments - such as the configuration of the protected attributes - use `data/{dataset_name}.yaml`
* To change algorithm hyperparameters, use `alg/{algorithm_name}.yaml`.
* To change constraint hyperparameters, use `constraint/{constraint_name}.yaml`

<!-- ; it is installed as one of the dependencies. -->
<!-- To learn more about using Hydra, please check out the [official tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app). -->

### Producing plots
The plots and tables like the ones in the paper can be produced using the two notebooks. `experiments/algo_plots.ipynb` houses the convergence plots, and `experiments/model_plots.ipynb` - all the others.


## License and terms of use

humancompatible-train is provided under the Apache 2.0 Licence.

The benchmark part of the package relies on the Folktables package, provided under MIT Licence.
It provides code to download data from the American Community Survey
(ACS) Public Use Microdata Sample (PUMS) files managed by the US Census Bureau.
The data itself is governed by the terms of use provided by the Census Bureau.
For more information, see https://www.census.gov/data/developers/about/terms-of-service.html

<!-- ## Cite this work -->

<!-- If you use this work, we encourage you to cite our paper, and the folktables dataset [[1]](#1). -->

<!-- ``` -->
<!-- @article{ding2021retiring, -->
<!--   title={Retiring Adult: New Datasets for Fair Machine Learning}, -->
<!--   author={Ding, Frances and Hardt, Moritz and Miller, John and Schmidt, Ludwig}, -->
<!--   journal={Advances in Neural Information Processing Systems}, -->
<!--   volume={34}, -->
<!--   year={2021} -->
<!-- } -->
<!-- ``` -->

## Future work

- Add more algorithms
- Add more examples from different fields where constrained training of DNNs is employed
- Migrate the benchmark to the new API

## References

If you use this work, we encourage you to cite [our paper](https://arxiv.org/abs/2507.04033),

```
@misc{kliachkin2025benchmarkingstochasticapproximationalgorithms,
      title={Benchmarking Stochastic Approximation Algorithms for Fairness-Constrained Training of Deep Neural Networks}, 
      author={Andrii Kliachkin and Jana Lepšová and Gilles Bareilles and Jakub Mareček},
      year={2025},
      eprint={2507.04033},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.04033}, 
}
```

<a id="1">[1]</a> 
Ding, Hardt & Miller et al. (2021) Retiring Adult: New Datasets for Fair Machine Learning, Curran Associates, Inc..

<a id="2">[2]</a> 
Facchinei & Kungurtsev (2023) Stochastic Approximation for Expectation Objective and Expectation Inequality-Constrained Nonconvex Optimization, arXiv.

<a id="3">[3]</a> 
Huang, Zhang & Alacaoglu (2025) Stochastic Smoothed Primal-Dual Algorithms for Nonconvex Optimization with Linear Inequality Constraints, arXiv.

<a id="4">[4]</a> 
Huang & Lin (2023) Oracle Complexity of Single-Loop Switching Subgradient Methods for Non-Smooth Weakly Convex Functional Constrained Optimization, Curran Associates Inc..