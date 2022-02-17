# Mixed neural likelihood estimation for models of decision-making

<img src="data/figure_1_mixed.png"
     alt="Figure 1"
     style="float: left; margin-right: 10px;" />

Mixed neural likelihood estimation (MNLE) enables Bayesian parameter inference for models of decision-making in which the likelihood function cannot be evaluated. In this scenario, inference is challenging because first, common approximate inference methods like MCMC or variational inference (VI) cannot be applied (no access to the likelihood), and second, models typically return mixed data types, e.g., categorical choices and continuous reaction times (Fig. 1 left; with the drift-diffusion model as an example). 

MNLE solves both challenges: It is based on the framework on neural likelihood estimation [[1](http://proceedings.mlr.press/v89/papamakarios19a.html)] which uses artificial neural networks to learn a synthetic likelihood using data simulated from the model, which in turn can be used to perform inference with MCMC or VI. In particular, it extends neural likelihood estimation to mixed data types (Fig. 1 middle), such that it can be applied to models of decision making. Once trained, MNLE can be used to obtain posterior samples via MCMC or VI (Fig. 1 right). 

In summary, MNLE can be used to obtain the posterior over parameters of decision-making models, given experimentally observed data. It needs to be trained only once and can then be used for inference in changing inference scenarios, e.g., observed data from different subjects with changing number of trials, hierarchical inference with group-level and subject-level parameter etc..

For more details we refer to our paper ["Flexible and efficient simulation-based inference for models of decision-making"](https://www.biorxiv.org/content/10.1101/2021.12.22.473472v2). 

## Usage

MNLE is implemented as an extension to the widely used python package for simulation-based inference, [`sbi`](https://github.com/mackelab/sbi). In `sbi` there is a tutorial on how to use `MNLE` for SBI with trial-based mixed data (see [here](https://github.com/mackelab/sbi/blob/main/tutorials/14_SBI_with_trial-based_mixed_data.ipynb)). Additionally, we provide a tutorial on how to use MNLE in for decision-making models at [`notebooks/MNLE-Tutorial.ipynb`](notebooks/MNLE-Tutorial.ipynb). 

Additionally, this repository contains the results and the code for reproducing the figures of the MNLE paper. To run the notebooks locally, clone this repository and then run
```python
pip install -r requirements.txt
```
in the `mnle-for-ddms` folder. It will install a custom branch of the `sbibm` benchmarking framework, the current version of `sbi` and additional required packages. 

The [`notebooks`](notebooks) folder contains jupyter notebooks for reproducing the figures presented in the paper.

The research code for MNLE is in [`notebooks/mnle_utils.py`](notebooks/mnle_utils.py).

Please do not hesitate to create an issue if you have questions or encounter problems.
