# Mixed neural likelihood estimation for models of decision-making

<img src="data/figure_1_mixed.png"
     alt="Figure 1"
     style="float: left; margin-right: 10px;" />

Mixed neural likelihood estimation (MNLE) enables Bayesian parameter inference for models of decision-making in which the likelihood function cannot be evaluated. In this scenario, inference is challenging because first, common approximate inference methods like MCMC or variational inference (VI) cannot be applied (no access to the likelihood), and second, models typically return mixed data types, e.g., categorical choices and continuous reaction times (Fig. 1 left; with the drift-diffusion model as an example). 

MNLE solves both challenges: It is based on the framework on neural likelihood estimation ([1](http://proceedings.mlr.press/v89/papamakarios19a.html)) which uses artificial neural networks to learn a synthetic likelihood using data simulated from the model, which in turn can be used to perform inference with MCMC or VI. Crucially, it extends this neural likelihood estimation to mixed data types, such that it can be applied to models of decision making (Fig. 1 middle). Once trained, MNLE can be used to obtain posterior samples via MCMC or VI (Fig. 1 right). 

In summary, you can use MNLE to obtain the posterior over parameters of your decision-making model, given experimentally observed data. MNLE needs to be trained only once and can then be used for inference in changing inference scenarios, e.g., observed data from different subject with changing number of trials, hierarchical inference with group-level and subject-level parameter etc..

For more details we refer to our paper ["Flexible and efficient simulation-based inference for models of decision-making"](https://www.biorxiv.org/content/10.1101/2021.12.22.473472v1). Feel free to create an issue if you have any questions. 

## Content

For now, this repository contains the research code for MNLE. However, it will be integrated into an established toolbox for simulation-based inference ([`sbi`](https://github.com/mackelab/sbi)) early 2022.

The [`notebooks`](notebooks) folder contains jupyter notebooks for reproducing the figures presented in the paper.

The core code for MNLE is in [`notebooks/mnle_utils.py`](notebooks/mnle_utils.py).

Please do not hesitate to create an issue if you have questions or encounter problems.

## Example
We give a detailed example of how to apply MNLE to the standard drift-diffusion model of decision making in [`notebooks/MNLE-DDM-example.ipynb`](notebooks/MNLE-Example.ipynb).
