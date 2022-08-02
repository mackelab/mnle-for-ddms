# Mixed neural likelihood estimation for models of decision-making 

<img src="data/mnle_concept_figure.png"
     alt="Figure 1"
     style="float: left; margin-right: 10px;" />

This repository contains the reproducing code and data for the paper ["Flexible and efficient simulation-based inference for models of decision-making"](https://elifesciences.org/articles/77220), which introduces Mixed neural likelihood estimation (MNLE). 

## Summary

MNLE enables Bayesian parameter inference for models of decision-making in which the likelihood function cannot be evaluated. In this scenario, inference is challenging because first, common approximate inference methods like MCMC or variational inference (VI) cannot be applied (no access to the likelihood), and second, models typically return mixed data types, e.g., categorical choices and continuous reaction times (Fig. 1 left; with the drift-diffusion model as an example). 

MNLE solves both challenges: It is based on the framework on neural likelihood estimation [[1](http://proceedings.mlr.press/v89/papamakarios19a.html)] which uses artificial neural networks to learn a synthetic likelihood using data simulated from the model, which in turn can be used to perform inference with MCMC or VI. In particular, it extends neural likelihood estimation to mixed data types (Fig. 1 middle), such that it can be applied to models of decision making. Once trained, MNLE can be used to obtain posterior samples via MCMC or VI (Fig. 1 right). 

In summary, MNLE can be used to obtain the posterior over parameters of decision-making models, given experimentally observed data. It needs to be trained only once and can then be used for inference in changing inference scenarios, e.g., observed data from different subjects with changing number of trials, hierarchical inference with group-level and subject-level parameter etc.

## Getting started

We set up a short executable tutorial here: http://tinyurl.com/mnle-colab. 

MNLE is implemented as an extension to the widely used python package for simulation-based inference, [`sbi`](https://github.com/mackelab/sbi). Thus, all you need is to `pip install sbi`. In `sbi` there is a tutorial on how to use `MNLE` for SBI with trial-based mixed data (see [here](https://github.com/mackelab/sbi/blob/main/tutorials/14_SBI_with_trial-based_mixed_data.ipynb)). Additionally, we provide a tutorial on how to use MNLE for the drift-diffusion model of decision-making models at [`notebooks/MNLE-Tutorial.ipynb`](notebooks/MNLE-Tutorial.ipynb). 

Please do not hesitate to create an issue if you have questions or encounter problems.

## Installation for reproducing the results

This repository contains the results and the code for reproducing the figures of the MNLE paper. To run the notebooks locally, clone this repository and then run
```python
pip install -r requirements.txt
```
in the `mnle-for-ddms` folder. It will install a custom branch of the `sbibm` benchmarking framework, the current version of `sbi` and additional required packages. 

The [`notebooks`](notebooks) folder contains jupyter notebooks for reproducing the figures presented in the paper.

The research code for MNLE is in [`notebooks/mnle_utils.py`](notebooks/mnle_utils.py).


## Storage of large files via `git-lfs`

This repository contains large files, e.g., the training data and the pre-trained neural networks for rerpoducing all results presented in the paper. These files are stored using [`git-lfs`](https://git-lfs.github.com). When your goal is to download the entire repository including all large files, please make sure to have `git-lfs` installed locally. 

## Citation
```
@article{10.7554/eLife.77220,
     article_type = {journal},
     title = {Flexible and efficient simulation-based inference for models of decision-making},
     author = {Boelts, Jan and Lueckmann, Jan-Matthis and Gao, Richard and Macke, Jakob H},
     editor = {Wyart, Valentin},
     volume = 11,
     year = 2022,
     month = {jul},
     pub_date = {2022-07-27},
     pages = {e77220},
     citation = {eLife 2022;11:e77220},
     doi = {10.7554/eLife.77220},
     url = {https://doi.org/10.7554/eLife.77220},
     journal = {eLife},
     issn = {2050-084X},
     publisher = {eLife Sciences Publications, Ltd},
}
```
