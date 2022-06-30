# Adapted from: https://github.com/AlexanderFengler/ssm_simulators
# Script to generate DDM data with LANs ssm_simulator package,
# for different simulation budgets.

# Load necessary packages
from copy import deepcopy

import sbibm
import ssms
import torch

# Get benchmark task to load Julia simulator.
seed = torch.randint(100000, (1,)).item()

task = sbibm.get_task("ddm")
prior = task.get_prior_dist()
simulator = task.get_simulator(seed=seed)  # Passing the seed to Julia.

# MAKE CONFIGS
# Initialize the generator config (for MLP LANs)
generator_config = deepcopy(ssms.config.data_generator_config["lan"]["mlp"])
# Specify generative model (one from the list of included models mentioned above)
generator_config["dgp_list"] = "ddm"
# Specify number of parameter sets to simulate:
generator_config["n_parameter_sets"] = 100
# Specify how many samples a simulation run should entail
generator_config["n_samples"] = 1000
# Number of KDE samples to draw from KDE to generate NN targets
generator_config["n_training_samples_by_parameter_set"] = 1000
# Specify folder in which to save generated data
generator_config["output_folder"] = "data/lan_mlp_10_5_^2^3/"
generator_config["n_cpus"] = 1
generator_config["n_subruns"] = 1

# Make model config dict
model_config = ssms.config.model_config["ddm"]

# MAKE DATA
my_dataset_generator = ssms.dataset_generators.data_generator(
    generator_config=generator_config,
    model_config=model_config,
    # Pass our simulator to use our prior.
    julia_simulator=simulator,
)
training_data = my_dataset_generator.generate_data_training_uniform(save=True)
