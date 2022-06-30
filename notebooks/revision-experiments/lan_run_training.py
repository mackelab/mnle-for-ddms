# Adapted from: https://github.com/AlexanderFengler/LANfactory
# Script to run LAN training given a pre-simulated data set.

import os
from copy import deepcopy
from pathlib import Path

# Load necessary packages
import lanfactory
import torch
from lanfactory.trainers import ModelTrainerTorchMLP

BASE_DIR = Path.cwd()

# NOTE: set budget and n_samples, e.g., for 10k budget set 10_4 and n_samples 100
# for 100k budget set 10_5 and n_samples 1000
budget = "10_5_ours"
n_samples = 100
num_repeats = 10
num_epochs = 20

# NOTE: The resulting trainiend LAN will be saved with a unique ID under torch_models/ddm_{budget}.

for _ in range(num_repeats):

    # MAKE DATALOADERS

    # List of datafiles (here only one)
    folder_ = f"data/lan_mlp_{budget}/training_data_0_nbins_0_n_{n_samples}/ddm/"
    file_list_ = [folder_ + file_ for file_ in os.listdir(folder_)]
    assert len(file_list_) == 1, "we must use only one training data file."

    # Training dataset
    torch_training_dataset = lanfactory.trainers.DatasetTorch(
        file_IDs=file_list_, batch_size=128
    )

    torch_training_dataloader = torch.utils.data.DataLoader(
        torch_training_dataset,
        shuffle=True,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
    )

    # Validation dataset
    torch_validation_dataset = lanfactory.trainers.DatasetTorch(
        file_IDs=file_list_, batch_size=128
    )

    torch_validation_dataloader = torch.utils.data.DataLoader(
        torch_validation_dataset,
        shuffle=True,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
    )

    # SPECIFY NETWORK CONFIGS AND TRAINING CONFIGS

    network_config = lanfactory.config.network_configs.network_config_mlp
    train_config = lanfactory.config.network_configs.train_config_mlp
    train_config["n_epochs"] = num_epochs

    # LOAD NETWORK
    net = lanfactory.trainers.TorchMLP(
        network_config=deepcopy(network_config),
        input_shape=torch_training_dataset.input_dim,
        save_folder="/data/torch_models",
        generative_model_id=f"ddm_{budget}",
    )

    # SAVE CONFIGS
    lanfactory.utils.save_configs(
        model_id=net.model_id + "_torch_",
        save_folder=f"data/torch_models/ddm_{budget}",
        network_config=network_config,
        train_config=train_config,
        allow_abs_path_folder_generation=True,
    )

    # TRAIN MODEL
    output_folder = (BASE_DIR / "data/torch_models").as_posix()

    model_trainer = ModelTrainerTorchMLP(
        train_config,
        torch_training_dataloader,
        torch_validation_dataloader,
        model=net,
        output_folder=output_folder,
        allow_abs_path_folder_generation=True,
    )

    model_trainer.train_model(save_history=True, save_model=True, verbose=0)
