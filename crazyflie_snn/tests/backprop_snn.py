# SPDX-FileCopyrightText: 2021 Stein Stroobants <s.stroobants@tudelft.nl>
#
# SPDX-License-Identifier: MIT

###############################################################################
# Import packages
###############################################################################

from datetime import datetime
import yaml
import os
from argparse import ArgumentParser
import torch
import wandb

from crazyflie_snn.dataloader.dataloader import load_datasets
from crazyflie_snn.training.bp_snn import fit
from crazyflie_snn.neural_models.models import models
from spiking.core.torch.model import get_model
from crazyflie_snn.training.loss_functions import loss_functions

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_INIT_TIMEOUT"] = "240"
os.environ["WANDB__DISABLE_SERVICE"] = "True"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bp_onelayer_snn_control_crazyflie_control_from_att.yaml")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    # Config from yaml file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config["device"] == 'cuda':
        torch.cuda.empty_cache()

    device = config["device"]

    # Load training and validation dataloaders
    train_loader, val_loader, _ = load_datasets(config)
    
    # Build the spiking model
    x, _ = next(iter(train_loader))
    if config["net"]["type"] == "ANN":
        model = get_model(models[config["net"]["type"]], config["net"]["params"], data=None, device=device)
    else: 
        model = get_model(models[config["net"]["type"]], config["net"]["params"], data=x[:, 0], device=device)
    if config["net"]["fix_weights"] is not False:
        print("Fixing weights")
        model.set_weights(config["net"]["fix_weights"], device)
    # model = models[config["net"]["type"]](**config["net"]["params"]).to(device)

    # If the checkpoint is specified, load the weights
    if config["net"]["checkpoint"] is not False:
        model_path = f"runs/{config['net']['checkpoint']}/model.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Use Adam optimizer 
    optimizer = torch.optim.RAdam(model.parameters(), lr=config["train"]["learning_rate"])

    # Use loss from config
    loss_function = loss_functions[config["train"]["loss"]]

    # initialize weights and biases
    wandb_mode = "online" if config["logging"] else "disabled"
    run = wandb.init(project=config["wandb"]["project"], entity=config["wandb"]["entity"], config=config, mode=wandb_mode, notes=args.notes)
    wandb.watch(model, log_freq=50, log="all")

    ################################################################################
    ## Create output folder and saving config to file if logging is enabled
    ################################################################################

    if config["logging"]:
        dt_string = run.name
        out_dir = f'runs/{dt_string}'
        os.mkdir(out_dir)
        config_save_file = f'{out_dir}/config.txt'
        with open(config_save_file, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
    else:
        out_dir = None

    ################################################################################
    ## Train model
    ################################################################################

    print("Starting training")

    best_model, best_fitness, ngens, avg_time, fitness_hist, val_hist = fit(model, train_loader, val_loader, optimizer, config, out_dir, loss_function)

    print(f'Best obtained loss was : {best_fitness}')

    ################################################################################
    ## Saving results to file
    ################################################################################

    if config["logging"]:
        results_save_file = f'{out_dir}/results.txt'
        results = {
                    "best_loss": float(best_fitness),
                    "avg_gen_calc_time": avg_time,
                    "number_generations": ngens, 
                    "fitness_hist": fitness_hist,
                    "validation_hist": val_hist
        }

        with open(results_save_file, 'w') as f:
            yaml.dump(results, f, sort_keys=False)

        model_path = f'{out_dir}/model.pt'
        # Log the model as an artifact
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    # Finish the run
    wandb.finish()