# SPDX-FileCopyrightText: 2023 Stein Stroobants <s.stroobants@tudelft.nl>
#
# SPDX-License-Identifier: MIT

###############################################################################
# Import packages
###############################################################################
from datetime import datetime
import yaml
import os
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats

from crazyflie_snn.dataloader.dataloader import load_datasets
from crazyflie_snn.neural_models.models import models
from crazyflie_snn.training.loss_functions import loss_functions, mse_pearson, pearson
from spiking.core.torch.model import get_model

run_name = "snn_backprop27092023_112523"
# run_name = "snn_backprop28092023_162128" # only roll/pitch as inputs
run_name = "snn_backprop03102023_150033" # px4 control data
run_name = "snn_backprop29112023_151454" # new config 
run_name = "snn_backprop04122023_215552" # new config with torque commands
run_name = "snn_backprop06122023_221021" # rate efficient_universe
run_name = "snn_backprop08012024_151516" # torque avid_river
run_name = "snn_backprop07122023_231736" # torque trim_forest
# run_name = "snn_backprop08012024_174911" # torque smart_snowball
run_name = "snn_backprop08012024_181706" # torque pleasant_morning
run_name = "snn_backprop08122023_224225" # rate decent snowball
run_name = "ancient-glade-149"
# run_name = "stilted-microwave-129"    
# run_name = "super-plant-138"
run_name = "wandering-dew-168"
run_name = "valiant-sunset-166"
# run_name = "astral-cosmos-135"
run_name = "scarlet-durian-175"
run_name = "warm-armadillo-179"
# run_name = "lucky-paper-187"
# run_name = "burning-darling-192"
# run_name = "unique-flower-282"
# run_name = "earthy-capybara-285"
# run_name = "wobbly-blaze-293"
# run_name = "rose-butterfly-296"
# run_name = "peachy-sponge-291"
run_name = "lively-vortex-308"
run_name = "denim-firefly-306"
run_name = "apricot-yogurt-310"
run_name = "revived-night-309"
run_name = "faithful-cloud-311"
run_name = "misunderstood-snow-312"
run_name = "lilac-wood-313"
# run_name = "sandy-hill-316"
# run_name = "crimson-dew-317"
# run_name = "desert-snow-318"
# run_name = "exalted-durian-315"
# run_name = "blooming-vortex-321"


MASK = False
thresh = 0.001

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_folder", type=str, default=run_name)
    args = parser.parse_args()

    # Config from yaml file
    with open(f"runs/{args.config_folder}/config.txt", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config["device"] == 'cuda':
        torch.cuda.empty_cache()

    NGENS = config["train"]["gens"]
    config["device"] = "cpu"
    device = config["device"]

    config["data"]["dir"] = "data/crazyflie_v4_ratesnn_v6_snn"
    config["data"]["seq_length"] = 5000
    config["data"]["target_shift"] = 0
    config["data"]["inter_seq_dist"] = 400
    config["data"]["train_batch_size"] = 10
    config["data"]["val_batch_size"] = 20
    config["data"]["n_repeats"] = 1

    # Load training and validation dataloaders
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    train_loader, val_loader, test_loader = load_datasets(config, generator=generator)

    # Build the spiking model
    x, _ = next(iter(train_loader))
    model = get_model(models[config["net"]["type"]], config["net"]["params"], data=x[:, 0], device=device)
    # model.set_weights(4, device)

    # Load pretrained model
    model.load_state_dict(torch.load(f"runs/{run_name}/model.pt"))

    # load mask
    model.eval()

    if MASK:
        mask = np.loadtxt(f"runs/{run_name}/mask.csv", delimiter=",")
        model.layers[1].synapse.weight = torch.nn.Parameter(model.layers[1].synapse.weight * torch.tensor(mask).type(torch.float))
        print(f"Masking {len(mask) - np.sum(mask)} neurons")

    # Perform inference
    data, target = next(iter(val_loader))
    data = data.permute(1, 0, 2)
    target = target.permute(1, 0, 2)
    output = []
    spikes = []
    for i in range(len(data)):
        output.append(model(data[i, :, :]))
        spikes.append(model.layers[0].state[2][2][0, :])
    output = torch.stack(output)
    spikes = torch.stack(spikes)

    # Plot spikes
    # plt.subplot(3, 1, 3)
    for i in range(spikes.size()[1]):
        spikes_t = spikes[:, i].cpu().detach().clone().numpy()
        spikes_t[spikes_t == 0] = np.nan

    # plot the spikes for 1 second as little lines and also plot horizontal lines for each neuron
    plt.figure(figsize=(4, 1.7))
    for i in range(spikes.size()[1]):
        spikes_t = spikes[:, i].cpu().detach().clone().numpy()
        spikes_t[spikes_t == 0] = np.nan
        plt.plot(spikes_t + i + 0.1, '|', color="#27282b", markersize=5, markeredgewidth=1)
        plt.plot([0, 5000], [i - 0.28, i - 0.28], color="#27282b", linewidth=0.4)
    plt.ylim(41.8, 56.8)
    plt.yticks([])
    plt.xlim(3490.5, 3580.2)
    plt.xticks([])
    plt.axis('off')
    # save as vector
    # plt.savefig("spikes.svg", format="svg")
    # save as png
    plt.savefig("spikes.png", format="png", dpi=2000, transparent=True)

    
    plt.show()