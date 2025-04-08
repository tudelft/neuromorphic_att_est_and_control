# SPDX-FileCopyrightText: 2021 Stein Stroobants <s.stroobants@tudelft.nl>
#
# SPDX-License-Identifier: MIT

###############################################################################
# Import packages
###############################################################################

import yaml
from argparse import ArgumentParser
import torch
import matplotlib.pyplot as plt

from crazyflie_snn.dataloader.dataloader import load_datasets
from crazyflie_snn.neural_models.models import models
from spiking.core.torch.model import get_model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bp_softsnn_control_crazyflie_control_from_att.yaml")
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
    model = get_model(models[config["net"]["type"]], config["net"]["params"], data=x[:, 0], device=device)
    model.set_weights(100)

    # Perform one forward pass
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    data = data.permute(1, 0, 2)
    target = target.permute(1, 0, 2)
    output = model.forward_sequence(data)

    # Plot the results
    n_plots = target.size()[2] + 2
    # for i in range(n_plots - 2):
    #     plt.subplot(n_plots, 1, i + 1)
    plt.subplot(n_plots, 1, 1)
    plt.plot(output[:, 0, 0].cpu().detach().numpy(), label="snn")
    plt.plot(target[:, 0, 0].cpu().detach().numpy(), label="target")
    plt.ylim([-1, 1])
    plt.legend()

    plt.subplot(n_plots, 3, 4)
    plt.hist(model.l1.state[2][0].flatten().cpu().detach().numpy(), bins=80, label="i")
    plt.subplot(n_plots, 3, 5)
    plt.hist(model.l1.state[2][1].flatten().cpu().detach().numpy(), bins=80, label="v")
    plt.subplot(n_plots, 3, 6)
    # plt.hist(model.l1.state[2][2].flatten().cpu().detach().numpy(), bins=80, label="s")
    plt.plot(model.l1.state[2][2].sum(dim=0).cpu().detach().numpy(), '.', label="s")
    plt.ylim([0, 100])

    plt.subplot(n_plots, 2, 5 )
    plt.hist(model.l1.state[0][0].flatten().cpu().detach().numpy(), bins=80, label="ff")
    plt.subplot(n_plots, 2, 6 )
    plt.hist(model.l1.state[1][0].flatten().cpu().detach().numpy(), bins=80, label="rec")
    plt.show()

