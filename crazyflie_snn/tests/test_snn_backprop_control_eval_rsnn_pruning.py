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
from crazyflie_snn.training.loss_functions import loss_functions, mse_pearson, pearson, mse_skip_init, smooth_l1_skip_init
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
# run_name = "wandering-dew-168"
# run_name = "valiant-sunset-166"
# run_name = "astral-cosmos-135"
run_name = "worldly-forest-160"
run_name = "absurd-snow-174"
# run_name = "distinctive-capybara-169"
# run_name = "lambent-paper-185"
run_name = "abundant-moon-184"
# run_name = "alight-ox-189"
# run_name = "denim-snow-283"
# run_name = "driven-firefly-297"
# run_name = "gallant-armadillo-298"
run_name = "vague-dragon-304"

MASK = False
thresh = 0.01

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
    # device = config["device"]
    device = "cpu"

    config["data"]["dir"] = "data/crazyflie_v4_ratesnn"
    config["data"]["val_batch_size"] = 10
    config["data"]["seq_length"] = 2000
    config["data"]["target_shift"] = 0
    config["data"]["inter_seq_dist"] = 400
    config["data"]["train_batch_size"] = 50
    # config["data"]["n_repeats"] = 1

    # Load training and validation dataloaders
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    train_loader, val_loader, test_loader = load_datasets(config, generator=generator)

    # Build the spiking model
    x, _ = next(iter(train_loader))
    model = get_model(models[config["net"]["type"]], config["net"]["params"], data=x[:, 0], device=device)

    # Load pretrained model
    model.load_state_dict(torch.load(f"runs/{run_name}/model.pt"))

    # load mask
    model.eval()
    if MASK:
        mask_1 = np.loadtxt(f"runs/{run_name}/mask_1.csv", delimiter=",")
        model.layers[1].synapse_ff.weight = torch.nn.Parameter(model.layers[1].synapse_ff.weight * torch.tensor(mask_1).type(torch.float))
        print(f"Layer 1: masking {len(mask_1) - np.sum(mask_1)} neurons")
        mask_2 = np.loadtxt(f"runs/{run_name}/mask_2.csv", delimiter=",")
        model.layers[2].synapse.weight = torch.nn.Parameter(model.layers[2].synapse.weight * torch.tensor(mask_2).type(torch.float))
        print(f"Layer 2: masking {len(mask_2) - np.sum(mask_2)} neurons")

    # Perform inference
    data, target = next(iter(val_loader))
    data = data.permute(1, 0, 2)
    target = target.permute(1, 0, 2)
    output = []
    spikes_1 = []
    spikes_2 = []
    for i in range(len(data)):
        output.append(model(data[i, :, :]))
        spikes_1.append(model.layers[0].state[1][2][0, :])
        spikes_2.append(model.layers[1].state[2][2][0, :])
    output = torch.stack(output)
    spikes_1 = torch.stack(spikes_1)
    spikes_2 = torch.stack(spikes_2)

    # Plot results for all sequences in batch
    for i in range(config["data"]["train_batch_size"]):
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(output[:, i, 0].cpu().detach().numpy(), label="SNN Roll")
        axes[0].plot(target[:, i, 0].cpu().detach().numpy(), label="Target Roll")
        axes[0].legend()

        axes[1].plot(output[:, i, 1].cpu().detach().numpy(), label="SNN Pitch")
        axes[1].plot(target[:, i, 1].cpu().detach().numpy(), label="Target Pitch")
        axes[1].legend()
        # plt.show()
        break

    # Plot spikes
    # plt.subplot(3, 1, 3)
    for i in range(spikes_2.size()[1]):
        spikes_t = spikes_2[:, i].cpu().detach().clone().numpy()
        spikes_t[spikes_t == 0] = np.nan
        axes[2].plot(spikes_t + i + 0.1, '.', markersize=0.5)

    # plt.plot(spikes.cpu().detach().numpy(), '.')
    def calculate_contribution(spikes, weights):
        # plot amount of spikes per neuron
        n_spikes = torch.nansum(spikes, dim=0).cpu().detach().numpy()
        # normalize
        n_spikes = n_spikes / 8000
        # plot contribution of neurons to output
        res = torch.einsum('ij,jk->ijk', spikes, weights)
        contrib = res.sum(dim=0).abs().sum(dim=1).cpu().detach().numpy()
        # normalize
        contrib = contrib / contrib.max()
        return n_spikes, contrib

    n_spikes_1, contrib_1 = calculate_contribution(spikes_1, model.layers[1].synapse_ff.weight.T)
    n_spikes_2, contrib_2 = calculate_contribution(spikes_2, model.layers[2].synapse.weight.T)

    plt.figure()
    plt.bar(np.arange(config["net"]["params"]["l1"]["synapse"]["out_features"]), n_spikes_1, align='edge', width=0.45)
    plt.bar(np.arange(config["net"]["params"]["l1"]["synapse"]["out_features"]), contrib_1, align='edge', width=-0.45)
    # plt.plot(contrib, '.')

    plt.figure()
    plt.bar(np.arange(config["net"]["params"]["l2"]["synapse_ff"]["out_features"]), n_spikes_2, align='edge', width=0.45)
    plt.bar(np.arange(config["net"]["params"]["l2"]["synapse_ff"]["out_features"]), contrib_2, align='edge', width=-0.45)

    # plt.figure()
    # plt.plot(torch.nansum(spikes, dim=1).cpu().detach().numpy(), '.')
    # print(contrib > 0.005)
    if not MASK:
        print(f"Layer 1: Saving a mask of {np.sum(contrib_1 > thresh)} neurons")
        np.savetxt(f"runs/{run_name}/mask_1.csv", contrib_1 > thresh, delimiter=",")
        print(f"Layer 2: Saving a mask of {np.sum(contrib_2 > thresh)} neurons")
        np.savetxt(f"runs/{run_name}/mask_2.csv", contrib_2 > thresh, delimiter=",")

    for i in range(0, 8):
        if i == 0:
            pearson_loss = pearson(target[:, 0, :], output[:, 0, :])
            # pearson_loss = 1 - scipy.stats.pearsonr(target[:, 0, 0].detach().numpy(), output[:, 0, 0].detach().numpy())[0]
        else:
            pearson_loss = pearson(target[:-i, 0, :], output[i:, 0, :])
            # pearson_loss = 1 - scipy.stats.pearsonr(target[:-i, 0, 0].detach().numpy(), output[i:, 0, 0].detach().numpy())[0]
        print(f"Pearson loss: {pearson_loss}")

    # Print the loss
    loss = mse_skip_init(target, output)
    print(f"Loss: {loss}")

    np.savetxt("input.csv", data[:, 0, :].cpu().detach().numpy(), delimiter=",", fmt='%1.5f')
    np.savetxt("output.csv", output[:, 0, :].cpu().detach().numpy(), delimiter=",", fmt='%1.5f')

    # Print average spike rate per timestep per neuron 
    print(f"Average spike rate per neuron layer 1: {torch.nansum(spikes_1).cpu().detach().numpy() / (spikes_1.size()[0] * spikes_1.size()[1])}")
    print(f"Average spike rate per neuron layer 2: {torch.nansum(spikes_2).cpu().detach().numpy() / (spikes_2.size()[0] * spikes_2.size()[1])}")

    
    # Plot historam of the input data per input feature
    plt.figure()
    for i in range(data.size()[2]):
        plt.hist(data[:, 0, i].cpu().detach().numpy(), bins=100, alpha=0.5, label=f"Input {i}")
    plt.legend()

    # plot the spikes for the 

    
    plt.show()