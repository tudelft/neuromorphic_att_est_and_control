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
import subprocess

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats

from crazyflie_snn.dataloader.dataloader import load_datasets
from crazyflie_snn.neural_models.models import models
from crazyflie_snn.training.loss_functions import loss_functions, mse_pearson, pearson, mse_skip_init
from spiking.core.torch.model import get_model

# run_name = "snn_backprop27092023_112523"
# # run_name = "snn_backprop28092023_162128" # only roll/pitch as inputs
# run_name = "snn_backprop03102023_150033" # px4 control data
# run_name = "snn_backprop29112023_151454" # new config 
# run_name = "snn_backprop04122023_215552" # new config with torque commands
# run_name = "snn_backprop06122023_221021" # rate efficient_universe
# run_name = "snn_backprop08012024_151516" # torque avid_river
# run_name = "snn_backprop07122023_231736" # torque trim_forest
# run_name = "snn_backprop08012024_174911" # torque smart_snowball
# run_name = "snn_backprop08012024_181706" # torque pleasant_morning
# run_name = "stilted-microwave-129"
# run_name = "super-plant-138"
# run_name = "happy-blaze-152"
# run_name = "misunderstood-bird-153"
# run_name = "sandy-sun-154"
# run_name = "astral-cosmos-135"
# run_name = "worldly-forest-160"
# run_name = "cosmic-oath-161"
# run_name = "vague-pine-162"
# run_name = "wandering-dew-168"
# run_name = "scarlet-durian-175"``
run_name = "warm-armadillo-179"
# run_name = "grateful-monkey-178"
# run_name = "winter-valley-177"
# run_name = "absurd-snow-174"
# run_name = "abundant-moon-184"
# run_name = "burning-darling-192"
# run_name = "glistening-firecracker-193"
# run_name = "vermilion-fish-196"

MASK = True
# torch.set_printoptions(precision=2, sci_mode=False, profile="full", linewidth=160)

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
    device = "cpu"

    config["data"]["dir"] = "data/crazyflie_v5"
    config["data"]["train_batch_size"] = 10
    config["data"]["val_batch_size"] = 10
    config["data"]["seq_length"] = 7000
    config["data"]["target_shift"] = 0
    # config["data"]["n_repeats"] = 2
    config["data"]["remove_integral"] = False
    config["data"]["remove_initial_value"] = False
    config["data"]["target_integral"] = False


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
        if len(model.layers) == 3:   
            mask_1 = np.loadtxt(f"runs/{run_name}/mask_1.csv", delimiter=",")
            model.layers[1].synapse_ff.weight = torch.nn.Parameter(model.layers[1].synapse_ff.weight * torch.tensor(mask_1).type(torch.float))
            print(f"Layer 1: masking {len(mask_1) - np.sum(mask_1)} neurons")
            mask_2 = np.loadtxt(f"runs/{run_name}/mask_2.csv", delimiter=",")
            model.layers[2].synapse.weight = torch.nn.Parameter(model.layers[2].synapse.weight * torch.tensor(mask_2).type(torch.float))
            print(f"Layer 2: masking {len(mask_2) - np.sum(mask_2)} neurons")
        else:
            mask = np.loadtxt(f"runs/{run_name}/mask.csv", delimiter=",")
            model.layers[1].synapse.weight = torch.nn.Parameter(model.layers[1].synapse.weight * torch.tensor(mask).type(torch.float))

            model.layers[0].synapse_rec.weight = torch.nn.Parameter(model.layers[0].synapse_rec.weight * torch.tensor(mask).type(torch.float).expand(len(mask), len(mask)))            # model.layers[0].synapse_rec.weight = torch.nn.Parameter(model.layers[0].synapse_rec.weight[torch.tensor(mask)==1, :][:, torch.tensor(mask)==1])
            print(f"Masking {len(mask) - np.sum(mask)} neurons")
    
    # print(torch.sigmoid(model.layers[0].neuron.leak_i * torch.tensor(mask).type(torch.float)))
    # print(torch.sigmoid(model.layers[0].neuron.leak_v * torch.tensor(mask).type(torch.float)))
    # Perform inference
    data, target = next(iter(test_loader))
    data = data.permute(1, 0, 2)
    target = target.permute(1, 0, 2)
    output = []
    spikes = []
    for i in range(len(data)):
        output.append(model(data[i, :, :]))
        # print(model.layers[0].state[2][0][0, :]* torch.tensor(mask).type(torch.float))
        # spikes.append(model.layers[1].state[0][2][0, :])
    output = torch.stack(output)
    # spikes = torch.stack(spikes)
    # print(spikes)

    np.savetxt("input.csv", data[:, 0, :].cpu().detach().numpy(), delimiter=",", fmt='%1.5f')
    np.savetxt("output.csv", output[:, 0, :].cpu().detach().numpy(), delimiter=",", fmt='%1.5f')

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(output[:, 0, 0].cpu().detach().numpy(), label="SNN Pitch")
    axes[0].plot(target[:, 0, 0].cpu().detach().numpy(), label="Target Pitch")
    # plt.plot(output[:, 0, 2].cpu().detach().numpy()*(5/3), label="SNN gyro")
    # plt.plot(target[:, 0, 2].cpu().detach().numpy()*(5/3), label="Target gyro")
    # plt.plot(data[:, 0, 2].cpu().detach().numpy()*0.5, label="Roll velocity")
    # plt.plot(data[:, 0, 0].cpu().detach().numpy()*5, label="Roll velocity")
    axes[0].legend()

    # plt.subplot(2, 1, 1)
    # plt.plot(output[:, 0, 2].cpu().detach().numpy(), label="SNN gyro")
    # plt.plot(target[:, 0, 2].cpu().detach().numpy(), label="Target gyro")
    # # plt.plot(data[:, 0, 0].cpu().detach().numpy(), label="Roll")
    # # plt.plot(data[:, 0, 2].cpu().detach().numpy(), label="Roll velocity")
    # # plt.plot(data[:, 0, 4].cpu().detach().numpy(), label="Target roll")
    # plt.legend()


    # plt.subplot(3, 1, 2)
    axes[1].plot(output[:, 0, 1].cpu().detach().numpy(), label="SNN Roll")
    axes[1].plot(target[:, 0, 1].cpu().detach().numpy(), label="Target Roll")
    # plt.plot(-output[:, 0, 3].cpu().detach().numpy()*(5/3), label="SNN gyro")
    # plt.plot(-target[:, 0, 3].cpu().detach().numpy()*(5/3), label="Target gyro")
    # plt.plot(data[:, 0, 3].cpu().detach().numpy()*0.5, label="Pitch velocity")
    # plt.plot(-data[:, 0, 1].cpu().detach().numpy()*5, label="Pitch velocity")
    axes[1].legend()

    # axes[2].plot(output[:, 0, 2].cpu().detach().numpy(), label="SNN yaw")
    # axes[2].plot(target[:, 0, 2].cpu().detach().numpy(), label="Target yaw")
    # # plt.plot(-output[:, 0, 3].cpu().detach().numpy()*(5/3), label="SNN gyro")
    # # plt.plot(-target[:, 0, 3].cpu().detach().numpy()*(5/3), label="Target gyro")
    # # plt.plot(data[:, 0, 3].cpu().detach().numpy()*0.5, label="Pitch velocity")
    # # plt.plot(-data[:, 0, 1].cpu().detach().numpy()*5, label="Pitch velocity")
    # axes[2].legend()

    # plt.subplot(4, 1, 4)
    # plt.plot(output[:, 0, 3].cpu().detach().numpy(), label="SNN gyro")
    # plt.plot(target[:, 0, 3].cpu().detach().numpy(), label="Target gyro")
    # # plt.plot(data[:, 0, 1].cpu().detach().numpy(), label="Pitch")
    # # plt.plot(data[:, 0, 3].cpu().detach().numpy(), label="Pitch velocity")
    # # plt.plot(data[:, 0, 5].cpu().detach().numpy(), label="Target pitch")
    # plt.legend()
    # plt.show()

        # Plot results
    # plt.subplot(4, 1, 1)
    # plt.plot(output[:, 0, 0].cpu().detach().numpy(), label="SNN Roll")
    # plt.plot(target[:, 0, 0].cpu().detach().numpy(), label="Target Roll Torque")
    # plt.plot(data[:, 0, 0].cpu().detach().numpy()*5, label="Roll velocity")
    # plt.legend()

    # plt.subplot(4, 1, 2)
    # plt.plot(data[:, 0, 0].cpu().detach().numpy(), label="gyro roll")
    # plt.plot(data[:, 0, 6].cpu().detach().numpy(), label="Target roll")
    # plt.legend()

    # plt.subplot(4, 1, 3)
    # plt.plot(output[:, 0, 1].cpu().detach().numpy(), label="SNN Pitch")
    # plt.plot(target[:, 0, 1].cpu().detach().numpy(), label="Target Pitch")
    # # plt.plot(data[:, 0, 3].cpu().detach().numpy()*0.5, label="Pitch velocity")
    # plt.plot(-data[:, 0, 1].cpu().detach().numpy()*5, label="Pitch velocity")
    # plt.legend()

    # plt.subplot(4, 1, 4)
    # plt.plot(data[:, 0, 1].cpu().detach().numpy(), label="Pitch")
    # plt.plot(data[:, 0, 7].cpu().detach().numpy(), label="Target pitch")
    # plt.legend()


    # Plot spikes
    # plt.subplot(3, 1, 3)
    # for i in range(spikes.size()[1]):
    #     spikes_t = spikes[:, i].cpu().detach().clone().numpy()
    #     spikes_t[spikes_t == 0] = np.nan
    #     axes[2].plot(spikes_t + i + 0.1, '.', markersize=0.5)

    # plt.plot(spikes.cpu().detach().numpy(), '.')

    # plot amount of spikes per neuron
    # plt.figure()
    # n_spikes = torch.nansum(spikes, dim=0).cpu().detach().numpy()
    # # normalize
    # n_spikes = n_spikes / 6000
    # plt.plot(n_spikes, '.')

    # plot contribution of neurons to output
    # res = torch.einsum('ij,jk->ijk', spikes, model.layers[1].synapse.weight.T)

    # contrib = res.sum(dim=0).abs().sum(dim=1).cpu().detach().numpy()
    # normalize
    # contrib = contrib / contrib.max()

    # plt.bar(np.arange(80), n_spikes, align='edge', width=0.45)
    # plt.bar(np.arange(80), contrib, align='edge', width=-0.45)


    # Plot 2d plot of the weights
    # plt.figure()
    # plt.imshow(model.layers[0].synapse_ff.weight.cpu().detach().numpy())
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(model.layers[0].synapse_rec.weight.cpu().detach().numpy())
    # plt.colorbar()

    # # Plot 2d plot of the weights
    # plt.figure()
    # plt.imshow(model.layers[1].synapse.weight.T.cpu().detach().numpy())
    # plt.colorbar()

    print(f"loss: {mse_skip_init(output, target)}")

    for i in range(0, 10):
        if i == 0:
            pearson_loss = pearson(target[:, 0, :], output[:, 0, :])
            # pearson_loss = 1 - scipy.stats.pearsonr(target[:, 0, 0].detach().numpy(), output[:, 0, 0].detach().numpy())[0]
        else:
            pearson_loss = pearson(target[:-i, 0, :], output[i:, 0, :])
            # pearson_loss = 1 - scipy.stats.pearsonr(target[:-i, 0, 0].detach().numpy(), output[i:, 0, 0].detach().numpy())[0]
        print(f"Pearson loss: {pearson_loss}")


    
    
    plt.show()