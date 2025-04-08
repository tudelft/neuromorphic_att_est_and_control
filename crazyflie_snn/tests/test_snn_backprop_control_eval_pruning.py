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

    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(output[:, 0, 0].cpu().detach().numpy(), label="SNN Roll")
    axes[0].plot(target[:, 0, 0].cpu().detach().numpy(), label="Target Roll")
    # plt.plot(output[:, 0, 2].cpu().detach().numpy()*(5/3), label="SNN gyro")
    # plt.plot(target[:, 0, 2].cpu().detach().numpy()*(5/3), label="Target gyro")
    # plt.plot(data[:, 0, 2].cpu().detach().numpy()*0.5, label="Roll velocity")
    # plt.plot(data[:, 0, 0].cpu().detach().numpy()*5, label="Roll velocity")
    axes[0].legend()


    axes[1].plot(output[:, 0, 1].cpu().detach().numpy(), label="SNN Pitch")
    axes[1].plot(target[:, 0, 1].cpu().detach().numpy(), label="Target Pitch")
    axes[1].legend()

    axes[2].plot(output[:, 0, 2].cpu().detach().numpy(), label="SNN Yaw")
    axes[2].plot(target[:, 0, 2].cpu().detach().numpy(), label="Target Yaw")
    axes[2].legend()

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
    for i in range(spikes.size()[1]):
        spikes_t = spikes[:, i].cpu().detach().clone().numpy()
        spikes_t[spikes_t == 0] = np.nan
        axes[3].plot(spikes_t + i + 0.1, '.', markersize=0.5)

    # plt.plot(spikes.cpu().detach().numpy(), '.')

    # plot amount of spikes per neuron
    plt.figure()
    n_spikes = torch.nansum(spikes, dim=0).cpu().detach().numpy()
    # normalize
    n_spikes = n_spikes / 4000
    # plt.plot(n_spikes, '.')

    # plot contribution of neurons to output
    res = torch.einsum('ij,jk->ijk', spikes, model.layers[1].synapse.weight.T)

    contrib = res.sum(dim=0).abs().sum(dim=1).cpu().detach().numpy()
    # normalize
    contrib = contrib / contrib.max()

    plt.bar(np.arange(config["net"]["params"]["l1"]["synapse_ff"]["out_features"]), n_spikes, align='edge', width=0.45)
    plt.bar(np.arange(config["net"]["params"]["l1"]["synapse_ff"]["out_features"]), contrib, align='edge', width=-0.45)
    # plt.plot(contrib, '.')
    # plt.figure()
    # plt.plot(torch.nansum(spikes, dim=1).cpu().detach().numpy(), '.')
    # print(contrib > 0.01)

    if not MASK:
        print(f"Saving a mask of {np.sum(contrib > thresh)} neurons")
        np.savetxt(f"runs/{run_name}/mask.csv", contrib > thresh, delimiter=",")

    for i in range(0, 8):
        if i == 0:
            pearson_loss = pearson(target[:, 0, :2], output[:, 0, :2])
            # pearson_loss = 1 - scipy.stats.pearsonr(target[:, 0, 0].detach().numpy(), output[:, 0, 0].detach().numpy())[0]
            mse_loss = loss_functions["mse"](target[:, 0, :2], output[:, 0, :2])
        else:
            pearson_loss = pearson(target[:-i, 0, :2], output[i:, 0, :2])
            # pearson_loss = 1 - scipy.stats.pearsonr(target[:-i, 0, 0].detach().numpy(), output[i:, 0, 0].detach().numpy())[0]
            mse_loss = loss_functions["mse"](target[:-i, 0, :2], output[i:, 0, :2])
        print(f"Pearson loss: {pearson_loss}, MSE loss: {mse_loss}")

    # Print MSE loss
    mse_loss = loss_functions["mse"](target, output)
    print(f"MSE loss: {mse_loss}")
    print(f"MSE Pearson loss {mse_pearson(target, output)}")


    # Print average spike rate per timestep per neuron 
    print(f"Average spike rate per neuron: {torch.nansum(spikes).cpu().detach().numpy() / (spikes.size()[0] * spikes.size()[1])}")


    # plot the spikes for 1 second as little lines
    plt.figure(figsize=(20, 3))
    for i in range(spikes.size()[1]):
        spikes_t = spikes[:, i].cpu().detach().clone().numpy()
        spikes_t[spikes_t == 0] = np.nan
        plt.plot(spikes_t + i + 0.1, '.', color="dimgrey", markersize=4)
    plt.ylim(19.5, 70.5)
    plt.yticks([])
    plt.xlim(3300.5, 3600.5)
    plt.xticks([])
    plt.axis('off')
    # save as vector
    plt.savefig("spikes.svg", format="svg")
    


    np.savetxt("input.csv", data[:, 0, :].cpu().detach().numpy(), delimiter=",", fmt='%1.5f')
    np.savetxt("output.csv", output[:, 0, :].cpu().detach().numpy(), delimiter=",", fmt='%1.5f')
    
    
    plt.show()