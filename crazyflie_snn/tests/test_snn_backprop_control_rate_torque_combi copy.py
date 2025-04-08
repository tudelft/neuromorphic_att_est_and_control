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

from ultrasonic_snn.dataloader.dataloader import load_datasets
from ultrasonic_snn.training.bp_snn import fit
from ultrasonic_snn.neural_models.models import models
from spiking.core.torch.model import get_model


# run_name = "snn_backprop27092023_112523"
# run_name = "snn_backprop28092023_162128" # only roll/pitch as inputs
# run_name = "snn_backprop03102023_150033" # px4 control data
# run_name = "snn_backprop29112023_151454" # new config 
# torque_run_name = "snn_backprop04122023_215552" # new config with torque commands
# torque_run_name = "snn_backprop06122023_101806" # happy night
# torque_run_name = "snn_backprop07122023_231736" # trim forest
# torque_run_name = "super-plant-138"
torque_run_name = "warm-armadillo-179"

# rate_run_name = "snn_backprop06112023_204504" # wordly_grass
# rate_run_name = "snn_backprop05122023_151604" # confused surf
# rate_run_name = "snn_backprop08122023_224225" # decent snowball
rate_run_name = "abundant-moon-184"
# rate_run_name = "likely-sponge-141"
# rate_run_name = "elated-bee-147"
# rate_run_name = "laced-jazz-148"

def load_network_and_eval(rate_run_name, torque_run_name):

    # Config from yaml file
    with open(f"runs/{rate_run_name}/config.txt", "r") as f:
        rate_config = yaml.load(f, Loader=yaml.FullLoader)

    # Config from yaml file
    with open(f"runs/{torque_run_name}/config.txt", "r") as f:
        torque_config = yaml.load(f, Loader=yaml.FullLoader)

    rate_config["data"]["seq_length"] = 2500
    # rate_config["data"]["dir"] = "data/crazyflie_v4_ratesnn"
    # rate_config["data"]["remove_integral"] = False

    rate_config["data"]["target_columns"] = ["pid_attitude.roll_output",
                                             "pid_attitude.pitch_output",
                                             "gyro.x",
                                             "gyro.y",
                                             "pid_rate.roll_output",
                                             "pid_rate.pitch_output"]
    rate_config["data"]["target_scaling"] = [0.05, 0.05, 0.03, 0.03, 0.00005, 0.00005]
    # Load training and validation dataloaders
    generator = torch.Generator(device=rate_config["device"])
    generator.manual_seed(0)
    rate_config["data"]["remove_initial_value"] = False
    rate_config["data"]["target_integral"] = False
    train_loader, val_loader, test_loader = load_datasets(rate_config, generator=generator)

    # Build the spiking model
    x, _ = next(iter(val_loader))
    rate_model = get_model(models[rate_config["net"]["type"]], rate_config["net"]["params"], data=x[:, 0], device=rate_config["device"])

    # Load pretrained model
    rate_model.load_state_dict(torch.load(f"runs/{rate_run_name}/model.pt"))
    rate_model.eval()
    rate_model.reset()

    # Perform inference
    data, target = next(iter(test_loader))
    data = data.permute(1, 0, 2)
    rate_target = target.permute(1, 0, 2)
    rate_output = rate_model.forward_sequence(data)

    # add gyro data to rate output if rate output is just roll/pitch   
    # rate_output = torch.cat((rate_output, data[:, :, 0:2]), dim=2)

    # Perform inference
    torque_model = get_model(models[torque_config["net"]["type"]], torque_config["net"]["params"], data=rate_output[:, 0], device=torque_config["device"])
    torque_model.load_state_dict(torch.load(f"runs/{torque_run_name}/model.pt"))
    torque_model.eval()
    torque_model.reset()
    torque_output = torque_model.forward_sequence(rate_output)


    return torque_output, rate_output, rate_target, data



if __name__ == "__main__":
    torque_output, rate_output, rate_target, data = load_network_and_eval(rate_run_name, torque_run_name)

        # Plot results
    plt.subplot(6, 1, 1)
    plt.plot(rate_output[:, 0, 0].cpu().detach().numpy(), label="SNN Roll")
    plt.plot(rate_target[:, 0, 0].cpu().detach().numpy(), label="Target Roll")
    # plt.plot(data[:, 0, 2].cpu().detach().numpy()*0.5, label="Roll velocity")
    # plt.plot(data[:, 0, 0].cpu().detach().numpy()*5, label="Roll velocity")
    plt.legend()

    plt.subplot(6, 1, 2)
    plt.plot(torque_output[:, 0, 0].cpu().detach().numpy(), label="SNN Roll")
    plt.plot(rate_target[:, 0, 4].cpu().detach().numpy(), label="Target Roll Torque")
    # plt.plot(data[:, 0, 0].cpu().detach().numpy(), label="Roll")
    # plt.plot(data[:, 0, 2].cpu().detach().numpy(), label="Roll velocity")
    # plt.plot(data[:, 0, 4].cpu().detach().numpy(), label="Target roll")
    plt.ylim([-0.5, 0.5])
    plt.legend()

    plt.subplot(6, 1, 3)
    plt.plot(rate_output[:, 0, 1].cpu().detach().numpy(), label="SNN Pitch")
    plt.plot(rate_target[:, 0, 1].cpu().detach().numpy(), label="Target Pitch")
    # plt.plot(data[:, 0, 3].cpu().detach().numpy()*0.5, label="Pitch velocity")
    # plt.plot(-data[:, 0, 1].cpu().detach().numpy()*5, label="Pitch velocity")
    plt.legend()

    plt.subplot(6, 1, 4)
    plt.plot(torque_output[:, 0, 1].cpu().detach().numpy(), label="SNN Pitch")
    plt.plot(rate_target[:, 0, 5].cpu().detach().numpy(), label="Target Pitch Torque")
    # plt.plot(data[:, 0, 1].cpu().detach().numpy(), label="Pitch")
    # plt.plot(data[:, 0, 3].cpu().detach().numpy(), label="Pitch velocity")
    # plt.plot(data[:, 0, 5].cpu().detach().numpy(), label="Target pitch")
    plt.ylim([-0.5, 0.5])
    plt.legend()

    plt.subplot(6, 1, 5)
    plt.plot(rate_output[:, 0, 2].cpu().detach().numpy(), label="SNN gyro x")
    # plt.plot(data[:, 0, 0].cpu().detach().numpy(), label="gyro x")
    plt.plot(rate_target[:, 0, 2].cpu().detach().numpy(), label="gyro x")

    plt.subplot(6, 1, 6)
    plt.plot(rate_output[:, 0, 3].cpu().detach().numpy(), label="SNN gyro y")
    plt.plot(rate_target[:, 0, 3].cpu().detach().numpy(), label="gyro y")
    # plt.plot(data[:, 0, 1].cpu().detach().numpy(), label="gyro y")
    plt.show()


    # # Plot results
    # plt.subplot(4, 1, 1)
    # plt.plot(output[:, 0, 0].cpu().detach().numpy(), label="SNN Roll")
    # plt.plot(target[:, 0, 0].cpu().detach().numpy(), label="Target Roll")
    # # plt.plot(data[:, 0, 2].cpu().detach().numpy()*0.5, label="Roll velocity")
    # # plt.plot(data[:, 0, 0].cpu().detach().numpy()*5, label="Roll velocity")
    # plt.legend()

    # plt.subplot(4, 1, 2)
    # plt.plot(data[:, 0, 0].cpu().detach().numpy(), label="Roll")
    # plt.plot(data[:, 0, 2].cpu().detach().numpy(), label="Roll velocity")
    # # plt.plot(data[:, 0, 4].cpu().detach().numpy(), label="Target roll")
    # plt.legend()

    # plt.subplot(4, 1, 3)
    # plt.plot(output[:, 0, 1].cpu().detach().numpy(), label="SNN Pitch")
    # plt.plot(target[:, 0, 1].cpu().detach().numpy(), label="Target Pitch")
    # # plt.plot(data[:, 0, 3].cpu().detach().numpy()*0.5, label="Pitch velocity")
    # # plt.plot(-data[:, 0, 1].cpu().detach().numpy()*5, label="Pitch velocity")
    # plt.legend()

    # plt.subplot(4, 1, 4)
    # plt.plot(data[:, 0, 1].cpu().detach().numpy(), label="Pitch")
    # plt.plot(data[:, 0, 3].cpu().detach().numpy(), label="Pitch velocity")
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
    # plt.show()

    np.savetxt("input.csv", data[:, 0, :].cpu().detach().numpy(), delimiter=",", fmt='%1.5f')
    np.savetxt("output.csv", torque_output[:, 0, :].cpu().detach().numpy(), delimiter=",", fmt='%1.5f')