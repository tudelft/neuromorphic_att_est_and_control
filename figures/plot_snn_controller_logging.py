import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.signal
import torch

avg_pearson_list = []
avg_mse_list = []
# datasets = ["att_dist/lilac_wood", "att_dist/crimson_dew"]
# datasets = ["crazyflie_v6/snn/log03", "att_dist/crimson_dew"]
datasets = ["27_08/att_dist/lilac_wood/log02"] # TODO make parameter as argument
title_fontsize = 10
legend_loc = "upper right"
for dataset in datasets:
    data = pd.read_csv(f"data/logs/{dataset}.csv")
    fig, axes = plt.subplots(nrows=6, ncols=1, sharex=True, figsize=[6, 9])

    axs_id = 0
    data["stateEstimate.roll"].plot(ax=axes[axs_id], label="roll")
    data["controller.roll"].plot(ax=axes[axs_id], label="roll command")
    axes[axs_id].axhline(0, 0, len(data["stateEstimate.roll"]), color='k')
    axes[axs_id].legend(loc=legend_loc)
    axes[axs_id].set_title("roll command", fontsize=title_fontsize)
    axes[axs_id].set_ylim([-12, 12])
    axes[axs_id].set_xlabel("timestep (0.002ms)")
    axs_id += 1

    data["stateEstimate.pitch"].plot(ax=axes[axs_id], label="pitch")
    data["controller.pitch"].plot(ax=axes[axs_id], label="pitch command")
    axes[axs_id].axhline(0, 0, len(data["stateEstimate.pitch"]), color='k')
    # axes[axs_id].set_ylim([-35000, 35000])
    axes[axs_id].legend(loc=legend_loc)
    axes[axs_id].set_title("pitch command", fontsize=title_fontsize)
    axes[axs_id].set_ylim([-12, 12])
    axes[axs_id].set_xlabel("timestep (0.002ms)")
    axs_id += 1

    # (data["pid_attitude.pitch_output"] - data["pid_attitude.pitch_outI"]).plot(ax=axes[axs_id], label="pid")
    data["pid_attitude.pitch_output"].plot(ax=axes[axs_id], label="pid")
    # data["snn_control.pitch_input"].plot(ax=axes[axs_id], label="snn input")
    (-data["gyro.y"]).plot(ax=axes[axs_id], label="gyroy")
    axes[axs_id].axhline(0, 0, len(data["pid_attitude.pitch_output"]), color='k')
    axes[axs_id].legend(loc=legend_loc)
    axes[axs_id].set_ylim([-100, 100])
    axes[axs_id].set_title("Pitch rate", fontsize=title_fontsize)
    axs_id += 1

    # (data["pid_attitude.roll_output"] - data["pid_attitude.roll_outI"]).plot(ax=axes[axs_id], label="pid")
    data["pid_attitude.roll_output"].plot(ax=axes[axs_id], label="pid")
    # data["snn_control.roll_input"].plot(ax=axes[axs_id], label="snn input")
    data["gyro.x"].plot(ax=axes[axs_id], label="gyrox")
    axes[axs_id].axhline(0, 0, len(data["pid_attitude.roll_output"]), color='k')
    axes[axs_id].legend(loc=legend_loc)
    axes[axs_id].set_ylim([-100, 100])
    axes[axs_id].set_title("Roll rate", fontsize=title_fontsize)
    axes[axs_id].set_xlabel("timestep (0.002ms)")

    axs_id += 1
    axes[axs_id].set_title("Roll torque command", fontsize=title_fontsize)
    (data["pid_rate.pitch_output"] - data["pid_rate.pitch_outI"]).plot(ax=axes[axs_id], label="pid pitch")
    data["snn_control.torque_pitch"].shift(0).fillna(0).plot(ax=axes[axs_id], label="snn pitch")
    axes[axs_id].set_ylim([-20000, 20000])
    axes[axs_id].legend(loc=legend_loc)

    axs_id += 1
    axes[axs_id].set_title("Pitch torque command", fontsize=title_fontsize)
    (data["pid_rate.roll_output"] - data["pid_rate.roll_outI"]).plot(ax=axes[axs_id], label="pid roll")
    # (data["pid_rate.roll_output"]).plot(ax=axes[axs_id], label="pid roll")
    data["snn_control.torque_roll"].shift(0).fillna(0).plot(ax=axes[axs_id], label="snn roll")
    axes[axs_id].set_ylim([-20000, 20000])
    axes[axs_id].legend(loc=legend_loc)

    fig.tight_layout()

    pitch_list = []
    roll_list = []
    avg_pearson = []
    avg_mse = []
    pearson_first = None
    for i in range(0, -13, -1):
        pitch = scipy.stats.pearsonr((data["pid_rate.pitch_output"] - data["pid_rate.pitch_outI"]), data["snn_control.torque_pitch"].shift(i).fillna(0))
        roll = scipy.stats.pearsonr((data["pid_rate.roll_output"] - data["pid_rate.roll_outI"]), data["snn_control.torque_roll"].shift(i).fillna(0))
        
        pitch_mse = torch.nn.functional.mse_loss(torch.tensor((data["pid_rate.pitch_output"] - data["pid_rate.pitch_outI"]).values), torch.tensor(data["snn_control.torque_pitch"].shift(i).fillna(0).values))
        roll_mse = torch.nn.functional.mse_loss(torch.tensor((data["pid_rate.roll_output"] - data["pid_rate.roll_outI"]).values), torch.tensor(data["snn_control.torque_roll"].shift(i).fillna(0).values))
        
        avg_prs = (pitch[0] + roll[0]) / 2
        pitch_list.append(pitch[0])
        roll_list.append(roll[0])

        avg_mse.append((pitch_mse + roll_mse) / 2)

        avg_pearson.append(np.array(avg_prs))
    print(f"[{dataset}] Avg max: {np.argmax(avg_pearson)}, pitch max: {np.argmax(pitch_list)}, roll max: {np.argmax(roll_list)}")
    avg_pearson_list.append(avg_pearson)
    avg_mse_list.append(avg_mse)




fig = plt.figure(figsize=[3,3])
for dataset, name in zip(avg_pearson_list, ["trained delayed", "trained non-delayed"]):
    dataset = dataset #- np.max(dataset)
    plt.plot(dataset, label=name)
plt.ylabel("shifted pearson correlation")
plt.xlabel("shift [timesteps]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()