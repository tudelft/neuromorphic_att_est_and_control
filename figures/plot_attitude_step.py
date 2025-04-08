import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import scipy


# Set up fonts and LaTeX preamble for consistent use of Libertine
rc_fonts = {
    "text.usetex": True,
    "font.family": "serif",  # Default to serif
    "text.latex.preamble": r"\usepackage{libertine}"  # Use Libertine font
}
plt.rcParams.update(rc_fonts)
plt.rcParams["svg.fonttype"] = "path"  # Embed fonts in eps



folders = ["data/logs/05_09/att_dist/lilac_wood", 
           "data/logs/05_09/att_dist/crimson_dew", 
           "data/logs/05_09/att_dist/desert_snow", 
           "data/logs/05_09/att_dist/sandy_hill", 
           "data/logs/22_01/att_dist/pid"]

colors = ["#2c7bb6", "#ff7f00", "#33a02c", "black", "#d7191c"]

names = ["A) SNN (augmented \& time-shifted)", 
         "B) SNN (augmented)", 
         "C) SNN (time-shifted)", 
         "D) SNN (baseline)",
         "E) PID (reference)"]
color_idx = 0
title_fontsize = 13

# Define the grid layout
fig = plt.figure(figsize=[12, 5])
# fig_fft = plt.figure(figsize=[5, 5])
# ax_fft = fig_fft.add_subplot(111)

gs = GridSpec(nrows=3, ncols=2, height_ratios=[2, 1, 1])

# First subplot: spans all columns in the first row
ax1 = fig.add_subplot(gs[0, :])

# Second row: two subplots
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# Third subplot: two subplots
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])

axes = [ax1, ax2, ax3, ax4, ax5]

# Custom formatter for y-axis ticks to add degree symbol
def degree_formatter(x, pos):
    return f'{int(x)}°'

# Set degree formatter for y-axis ticks in all subplots
for ax in axes:
    ax.yaxis.set_major_formatter(FuncFormatter(degree_formatter))

alpha = 1.0

axs_id = 0
t_roll_command = None
for folder in folders:
    avg_response_roll = []
    avg_response_pitch = []
    mse_list = []
    std_list = []
    avg_fft_f = []
    avg_fft_Pxx = []

    for dataset in os.listdir(folder):
        # only use .csv files
        if not dataset.endswith(".csv"):
            continue
        data = pd.read_csv(f"{folder}/{dataset}")

        this_t_roll_command = data["controller.roll"].idxmin()
        if t_roll_command is None:
            # find timestep of first roll command below 0
            t_roll_command = data["controller.roll"].idxmin()
        
        # shift entire dataset to have the first roll command at the same time
        data.index = data.index - this_t_roll_command
        data.index = data.index + t_roll_command

        # only look at the first 4000 steps
        data = data[data.index > 0]
        data = data[data.index < 4000]

        avg_response_roll.append(data["stateEstimate.roll"].values)
        # avg_response_pitch.append(data["stateEstimate.pitch"].values)

    # calculate average roll estimate
    avg_response_roll = np.array(avg_response_roll).mean(axis=0)

    for dataset in os.listdir(folder):
        # only use .csv files
        if not dataset.endswith(".csv"):
            continue
        data = pd.read_csv(f"{folder}/{dataset}")

        this_t_roll_command = data["controller.roll"].idxmin()
        if t_roll_command is None:
            # find timestep of first roll command below 0
            t_roll_command = data["controller.roll"].idxmin()
        
        # shift entire dataset to have the first roll command at the same time
        data.index = data.index - this_t_roll_command
        data.index = data.index + t_roll_command

        # only look at the first 4000 steps
        data = data[data.index > 0]
        data = data[data.index < 4000]

        # plot roll and command data
        axes[axs_id].plot(data["stateEstimate.roll"], label="roll", color=colors[color_idx], alpha=0.25)
        # axes[axs_id].plot(avg_response_roll, label="roll", color=colors[color_idx], alpha=0.25)
        axes[axs_id].plot(data["controller.roll"], label="roll command", color="grey")

        # avg_response_roll.append(data["stateEstimate.roll"].values)
        # avg_response_pitch.append(data["stateEstimate.pitch"].values)

        # calculate rmse
        rmse = ((data["stateEstimate.roll"] - data["controller.roll"])**2).mean() ** 0.5
        # calculate mae
        mae = (data["stateEstimate.roll"] - data["controller.roll"]).abs().mean()
        mse_list.append(rmse)
        
        # calculate std of state vs average 
        std_list.append(np.std(data["stateEstimate.roll"] - avg_response_roll))


    # print avg mse for the folder
    print(f"{folder.split('/')[-1]} avg mse: {sum(mse_list) / len(mse_list)}")

    # print avg std for the folder
    print(f"{folder.split('/')[-1]} avg std: {sum(std_list) / len(std_list)}")

    axes[axs_id].set_facecolor("#EBEBEB")
    # Add horizontal line at 0 for reference
    axes[axs_id].axhline(0, 0, len(data["stateEstimate.roll"]), color='k')

    # Set title and limits
    axes[axs_id].text(0.03, 0.93, f"{names[color_idx]}", transform=axes[axs_id].transAxes, fontsize=title_fontsize, verticalalignment='top', horizontalalignment='left')  # Serif font for title
    # axes[axs_id].set_title(f"{names[color_idx]}", fontsize=title_fontsize, loc="left")  # Serif font for title
    axes[axs_id].set_ylim([-18, 18])
    axes[axs_id].set_xlim([-40, data.index.max() + 50])
    

    # Convert x-axis from milliseconds to seconds
    # xticks = axes[axs_id].get_xticks()  # Get current ticks in ms
    # if axs_id == 0:
    #     xticks = np.arange(0, data.index.max()+100, 500)
    # else:
    #     xticks = np.arange(0, data.index.max()+100, 1000)

    # axes[axs_id].set_xticks(xticks)  # Explicitly set the ticks

    # if (axs_id == 0 or axs_id == 3 or axs_id == 4):
    #     axes[axs_id].set_xticklabels([f"{x/500:.0f}" for x in xticks])  # Set tick labels in seconds
    # else:
    #     axes[axs_id].set_xticklabels([f"" for x in xticks])

    # if (axs_id == 2 or axs_id == 4):
    #     axes[axs_id].set_yticklabels([f"" for x in axes[axs_id].get_yticks()]) 

    # Set grid and make ticks go inside the graph
    axes[axs_id].grid(True)
    axes[axs_id].tick_params(direction='in')  # Ticks inside

    if axs_id > 2:
        axes[axs_id].set_xlabel("Time [s]")  # X label in seconds 

    axs_id += 1
    color_idx += 1

# Add a common y-label on the left side of the figure
fig.text(0.06, 0.5, 'Roll Angle [deg]', va='center', rotation='vertical', fontsize=12)

# Create a custom line style with four dashes of different colors for "Measured"
legend_measured = [
    Line2D([0], [0], color=colors[0], lw=2),
    Line2D([0], [0], color=colors[1], lw=2),
    Line2D([0], [0], color=colors[2], lw=2),
    Line2D([0], [0], color=colors[3], lw=2),
    Line2D([0], [0], color=colors[4], lw=2),
    Line2D([0], [0], color='white', lw=2, alpha=0.0),
    Line2D([0], [0], color='grey', lw=2, label='Setpoint')
]

# Add the legend with the custom "Measured" and "Setpoint" lines
fig.legend(handles=legend_measured, 
            labels=["", "", "", "", "Measured", "", "Setpoint"], 
            loc='lower center',
            bbox_to_anchor=(0.55, 0.04),
            ncol=7, 
            fontsize=10, 
            handletextpad=0.5, 
            handlelength=1.0, 
            columnspacing=0.0,
            fancybox=True,
            shadow=True)

# Adjust layout to make room for the legend
fig.tight_layout(rect=[0.06, 0.04, 1.0, 1.0])
# fig.tight_layout()

# Save the figure
plt.savefig("figures/roll_commands.svg", format='svg')

plt.show()
