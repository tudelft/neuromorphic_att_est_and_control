import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.stats
import os

# Set up fonts and LaTeX preamble for consistent use of Libertine
rc_fonts = {
    "text.usetex": True,
    "font.family": "serif",  # Default to serif
    "text.latex.preamble": r"\usepackage{libertine}"  # Use Libertine font
}
plt.rcParams.update(rc_fonts)

fig, axs = plt.subplots(2, 1, figsize=[4, 3.5])

# Reduce space between plots
plt.subplots_adjust(hspace=0)

axs[0].set_facecolor("#EBEBEB")
axs[0].grid()
alpha = 1.0

folders = ["data/logs/27_08/att_dist/crimson_dew"]
colors = ["#2c7bb6", "", "", "#d7191c"]
color_idx = 0

# Loop through folders and calculate correlations
for folder in folders:
    avg_pearson_list = []
    for dataset in os.listdir(folder):
        if not dataset.endswith(".csv"):
            continue
        data = pd.read_csv(f"{folder}/{dataset}")
        pitch_list = []
        roll_list = []
        avg_pearson = []
        for i in range(0, -13, -1):
            pitch = scipy.stats.pearsonr(
                (data["pid_rate.pitch_output"] - data["pid_rate.pitch_outI"]),
                data["snn_control.torque_pitch"].shift(i).fillna(0)
            )
            roll = scipy.stats.pearsonr(
                (data["pid_rate.roll_output"] - data["pid_rate.roll_outI"]),
                data["snn_control.torque_roll"].shift(i).fillna(0)
            )
            avg_prs = (pitch[0] + roll[0]) / 2
            pitch_list.append(pitch[0])
            roll_list.append(roll[0])
            avg_pearson.append(np.array(avg_prs))
        avg_pearson_list.append(avg_pearson)

    axs[0].plot(np.mean(avg_pearson_list, axis=0), label=f"{folder.split('/')[-1]}", color=colors[0])
    color_idx += 1

axs[0].set_ylim([0.69, 0.81])
axs[0].legend(["SNN delay"], fancybox=True, shadow=True, loc="upper right")
axs[0].tick_params(labelsize="small", direction="in")
axs[0].set_ylabel("Pearson Correlation")
axs[0].set_xlabel("Timeshift $d$ [timesteps]")

inset_ax = axs[1]
inset_ax.plot((data["pid_rate.pitch_output"] - data["pid_rate.pitch_outI"]).rolling(5).mean(), color=colors[3]) 
inset_ax.plot(data["snn_control.torque_pitch"].rolling(5).mean(), color=colors[0], alpha=0.8)
inset_ax.plot(data["snn_control.torque_pitch"].shift(-6).fillna(0).rolling(5).mean(), color=colors[0], alpha=0.8, linestyle="dotted")
inset_ax.plot(data["snn_control.torque_pitch"].shift(-12).fillna(0).rolling(5).mean(), color=colors[0], alpha=0.8, linestyle="dashed")
inset_ax.set_xlim([1100, 1250])
inset_ax.set_ylim([-8000, 8000])

axs[1].set_facecolor("#EBEBEB")
axs[1].grid()
axs[1].tick_params(labelsize="small", direction="in")
axs[1].set_ylabel("Motor Command")
axs[1].set_xlabel("Time [ms]")

# Add legend
axs[1].legend(["ref", "$d=0$", "$d=6$", "$d=12$"], fancybox=True, shadow=True, loc="lower right", ncols=4, columnspacing=0.6)

# Set y-axis to scientific notation
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
axs[1].yaxis.set_major_formatter(formatter)

# Move scientific notation text to the top left corner and decrease font size
offset_text = axs[1].yaxis.get_offset_text()
offset_text.set_position((0, 1))
offset_text.set_verticalalignment('bottom')
offset_text.set_horizontalalignment('left')
offset_text.set_fontsize(8)  # Adjust the font size as desired

fig.tight_layout()
plt.savefig("figures/delay.svg")
plt.show()
