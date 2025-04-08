import pandas as pd
import matplotlib.pyplot as plt
import os

# LaTeX and font settings for 'serif'
rc_fonts = {
    "font.family": "serif",
    "text.usetex": True,
    "axes.labelsize": 12,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{libertine}')
plt.rcParams.update(rc_fonts)

def plot_datasets(folder, axs_id, axes, color1, color2, title):
    datasets = os.listdir(folder)
    for dataset in datasets:
        if not dataset.endswith(".csv"):
            continue
        data = pd.read_csv(f"{folder}/{dataset}")
        data["timestamp"] = data["timestamp"] - data["timestamp"].iloc[0]
        data = data[data["timestamp"] > 5000]

        data["timestamp"] = data["timestamp"] / 1000

        # # Plot time vs x  
        axes[axs_id].plot(data["timestamp"], data["ctrltarget.x"], color=color2, linewidth=2, label="Target")
        axes[axs_id].plot(data["timestamp"], data["locSrv.x"], color=color1, alpha=0.5, linewidth=0.5, label="OptiTrack")

    axes[axs_id].set_xlim(6.5, 20.5)
    axes[axs_id].set_title(title, fontsize=14)
    axes[axs_id].legend(["Target", "Measured"], fontsize=10, fancybox=True, shadow=True, loc="upper right")
    axes[axs_id].grid(True)
    axes[axs_id].set_facecolor("#EBEBEB")
    axes[axs_id].tick_params(direction='in')


# Create a figure and axes
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(5,4))

# Colorblind-friendly colors
color1 = "#2c7bb6"  # Blue
color2 = "grey"  # Orange
color3 = "#d7191c"

# colors = ["#2c7bb6", "#ff7f00", "#33a02c", "#d7191c"]

# SNN dataset
axs_id = 0
snn_folder = "data/logs/step_20_08/snn"
plot_datasets(snn_folder, axs_id, axes, color1, color2, "SNN Fusion and Control")

# PID dataset
axs_id += 1
pid_folder = "data/logs/step_20_08/pid"
plot_datasets(pid_folder, axs_id, axes, color3, color2, "Complementary filter and Cascaded PID")

fig.text(0.01, 0.5, "$x$ position [m]", va='center', rotation='vertical', fontsize=13, family='serif')
axes[axs_id].set_xlabel("Time [s]", fontsize=12)

# Adjust layout
plt.tight_layout()
# plt.show()
plt.savefig("figures/snn_pid_comparison.png", dpi=400)