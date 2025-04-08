import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import yaml

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

logs = {
    "Fixed": [
        "scarlet-plasma-334",
        "hardy-frog-377",
        "vocal-sea-378",
        "desert-eon-379",
        "rural-bee-380",
        "decent-breeze-381"
    ],
    "Free": [
        "autumn-donkey-335",
        "ethereal-moon-336",
        "fresh-fog-344",
        "cosmic-snow-360",
        "copper-frost-368",
        "faithful-snowflake-375"
    ],
}

n_values = 25

# datasets = "fitness_hist"
datasets = "validation_hist"

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.2))
    for key, value in logs.items():
        minimum_values = None
        maximum_values = None
        avg_values = None
        line_colr = "tab:blue" if key == "Fixed" else "tab:purple"
        for log in value:
            with open(f"runs/{log}/results.txt", "r") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            if minimum_values == None:
                minimum_values = data[datasets][:n_values]
                maximum_values = data[datasets][:n_values]
                avg_values = data[datasets][:n_values]
            else:
                minimum_values = [min(minimum_values[i], data[datasets][i]) for i in range(n_values)]
                maximum_values = [max(maximum_values[i], data[datasets][i]) for i in range(n_values)]
                avg_values = [avg_values[i] + data[datasets][i] for i in range(n_values)]

            # ax.plot(data["validation_hist"], color=line_colr, label=log, alpha=0.2)
        avg_values = [x / len(value) for x in avg_values]

        ax.plot(range(1, len(avg_values) + 1), avg_values, label=key, color=line_colr)
        ax.fill_between(range(1, len(minimum_values)+1), minimum_values, maximum_values, color=line_colr, alpha=0.2)
    ax.set_xlabel("Epoch", ha="right", x=1)
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_xlim([1, n_values])
    ax.set_ylim([0.0001, 1])
    ax.set_title("Loss curves for fixed and free parameters")
    ax.grid(True)
    ax.tick_params(which='both', direction='in')
    ax.set_facecolor("#EBEBEB")

    # ax.legend(loc="lower center", fancybox=True, shadow=True, ncol=2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("figures/loss_curves_integration.png", bbox_inches="tight", pad_inches=0.04, dpi=600)
    plt.show()