import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming df is your DataFrame and it has columns 'SNN', 'roll', 'pitch', 'yawrate'
df = pd.read_csv('data/crazyflie_v6_snn/Train/log00.csv')  


plt.rc('font', family='serif')
fig, axs = plt.subplots(2, 1, figsize=[8, 4.5], sharex=True)
# rc_fonts = {
#     "font.family": "serif",
#     # "font.serif": "libertine",
#     # "font.size": 20,
#     # 'figure.figsize': (5, 3),
#     "text.usetex": True,
#     # # 'text.latex.preview': True,
#     # "text.latex.preamble": r"usepackage{libertine}",
# }
font_serif = {"family": "serif"}
# plt.rcParams.update(rc_fonts)

axs[0].set_facecolor("#EBEBEB")
axs[0].grid()

# Create x axis
t = np.arange(0, len(df['snn_control.torque_pitch'][38000:40000])) / 500
# Plot for pitch
axs[0].plot(t, df['snn_control.torque_pitch'][38000:40000], label='SNN', linewidth=2)
pid_pitch = df['pid_rate.pitch_output'] - df['pid_rate.pitch_outI']
axs[0].plot(t, pid_pitch[38000:40000], label='Crazyflie', linewidth=2)
axs[0].set_title('SNN vs PID torque commands', fontsize=14)
axs[0].set_ylim([-10000,10000])
axs[0].set_ylabel('pitch', fontsize=14)
axs[0].legend(fontsize=12, fancybox=True, shadow=True)
axs[0].tick_params(labelsize="x-small", direction="in")
for tick in axs[0].get_xticklabels():
    # tick.set_fontname(font_serif["family"])
    tick.set_visible(False)
for tick in axs[0].get_yticklabels():
    tick.set_fontname(font_serif["family"])


# Plot for roll
axs[1].set_facecolor("#EBEBEB")
axs[1].plot(t, df['snn_control.torque_roll'][38000:40000], label='SNN', linewidth=2)
pid_roll = df['pid_rate.roll_output'] - df['pid_rate.roll_outI']
axs[1].plot(t, pid_roll[38000:40000], label='Crazyflie', linewidth=2)
# plt.title('SNN vs pid roll torque commands', fontsize=16)
axs[1].set_ylim([-10000,10000])
axs[1].set_xlabel('time [s]', fontsize=14)
axs[1].set_ylabel('roll', fontsize=14)
axs[1].legend(fontsize=12, fancybox=True, shadow=True)
axs[1].grid()
axs[1].tick_params(labelsize="x-small", direction="in")
for tick in axs[1].get_xticklabels():
    tick.set_fontname(font_serif["family"])
for tick in axs[1].get_yticklabels():
    tick.set_fontname(font_serif["family"])
plt.savefig('snn_vs_pid_torque_commands.svg', dpi=300)
plt.show()


# Plot attitude command and response
fig, axs = plt.subplots(2, 1, figsize=[8, 4.5], sharex=True)
axs[0].set_facecolor("#EBEBEB")
axs[0].grid()
# Create x axis
t = np.arange(0, len(df['controller.pitch'][38000:40000])) / 500
# Plot for pitch
axs[0].plot(t, df['controller.pitch'][38000:40000], label='Controller', linewidth=2)
axs[0].plot(t, df['stateEstimate.pitch'][38000:40000], label='Estimate', linewidth=2)
axs[0].set_title('SNN vs PID attitude commands', fontsize=14)
# axs[0].set_ylim([-10000,10000])
axs[0].set_ylabel('pitch', fontsize=14)
axs[0].legend(fontsize=12, fancybox=True, shadow=True)
axs[0].tick_params(labelsize="x-small", direction="in")
for tick in axs[0].get_xticklabels():
    # tick.set_fontname(font_serif["family"])
    tick.set_visible(False)
for tick in axs[0].get_yticklabels():
    tick.set_fontname(font_serif["family"])

# Plot for roll
axs[1].set_facecolor("#EBEBEB")
axs[1].plot(t, df['controller.roll'][38000:40000], label='Controller', linewidth=2)
axs[1].plot(t, df['stateEstimate.roll'][38000:40000], label='Estimate', linewidth=2)
# plt.title('SNN vs pid roll torque commands', fontsize=16)
# axs[1].set_ylim([-10000,10000])
axs[1].set_xlabel('time [s]', fontsize=14)
axs[1].set_ylabel('roll', fontsize=14)
axs[1].legend(fontsize=12, fancybox=True, shadow=True)
axs[1].grid()
axs[1].tick_params(labelsize="x-small", direction="in")
for tick in axs[1].get_xticklabels():
    tick.set_fontname(font_serif["family"])
for tick in axs[1].get_yticklabels():
    tick.set_fontname(font_serif["family"])
# plt.savefig('snn_vs_pid_attitude_commands.svg', dpi=300)

plt.show()


# Plot attitude command and response
fig, axs = plt.subplots(2, 1, figsize=[8, 4.5], sharex=True)
axs[0].set_facecolor("#EBEBEB")
axs[0].grid()
# Create x axis
t = np.arange(0, len(df['controller.pitch'][38000:40000])) / 500
# Plot for pitch
axs[0].plot(t, df['controller.pitch'][38000:40000], label='Controller', linewidth=2)
axs[0].plot(t, df['stateEstimate.pitch'][38000:40000], label='Estimate', linewidth=2)
axs[0].set_title('SNN vs PID attitude commands', fontsize=14)
# axs[0].set_ylim([-10000,10000])
axs[0].set_ylabel('pitch', fontsize=14)
axs[0].legend(fontsize=12, fancybox=True, shadow=True)
axs[0].tick_params(labelsize="x-small", direction="in")
for tick in axs[0].get_xticklabels():
    # tick.set_fontname(font_serif["family"])
    tick.set_visible(False)
for tick in axs[0].get_yticklabels():
    tick.set_fontname(font_serif["family"])

# Plot for roll
axs[1].set_facecolor("#EBEBEB")
axs[1].plot(t, df['controller.roll'][38000:40000], label='Controller', linewidth=2)
axs[1].plot(t, df['stateEstimate.roll'][38000:40000], label='Estimate', linewidth=2)
# plt.title('SNN vs pid roll torque commands', fontsize=16)
# axs[1].set_ylim([-10000,10000])
axs[1].set_xlabel('time [s]', fontsize=14)
axs[1].set_ylabel('roll', fontsize=14)
axs[1].legend(fontsize=12, fancybox=True, shadow=True)
axs[1].grid()
axs[1].tick_params(labelsize="x-small", direction="in")
for tick in axs[1].get_xticklabels():
    tick.set_fontname(font_serif["family"])
for tick in axs[1].get_yticklabels():
    tick.set_fontname(font_serif["family"])
# plt.savefig('snn_vs_pid_attitude_commands.svg', dpi=300)

plt.show()
