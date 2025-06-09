import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_trajectories = 50
timesteps = 35
x = np.arange(timesteps)

# Generate Y trajectories with noise
trajectories = np.cumsum(np.random.randn(n_trajectories, timesteps) * 2 + 0.5, axis=1)

# Identify high mean Y-value trajectories (top 10%)
mean_y_values = trajectories.mean(axis=1)
threshold = np.percentile(mean_y_values, 90)
high_value_indices = np.where(mean_y_values > threshold)[0]

# Create plot
fig, ax = plt.subplots(figsize=(8, 5))

# Plot low-value trajectories
for i in range(n_trajectories):
    if i not in high_value_indices:
        ax.plot(x, trajectories[i], color='gray', alpha=0.50, linewidth=2)

# Plot high-value trajectories in red
for i, idx in enumerate(high_value_indices):
    label = "High mean Y-value scenarios" if i == 0 else None
    ax.plot(x, trajectories[idx], color='crimson', linewidth=2, label=label)
    # ax.plot(x, trajectories[idx], color='gray', alpha=0.50, linewidth=2, label=label)

# Customize axes: remove grid, set limits
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Thicken X and Y axes
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(3)

# Arrows for axes
x_max = timesteps
y_max = np.max(trajectories) * 1.05
ax.annotate('', xy=(x_max, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=3, color='black'))
ax.annotate('', xy=(0, y_max), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=3, color='black'))

# Labels and legend
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Y-value', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("scenario_fig", dpi=600)
plt.show()
