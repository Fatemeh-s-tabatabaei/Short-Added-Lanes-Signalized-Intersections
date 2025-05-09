import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Function to run the event-based simulation ===
def event_based_simulation(simulations=1000, v=1000, alpha=1/3, N=7, red=36, seed=None):
    if seed is not None:
        np.random.seed(seed)

    results = []
    for _ in range(simulations):
        arrival_times = []
        current_time = 0
        arrival_rate_per_sec = v / 3600

        while current_time < red:
            inter_arrival = np.random.exponential(1 / arrival_rate_per_sec)
            current_time += inter_arrival
            if current_time < red:
                arrival_times.append(current_time)

        short_lane_queue = 0
        through_lane_queue = 0
        blockage_occurred = False
        blocked_lane = None
        blockage_time = red
        queue_at_blocked_lane = 0

        for t in arrival_times:
            if np.random.rand() < alpha:
                short_lane_queue += 1
                if short_lane_queue == N + 1:
                    blockage_occurred = True
                    blockage_time = t
                    blocked_lane = 'through'
                    queue_at_blocked_lane = through_lane_queue
                    break
            else:
                through_lane_queue += 1
                if through_lane_queue == N + 1:
                    blockage_occurred = True
                    blockage_time = t
                    blocked_lane = 'short'
                    queue_at_blocked_lane = short_lane_queue
                    break

        if blockage_occurred:
            results.append({
                "BlockedLane": blocked_lane,
                "QueueInBlockedLane": queue_at_blocked_lane
            })

    return results

# === Parameters ===
alphas = np.arange(0.01, 1.0, 0.02)
Ns = range(1, 11)
simulations_per_config = 50
v = 900
red = 36

# === Run simulations and store mean queue ratios ===
all_means = []

for N in Ns:
    for alpha in alphas:
        ratio = round(alpha / (1 - alpha), 2)
        results = event_based_simulation(simulations=simulations_per_config, v=v, alpha=alpha, N=N, red=red, seed=42)
        if results:
            queue_ratios = [r["QueueInBlockedLane"] / N for r in results if r["QueueInBlockedLane"] is not None]
            if queue_ratios:
                mean_q = np.mean(queue_ratios)
                all_means.append({
                    "AlphaRatio": ratio,
                    "N": N,
                    "MeanQueueRatio": mean_q
                })

df_means = pd.DataFrame(all_means)


plt.figure(figsize=(14, 7))
palette = sns.color_palette("tab10", len(Ns))  # limited to distinguishable colors

num_N = len(Ns)
fig, axes = plt.subplots(nrows=num_N, ncols=1, figsize=(10, 3 * num_N), sharex=True)

for i, N in enumerate(Ns):
    ax = axes[i]
    df_subset = df_means[df_means["N"] == N]
    if not df_subset.empty:
        ax.plot(df_subset["AlphaRatio"], df_subset["MeanQueueRatio"],
                label=f"N = {N}", color="tab:blue", linewidth=2, alpha=0.7)

    # Reference curves
    x_vals_1 = np.linspace(0.01, 1, 100)
    ax.plot(x_vals_1, x_vals_1, color="black", linestyle="--", label="y = x")

    x_vals_2 = np.linspace(1, 100, 200)
    ax.plot(x_vals_2, 1 / x_vals_2, color="black", linestyle="--", label="y = 1/x")

    ax.set_title(f"Mean Queue Ratio for N = {N}")
    ax.set_ylabel("Queue/N")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right")

axes[-1].set_xlabel(r"$\alpha / (1 - \alpha)$")
plt.tight_layout()
plt.savefig("separate_mean_lines_by_N.png", dpi=300)
plt.show()