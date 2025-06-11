import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
import seaborn as sns

# Parameters
alphas = np.round(np.arange(0, 1.0, 0.02), 2)
ratios = np.round(alphas / (1 - alphas), 2)
valid_indices = ~np.isinf(ratios)
alphas = alphas[valid_indices]
ratios = ratios[valid_indices]
ratio_labels = [f"{r:.2f}" for r in ratios]

# Constants
N = 7
b = 6
v = 1800
red = 36
simulation_runs = 1000

# Reusable Simulation
def event_based_simulation(simulations=1000, v=1800, alpha=1/3, N=N, b=b, red=36, seed=None):
    if seed is not None:
        np.random.seed(seed)
    results = []
    arrival_rate_per_sec = v / 3600
    departure_time_per_vehicle = 2.0
    for _ in range(simulations):
        current_time = 0.0
        arrival_times = []
        while current_time < red:
            inter_arrival = np.random.exponential(1 / arrival_rate_per_sec)
            current_time += inter_arrival
            arrival_times.append(current_time)

        short_lane_queue = 0
        through_lane_queue = 0
        blockage_occurred = False
        blocked_lane = None
        queue_at_blocked_lane = None

        for t in arrival_times:
            if np.random.rand() < alpha:
                short_lane_queue += 1
                if short_lane_queue == N:
                    blockage_occurred = True
                    blocked_lane = 'through'
                    queue_at_blocked_lane = through_lane_queue
                    break
            else:
                through_lane_queue += 1
                if through_lane_queue == N:
                    blockage_occurred = True
                    blocked_lane = 'short'
                    queue_at_blocked_lane = short_lane_queue
                    break

        if blockage_occurred:
            results.append({
                "BlockedLane": blocked_lane,
                "QueueInBlockedLane": queue_at_blocked_lane,
                "BlockageOccurred": True
            })
            continue

        short_end = red + short_lane_queue * departure_time_per_vehicle
        through_end = red + through_lane_queue * departure_time_per_vehicle
        next_arrival = arrival_times[-1]
        while True:
            if np.random.rand() < alpha:
                if next_arrival > short_end: break
                short_lane_queue += 1
                if short_lane_queue == N:
                    results.append({
                        "BlockedLane": "through",
                        "QueueInBlockedLane": through_lane_queue,
                        "BlockageOccurred": True
                    })
                    break
                short_end += departure_time_per_vehicle
            else:
                if next_arrival > through_end: break
                through_lane_queue += 1
                if through_lane_queue == N:
                    results.append({
                        "BlockedLane": "short",
                        "QueueInBlockedLane": short_lane_queue,
                        "BlockageOccurred": True
                    })
                    break
                through_end += departure_time_per_vehicle
            inter_arrival = np.random.exponential(1 / arrival_rate_per_sec)
            next_arrival += inter_arrival
    return pd.DataFrame(results)

# Run simulation once for all alpha values
full_records = []

for alpha in alphas:
    ratio = round(alpha / (1 - alpha), 2)
    df_sim = event_based_simulation(alpha=alpha, N=N, b=b, simulations=simulation_runs, red=red, v=v, seed=42)
    for _, row in df_sim.iterrows():
        if row["BlockedLane"] == "short":
            full_records.append({
                "Alpha": alpha,
                "AlphaRatio": ratio,
                "BlockedLane": "Short",
                "QueueRatio": row["QueueInBlockedLane"] / N,
                "BlockageOccurred": row["BlockageOccurred"]
            })
        elif row["BlockedLane"] == "through":
            full_records.append({
                "Alpha": alpha,
                "AlphaRatio": ratio,
                "BlockedLane": "Through",
                "QueueRatio": row["QueueInBlockedLane"] / b,
                "BlockageOccurred": row["BlockageOccurred"]
            })

df_plot = pd.DataFrame(full_records)

# === Plot 1: Boxplot ===
plot_data = []
plot_colors = []
plot_pos = []
x_labels = []
base_pos = 0
offset = 0.15

for r in sorted(df_plot["AlphaRatio"].unique()):
    df_r = df_plot[df_plot["AlphaRatio"] == r]
    short_vals = df_r[df_r["BlockedLane"] == "Short"]["QueueRatio"]
    through_vals = df_r[df_r["BlockedLane"] == "Through"]["QueueRatio"]

    if not short_vals.empty:
        plot_data.append(short_vals.tolist())
        plot_colors.append("blue")
        plot_pos.append(base_pos - offset)
    if not through_vals.empty:
        plot_data.append(through_vals.tolist())
        plot_colors.append("red")
        plot_pos.append(base_pos + offset)

    x_labels.append(f"{r:.2f}")
    base_pos += 1

plt.figure(figsize=(16, 6))
box = plt.boxplot(plot_data, positions=plot_pos, widths=0.25, patch_artist=True)
for patch, color in zip(box['boxes'], plot_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

xtick_positions = list(range(len(x_labels)))
plt.xticks(xtick_positions, x_labels, rotation=45)
plt.axhline(1, color='gray', linestyle='--')
plt.xlabel(r'$\alpha / (1 - \alpha)$')
plt.ylabel('Queue in Blocked Lane / N')
plt.title('Queue Ratio by Blocked Lane Type for Each Lane Usage Ratio')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(handles=[
    plt.Line2D([0], [0], color='blue', lw=6, label='Short Lane Blocked', alpha=0.5),
    plt.Line2D([0], [0], color='red', lw=6, label='Through Lane Blocked', alpha=0.5)
])
plt.tight_layout()
plt.savefig('boxplot.png', dpi=300)
plt.show()

# === Plot 2: Line Plot of Mean Queue Ratio ===
plt.figure(figsize=(14, 6))
mean_df = df_plot.groupby(["AlphaRatio", "BlockedLane"])["QueueRatio"].mean().reset_index()
sns.lineplot(data=mean_df, x="AlphaRatio", y="QueueRatio", hue="BlockedLane", marker="o",
             palette={"Short": "blue", "Through": "red"})
plt.axhline(1, linestyle="--", color="gray")
plt.title("Mean Queue Ratio vs. Alpha Ratio by Blocked Lane")
plt.xlabel(r"$\alpha / (1 - \alpha)$")
plt.ylabel("Mean Queue in Blocked Lane / N")
plt.legend(title="Blocked Lane")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('lineplot.png', dpi=300)
plt.show()

# === Plot 3: Stacked Bar Plot with Theoretical Curve ===
freq_df = df_plot.groupby(["AlphaRatio", "BlockedLane"]).size().unstack(fill_value=0)
freq_df["Total"] = freq_df.sum(axis=1)
freq_df["Short"] = freq_df["Short"] / freq_df["Total"]
freq_df["Through"] = freq_df["Through"] / freq_df["Total"]

cdf = nbinom.cdf(N - 1, N, alphas)
one_minus_cdf = 1 - cdf

fig, ax1 = plt.subplots(figsize=(14, 6))
freq_df[["Short", "Through"]].plot(kind="bar", stacked=True, color=["blue", "red"], ax=ax1, alpha=0.6)

ax1.set_ylabel("Fraction of Blockages")
ax1.set_xlabel(r"$\alpha / (1 - \alpha)$")
ax1.set_title("Blockage Frequency and Theoretical Probability vs Lane Usage Ratio")
ax1.set_xticks(np.arange(len(ratios)))
ax1.set_xticklabels([f"{r:.2f}" for r in ratios], rotation=45)
ax1.grid(axis='y', linestyle='--', linewidth=0.5)
ax1.legend(title="Blocked Lane", loc="upper left")

ax1.plot(np.arange(len(ratios)), one_minus_cdf, linestyle='-', color='black', label='1 - CDF (Theory)')

plt.tight_layout()
plt.savefig('blockage_frequency.png', dpi=300)
plt.show()

# === Plot 4: Violin Plot ===
target_alphas = np.round(np.arange(0.0, 1.01, 0.1), 2)
target_ratios = [round(a / (1 - a), 2) if a < 1 else np.inf for a in target_alphas]
df_violin = df_plot[df_plot["AlphaRatio"].isin(target_ratios)].copy()

plt.figure(figsize=(14, 6))
sns.violinplot(
    data=df_violin,
    x="AlphaRatio",
    y="QueueRatio",
    hue="BlockedLane",
    split=True,
    palette={"Short": "blue", "Through": "orange"},
    inner="quartile"
)
plt.title("Queue Ratios by Blocked Lane (Selected α Values)")
plt.xlabel(r"$\alpha / (1 - \alpha)$")
plt.ylabel("Queue in Blocked Lane / N")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(title="Blocked Lane")
plt.tight_layout()
plt.savefig("violin.png", dpi=300)
plt.show()

# === Plot 5: Scatter (Bubble) Plot ===
counts_df = df_plot.groupby(["AlphaRatio", "QueueRatio"]).size().reset_index(name="Count")
counts_df["Size"] = counts_df["Count"] / counts_df["Count"].max() * 300
counts_df["Size"] = counts_df["Size"].clip(lower=20)

mean_line_df = df_plot.groupby("AlphaRatio")["QueueRatio"].mean().reset_index()

plt.figure(figsize=(14, 6))
plt.scatter(counts_df["AlphaRatio"], counts_df["QueueRatio"], s=counts_df["Size"],
            alpha=0.5, color="steelblue", edgecolors="gray", label='Simulation Results')
plt.plot(mean_line_df["AlphaRatio"], mean_line_df["QueueRatio"],
         color="darkred", marker='o', linestyle='-', linewidth=2, label="Mean Queue Ratio")
x_vals_1 = np.linspace(0.01, 1, 100)
x_vals_2 = np.linspace(1, 19, 200)
plt.plot(x_vals_1, x_vals_1, color="green", linewidth=2, label="y = x (x ≤ 1)")
plt.plot(x_vals_2, 1 / x_vals_2, color="green", linewidth=2, label="y = 1/x (x > 1)")
plt.title("Bubble Plot with Mean Queue Ratio and Reference Curves")
plt.xlabel(r"$\alpha / (1 - \alpha)$")
plt.ylabel("Queue in Blocked Lane / N")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.xticks(rotation=45)
plt.xlim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('scatter.png', dpi=300)
plt.show()

# === Plot 6: Blockage Probability ===
blockage_prob_df = df_plot.groupby("AlphaRatio").size().reset_index(name="Blockages")
blockage_prob_df["BlockageRate"] = blockage_prob_df["Blockages"] / simulation_runs

plt.figure(figsize=(14, 6))
sns.lineplot(data=blockage_prob_df, x="AlphaRatio", y="BlockageRate", marker="o", color="purple")
plt.title("Blockage Probability vs Lane Usage Ratio")
plt.xlabel(r"$\alpha / (1 - \alpha)$")
plt.ylabel("Probability of Blockage Occurrence")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.xlim(0, 1)
plt.savefig('blockage_probability.png', dpi=300)
plt.show()

mean_lines = []
Ns = range(2, 20)
for N in Ns:
    records = []
    for alpha in alphas:
        if alpha == 1:
            continue
        ratio = round(alpha / (1 - alpha), 2)
        df = event_based_simulation(simulations=simulation_runs, v=v, alpha=alpha, N=N, red=red, seed=42)
        df = df[df["BlockageOccurred"]]
        if not df.empty:
            for _, row in df.iterrows():
                records.append({
                    "N": N,
                    "AlphaRatio": ratio,
                    "QueueRatio": row["QueueInBlockedLane"] / N
                })

    df_temp = pd.DataFrame(records)
    df_mean = df_temp.groupby("AlphaRatio")["QueueRatio"].mean().reset_index()
    df_mean["N"] = N
    mean_lines.append(df_mean)

df_all_means = pd.concat(mean_lines, ignore_index=True)

# Now regenerate the individual subplots for each N
fig, axes = plt.subplots(len(Ns), 1, figsize=(8, 2.5 * len(Ns)), sharex=True)

# Define x values for reference curves
x_vals = np.linspace(0.01, 100, 500)
y_x = x_vals
y_invx = 1 / x_vals

for idx, N in enumerate(Ns):
    ax = axes[idx]
    df_n = df_all_means[df_all_means["N"] == N]
    ax.plot(df_n["AlphaRatio"], df_n["QueueRatio"], label=f"N = {N}", color="tab:blue")

    # Reference lines
    ax.plot(x_vals, y_x, '--', color='black', linewidth=1, label='y = x' if idx == 0 else "")
    ax.plot(x_vals, y_invx, '--', color='black', linewidth=2, label='y = 1/x' if idx == 0 else "")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Queue / N")
    ax.set_title(f"Mean Queue Ratio for N = {N}")
    ax.grid(True, linestyle="--", linewidth=0.5)
    if idx == len(Ns) - 1:
        ax.set_xlabel(r"$\alpha / (1 - \alpha)$")
    if idx == 0:
        ax.legend()
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("separate_mean_lines_by_N.png", dpi=300)
plt.show()

# Plot sensitivity to short lane length

alpha = 0.1
N_values = range(2, 21)
avg_queue_ratios = []
v = 1800
red = 36
simulation_runs = 1000

for N in N_values:
    df = event_based_simulation(simulations=simulation_runs, v=v, alpha=alpha, N=N, red=red, seed=42)
    if not df.empty:
        df_blocked = df[df["BlockageOccurred"]]
        if not df_blocked.empty:
            avg_queue_ratios.append(np.mean(df_blocked["QueueInBlockedLane"] / N))
        else:
            avg_queue_ratios.append(0)
    else:
        avg_queue_ratios.append(0)

plt.figure(figsize=(10, 6))
plt.plot(N_values, avg_queue_ratios, marker='o', linestyle='-', color='blue')
plt.axhline(y=alpha/(1-alpha), color='r', linestyle='--', linewidth=2)
plt.title('Average Queue/N for Blockage Cases (α = 0.15)')
plt.xlabel('N (Queue Threshold)')
plt.ylabel('Average Queue in Blocked Lane / N')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('Sensitivity_to_Length.png', dpi=300)
plt.show()

# Parameters
Ns = range(1, 16)
alphas = np.round(np.linspace(0.01, 0.51, 50), 3)

# Prepare figure with individual axes (no shared axes)
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20), sharex=False, sharey=False)
axes = axes.flatten()

# Plotting
for idx, N in enumerate(Ns):
    expected_blocking_queues = []
    for alpha in alphas:
        p_short = alpha
        p_through = 1 - alpha

        expected_val = sum(
            k * (nbinom.pmf(k, N, p_short) + nbinom.pmf(k, N, p_through))
            for k in range(N + 1)
        )
        expected_blocking_queues.append(expected_val)

    alphas_line = np.linspace(0.01, 0.99, 500)
    ref_y1 = N * alphas_line / (1 - alphas_line)

    ax = axes[idx]
    ax.plot(alphas, expected_blocking_queues, color='blue', linestyle='-', label='Expected Blocking Queue')
    ax.plot(alphas_line[alphas_line <= 0.5], ref_y1[alphas_line <= 0.5], 'r--', label='Approx: Nα/(1-α)')

    ax.set_title(f'N = {N}', fontsize=10)
    ax.set_xlabel("α", fontsize=8)
    ax.set_ylabel("E[Blocking Queue]", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, linestyle='--', linewidth=0.5)

# Remove extra blank axes if any
for j in range(len(Ns), len(axes)):
    fig.delaxes(axes[j])

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.savefig('Expected_Bonus_Flow.png')
plt.show()
