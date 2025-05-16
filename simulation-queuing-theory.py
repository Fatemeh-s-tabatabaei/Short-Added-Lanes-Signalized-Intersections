import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def event_based_simulation(simulations=1000, v=1000, alpha=1/3, N=7, red=36, seed=None):
    if seed is not None:
        np.random.seed(seed)

    results = []
    arrival_rate_per_sec = v / 3600
    departure_time_per_vehicle = 3600 / 1800  # 2 seconds per vehicle

    for _ in range(simulations):
        current_time = 0.0
        arrival_times = []
        blockage_occurred = False
        blockage_time = None
        blocked_lane = None
        queue_at_blocked_lane = None
        # Step 1: arrivals during red
        while current_time < red:
            inter_arrival = np.random.exponential(1 / arrival_rate_per_sec)
            current_time += inter_arrival
            arrival_times.append(current_time)


        # Step 2: assign to lanes
        short_lane_queue = 0
        through_lane_queue = 0
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
                "BlockageOccurred": blockage_occurred,
                "BlockedLane": blocked_lane,
                "BlockageTime": blockage_time,
                "QueueInBlockedLane": queue_at_blocked_lane
            })
            continue

        # Step 3: calculate initial discharge times
        short_discharge_end = red + short_lane_queue * departure_time_per_vehicle
        through_discharge_end = red + through_lane_queue * departure_time_per_vehicle

        # Step 4: simulate arrival during discharge and check for blockage
        next_arrival = arrival_times[-1]

        while True:

            if np.random.rand() < alpha:
                if next_arrival > short_discharge_end:
                    break
                else:
                    short_lane_queue += 1
                    if short_lane_queue == N + 1:
                        blockage_occurred = True
                        blockage_time = next_arrival
                        blocked_lane = 'through'
                        queue_at_blocked_lane = through_lane_queue  # Record value before it can change
                        break
                    short_discharge_end += departure_time_per_vehicle
            else:
                if next_arrival > through_discharge_end:
                    break
                else:

                    through_lane_queue += 1
                    if through_lane_queue == N + 1:
                        blockage_occurred = True
                        blockage_time = next_arrival
                        blocked_lane = 'short'
                        queue_at_blocked_lane = short_lane_queue  # Record value before it can change
                        break
                    through_discharge_end += departure_time_per_vehicle
            inter_arrival = np.random.exponential(1 / arrival_rate_per_sec)
            next_arrival += inter_arrival

        results.append({
            "BlockageOccurred": blockage_occurred,
            "BlockedLane": blocked_lane,
            "BlockageTime": blockage_time if blockage_occurred else max(short_discharge_end, through_discharge_end),
            "QueueInBlockedLane": queue_at_blocked_lane if blockage_occurred else np.nan
        })

    return pd.DataFrame(results)

# ==== Plotting over varying alpha ====
alphas = np.arange(0, 1.01, 0.02)
N = 7
simulation_runs = 1000
v = 1000
red = 36


x_labels = []
x_positions = []
blue_data_flat = []
red_data_flat = []

offset = 0.15
base_pos = 0

for alpha in alphas:
    if alpha == 1:
        continue
    ratio = round(alpha / (1 - alpha), 2)
    x_labels.append(f"{ratio:.2f}")
    x_positions.extend([base_pos - offset, base_pos + offset])

    df = event_based_simulation(simulations=simulation_runs, v=v, alpha=alpha, N=N, red=red)
    df = df[df["BlockageOccurred"]]

    blue_data_flat.append([row["QueueInBlockedLane"] / N for _, row in df.iterrows() if row["BlockedLane"] == "short"])
    red_data_flat.append([row["QueueInBlockedLane"] / N for _, row in df.iterrows() if row["BlockedLane"] == "through"])

    base_pos += 1

# Prepare data for plotting
plot_data = []
plot_colors = []
plot_pos = []

for i in range(len(blue_data_flat)):
    if blue_data_flat[i]:
        plot_data.append(blue_data_flat[i])
        plot_colors.append("blue")
        plot_pos.append(x_positions[2 * i])
    if red_data_flat[i]:
        plot_data.append(red_data_flat[i])
        plot_colors.append("red")
        plot_pos.append(x_positions[2 * i + 1])

# === Common DataFrame for all other plots ===
plot_records = []
for alpha in alphas:
    if alpha == 1:
        continue
    ratio = round(alpha / (1 - alpha), 2)
    df = event_based_simulation(simulations=simulation_runs, v=v, alpha=alpha, N=N, red=red, seed=42)
    df = df[df["BlockageOccurred"]]
    for _, row in df.iterrows():
        plot_records.append({
            "AlphaRatio": ratio,
            "QueueRatio": row["QueueInBlockedLane"] / N,
            "BlockedLane": "Short" if row["BlockedLane"] == "short" else "Through"
        })

df_plot = pd.DataFrame(plot_records)

fig, ax = plt.subplots(figsize=(16, 6))

unique_ratios = sorted(df_plot["AlphaRatio"].unique())
positions = []
data = []
colors = []

for i, ratio in enumerate(unique_ratios):
    short_vals = df_plot[(df_plot["AlphaRatio"] == ratio) & (df_plot["BlockedLane"] == "Short")]["QueueRatio"].values
    through_vals = df_plot[(df_plot["AlphaRatio"] == ratio) & (df_plot["BlockedLane"] == "Through")]["QueueRatio"].values

    if len(short_vals) > 0:
        data.append(short_vals)
        positions.append(i - 0.15)
        colors.append("blue")

    if len(through_vals) > 0:
        data.append(through_vals)
        positions.append(i + 0.15)
        colors.append("red")


# === Plot 1: Boxplot ===
box = plt.boxplot(plot_data, positions=plot_pos, widths=0.25, patch_artist=True)
for patch, color in zip(box['boxes'], plot_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

xtick_positions = [i for i in range(len(x_labels))]
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

# === Plot 2: Line Plot of Mean Values ===
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
plt.grid(True)
plt.tight_layout()
plt.savefig('lineplot.png', dpi=300)
plt.show()


# === Plot 3: Stacked Bar Plot for Frequency ===
blockage_counts = df_plot.groupby(["AlphaRatio", "BlockedLane"]).size().reset_index(name="Count")
total_counts = df_plot.groupby("AlphaRatio").size().reset_index(name="Total")
merged = pd.merge(blockage_counts, total_counts, on="AlphaRatio")
merged["Fraction"] = merged["Count"] / merged["Total"]

pivot_df = merged.pivot(index="AlphaRatio", columns="BlockedLane", values="Fraction").fillna(0)
pivot_df = pivot_df[["Short", "Through"]]  # Ensure order

pivot_df.plot(kind="bar", stacked=True, color=["blue", "red"], alpha=0.6, figsize=(14, 6))
plt.title("Frequency of Blocked Lane Type vs Lane Usage Ratio")
plt.xlabel(r"$\alpha / (1 - \alpha)$")
plt.ylabel("Fraction of Blockages")
plt.xticks(rotation=45)
plt.legend(title="Blocked Lane")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('blockage_frequency.png', dpi=300)
plt.show()


# === Plot 4: Violin plot for queue ratio ===
# Define target alpha values for violin plot and compute their ratios
target_alphas = np.round(np.arange(0.0, 1.01, 0.1), 2)
target_ratios = [round(a / (1 - a), 2) if a < 1 else np.inf for a in target_alphas]

# Step 2: Filter df_plot to include only those alpha ratios
df_violin = df_plot[df_plot["AlphaRatio"].isin(target_ratios)].copy()

# Step 3: Create violin plot
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


# === Plot 5: Scatter plot for the simulation ===
# Count frequency of each (AlphaRatio, QueueRatio) pair
counts_df = df_plot.groupby(["AlphaRatio", "QueueRatio"]).size().reset_index(name="Count")

# Normalize the count to control the size range
counts_df["Size"] = counts_df["Count"] / counts_df["Count"].max() * 300
counts_df["Size"] = counts_df["Size"].clip(lower=20)  # enforce visibility

# Mean line
mean_line_df = df_plot.groupby("AlphaRatio")["QueueRatio"].mean().reset_index()

plt.figure(figsize=(14, 6))

# Bubble points
plt.scatter(counts_df["AlphaRatio"], counts_df["QueueRatio"], s=counts_df["Size"],
            alpha=0.5, color="steelblue", edgecolors="gray", label='Simulation Results')

# Mean line
plt.plot(mean_line_df["AlphaRatio"], mean_line_df["QueueRatio"],
         color="darkred", marker='o', linestyle='-', linewidth=2, label="Mean Queue Ratio")

# Reference curves
x_vals_1 = np.linspace(0.01, 1, 100)
plt.plot(x_vals_1, x_vals_1, color="green", linewidth=2, label="y = x (x ≤ 1)")

x_vals_2 = np.linspace(1, 19, 200)
plt.plot(x_vals_2, 1 / x_vals_2, color="green", linewidth=2, label="y = 1/x (x > 1)")

# Axes and labels
plt.title("Bubble Plot with Mean Queue Ratio and Reference Curves")
plt.xlabel(r"$\alpha / (1 - \alpha)$")
plt.ylabel("Queue in Blocked Lane / N")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('scatter.png', dpi=300)
plt.show()


# === Plot 6: Blockage Probability ===
blockage_prob = []
for alpha in alphas:
    if alpha == 1:
        continue
    ratio = round(alpha / (1 - alpha), 2)
    df_all = event_based_simulation(simulations=simulation_runs, v=v, alpha=alpha, N=N, red=red, seed=42)
    blockage_rate = df_all["BlockageOccurred"].mean()
    blockage_prob.append({"AlphaRatio": ratio, "BlockageRate": blockage_rate})

df_prob = pd.DataFrame(blockage_prob)

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_prob, x="AlphaRatio", y="BlockageRate", marker="o", color="purple")
plt.title("Blockage Probability vs Lane Usage Ratio")
plt.xlabel(r"$\alpha / (1 - \alpha)$")
plt.ylabel("Probability of Blockage Occurrence")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('blockage_probability.png', dpi=300)
plt.show()

mean_lines = []
Ns = range(1, 21)
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
plt.xlim(0, 20)
plt.tight_layout()
plt.savefig("separate_mean_lines_by_N.png", dpi=300)
plt.show()
