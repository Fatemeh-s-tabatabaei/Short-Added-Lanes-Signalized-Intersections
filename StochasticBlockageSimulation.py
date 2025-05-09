import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
            else:
                break

        short_lane_queue = 0
        through_lane_queue = 0
        blockage_time = None
        blocked_lane = None
        queue_at_blocked_lane = None
        blockage_occurred = False

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

        results.append({
            "BlockageOccurred": blockage_occurred,
            "BlockedLane": blocked_lane,
            "BlockageTime": blockage_time if blockage_occurred else red,
            "QueueInBlockedLane": queue_at_blocked_lane if blockage_occurred else (through_lane_queue if blocked_lane == 'through' else short_lane_queue)
        })

    return pd.DataFrame(results)

# ==== Plotting over varying alpha ====
alphas = np.arange(0, 1.01, 0.05)
N = 5
simulation_runs = 50
v = 900
red = 36

from collections import defaultdict

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

    df = event_based_simulation(simulations=simulation_runs, v=v, alpha=alpha, N=N, red=red, seed=42)
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

# === Plot 3: Histogram ===
plt.figure(figsize=(10, 5))
sns.histplot(data=df_plot, x="QueueRatio", hue="BlockedLane", element="step", stat="density",
             common_norm=False, palette={"Short": "blue", "Through": "red"})
plt.title("Histogram of Queue Ratios in Blocked Lane")
plt.xlabel("Queue in Blocked Lane / N")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.savefig('histogram.png', dpi=300)
plt.show()

# === Plot 4: Stacked Bar Plot for Frequency ===
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

plt.figure(figsize=(14, 6))
sns.violinplot(
    data=df_plot,
    x="AlphaRatio",
    y="QueueRatio",
    hue="BlockedLane",
    split=True,
    palette={"Short": "blue", "Through": "orange"},
    inner="quartile"
)

plt.title("Queue Ratios by Blocked Lane")
plt.xlabel(r"$\alpha / (1 - \alpha)$")
plt.ylabel("Queue in Blocked Lane / N")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(title="Blocked Lane")
plt.savefig('violin.png', dpi=300)
plt.show()


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
