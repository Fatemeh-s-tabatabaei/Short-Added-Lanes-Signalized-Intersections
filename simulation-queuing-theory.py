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
b = 7
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
plt.xlabel(r'$\alpha/(1-\alpha)$')
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
ax1.set_xlabel(r"$\alpha$")
# plt.axhline(0.05, color='black', linestyle='--')
# plt.axhline(0.95, color='black', linestyle='--')
ax1.set_title("Blockage Frequency and Theoretical Probability vs Lane Usage Ratio")
ax1.set_xticks(np.arange(len(ratios)))
ax1.set_xticklabels([f"{r:.2f}" for r in alphas], rotation=45)
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

from scipy.interpolate import make_interp_spline, BSpline

# Define parameter ranges
Ns = list(range(1, 21))
alphas = np.round(np.linspace(0.01, 0.51, 200), 3)

# Store alpha thresholds
alpha_thresholds = []

for N in Ns:
    threshold_alpha = None
    for alpha in alphas:
        p_short = alpha
        p_through = 1 - alpha
        expected_val = sum(
            k * (nbinom.pmf(k, N+1, p_short) + nbinom.pmf(k, N+1, p_through))
            for k in range(N+1)
        )
        ref_val = (N+1) * alpha / (1 - alpha)
        if abs(ref_val - expected_val) > 0.5:
            threshold_alpha = alpha
            break
    alpha_thresholds.append(threshold_alpha)

# === Filter NaNs ===
Ns_filtered = [n for n, a in zip(Ns, alpha_thresholds) if a is not None]
alphas_filtered = [a for a in alpha_thresholds if a is not None]

# Interpolate
N_new = np.linspace(min(Ns_filtered), max(Ns_filtered), 200)
spl = make_interp_spline(Ns_filtered, alphas_filtered, k=3)
smooth_values = spl(N_new)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(N_new, smooth_values, label="Smoothed Threshold", color='blue')
plt.plot(Ns_filtered, alphas_filtered, 'o', color='black', markersize=4, label="Threshold α")
plt.fill_between(N_new, 0, smooth_values, color='lightgray', alpha=0.4, label="Acceptable Region")
plt.xlabel("N_cont (bottleneck distance)")
plt.ylabel("short lanes preference")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("Smoothed_Thresholds.png", dpi=300)
plt.show()

from scipy.optimize import curve_fit

# Define a fitting function, e.g., exponential saturation form
def fit_func(N, a, b, c):
    return a * (1 - np.exp(-b * (N - c)))

# Fit curve to data
popt, _ = curve_fit(fit_func, Ns[1:], alpha_thresholds[1:])

# Generate smooth curve
N_fine = np.linspace(1, 20, 200)
alpha_fit = fit_func(N_fine, *popt)

# Plot
plt.rcParams['font.family'] = 'Serif'
plt.figure(figsize=(6, 4))
plt.plot(N_fine, alpha_fit, '-', color='blue')
plt.fill_between(N_fine, 0, alpha_fit, color="#84b2bd", alpha=0.4)

# Add embedded text label on shaded area
x_text = 4  # X-position of text (adjust as needed)
y_text = 0.32  # Y-position of text (adjust as needed)
plt.text(x_text, y_text, "Region in which Equation 1 \n is an acceptable approximation", fontsize=14, fontweight='bold', color='black')

# Axis labels and ticks
plt.xlabel(r"$N_{cont}$ (Bottleneck Distance)")
plt.ylabel(r"Short Lane Preference ($p_i/p_{cont}$)")
plt.ylim(0.30, 0.40)
plt.xticks(np.arange(int(N_fine.min()), int(N_fine.max()) + 1, 1))
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.tight_layout()
plt.savefig('AlphaThreshold.png')
plt.show()