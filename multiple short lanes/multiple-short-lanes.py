import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
import seaborn as sns

def event_based_simulation_two_short_lanes(simulations=1000, v=1800, alpha1=0.13, alpha2=0.32, N=7, red=36, seed=None):
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

        # Initialize queues
        q1, q2, q3 = 0, 0, 0  # short1, short2, through

        blockage_occurred = False
        blocked_lane = None
        queues_at_blockage = {}

        # Step 1: Red light phase assignment
        for t in arrival_times:
            r = np.random.rand()
            if r < alpha1:
                q1 += 1
                if q1 == N:
                    blockage_occurred = True
                    blocked_lane = "short1"
                    queues_at_blockage = {"short2": q2, "through": q3}
                    break
            elif r < alpha1 + alpha2:
                q2 += 1
                if q2 == N:
                    blockage_occurred = True
                    blocked_lane = "short2"
                    queues_at_blockage = {"short1": q1, "through": q3}
                    break
            else:
                q3 += 1
                if q3 == N:
                    blockage_occurred = True
                    blocked_lane = "through"
                    queues_at_blockage = {"short1": q1, "short2": q2}
                    break

        if blockage_occurred:
            results.append({
                "BlockageOccurred": True,
                "BlockedLane": blocked_lane,
                "Q_short1": q1,
                "Q_short2": q2,
                "Q_through": q3
            })
            continue

        # Step 2: Green phase processing
        d1 = red + q1 * departure_time_per_vehicle
        d2 = red + q2 * departure_time_per_vehicle
        d3 = red + q3 * departure_time_per_vehicle

        next_arrival = arrival_times[-1]
        while True:
            r = np.random.rand()
            if r < alpha1:
                if next_arrival > d1:
                    break
                q1 += 1
                if q1 == N:
                    blockage_occurred = True
                    blocked_lane = "short1"
                    queues_at_blockage = {"short2": q2, "through": q3}
                    break
                d1 += departure_time_per_vehicle
            elif r < alpha1 + alpha2:
                if next_arrival > d2:
                    break
                q2 += 1
                if q2 == N:
                    blockage_occurred = True
                    blocked_lane = "short2"
                    queues_at_blockage = {"short1": q1, "through": q3}
                    break
                d2 += departure_time_per_vehicle
            else:
                if next_arrival > d3:
                    break
                q3 += 1
                if q3 == N:
                    blockage_occurred = True
                    blocked_lane = "through"
                    queues_at_blockage = {"short1": q1, "short2": q2}
                    break
                d3 += departure_time_per_vehicle

            inter_arrival = np.random.exponential(1 / arrival_rate_per_sec)
            next_arrival += inter_arrival

        results.append({
            "BlockageOccurred": blockage_occurred,
            "BlockedLane": blocked_lane if blockage_occurred else None,
            "Q_short1": q1,
            "Q_short2": q2,
            "Q_through": q3
        })

    return pd.DataFrame(results)

# Run test
df_two_short = event_based_simulation_two_short_lanes(simulations=1000, alpha1=0.13, alpha2=0.32, N=7, red=36, seed=42)
df_two_short


# Create AlphaRatio column for consistent x-axis
df_two_short["AlphaRatio"] = round(0.13 / (1 - 0.13 - 0.32), 2)

# Prepare melted data for queue ratios
df_two_short["BlockedLane"] = df_two_short["BlockedLane"].map({
    "short1": "Auxiliary Lane",
    "short2": "RT Lane",
    "through": "Continuous Lane"
})

df_melted = pd.melt(
    df_two_short,
    id_vars=["BlockedLane", "AlphaRatio"],
    value_vars=["Q_short1", "Q_short2", "Q_through"],
    var_name="Lane",
    value_name="Queue"
)

# Normalize queue by N
df_melted["QueueRatio"] = df_melted["Queue"] / 7
df_melted["Lane"] = df_melted["Lane"].map({
    "Q_short1": "Auxiliary Lane",
    "Q_short2": "RT Lane",
    "Q_through": "Continuous Lane"
})


# === Boxplot by lane type ===
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_melted, x="Lane", y="QueueRatio", palette=["blue", "orange", "red"])
plt.axhline(1, linestyle="--", color="gray")
plt.title("Normalized Queue Lengths at Blockage Time (Two Short Lanes)")
plt.ylabel("Queue / N")
plt.xlabel("Lane")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('QueueRatioBoxplot.png')
plt.show()

# === Frequency bar plot of blocking lanes ===
plt.figure(figsize=(8, 6))
sns.countplot(data=df_two_short, x="BlockedLane", palette={"Auxiliary Lane": "blue", "RT Lane": "orange", "Continuous Lane": "red"})
plt.title("Frequency of Blocking Lane")
plt.ylabel("Count")
plt.xlabel("Blocking Lane")
plt.grid(axis='y', linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('BlockageFrequency.png')
plt.show()

# 2. Violin Plot by Blocking Lane (showing non-blocking lanes)
df_melted["BlockingLane"] = df_melted["BlockedLane"]  # bring in BlockingLane for plot
plt.figure(figsize=(14, 6))
sns.violinplot(
    data=df_melted,
    x="BlockingLane", y="QueueRatio", hue="Lane",
    palette={"Auxiliary Lane": "blue", "RT Lane": "orange", "Continuous Lane": "red"},
    split=False
)
plt.title("Queue Ratio in All Lanes Grouped by Blocking Lane")
plt.ylabel("Queue / N")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('QueueRatioViolin.png')
plt.show()
