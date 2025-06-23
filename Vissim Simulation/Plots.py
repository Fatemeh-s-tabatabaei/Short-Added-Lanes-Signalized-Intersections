import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
serif_font = FontProperties(family='Serif', size=16)

# Load data
df_sim = pd.read_csv('SimulationVsTheory.csv')
cycles_sim = sorted(df_sim['cycle'].unique())
data = [df_sim[df_sim['cycle'] == cycle]['SB'].values for cycle in cycles_sim]

df_capacity = pd.read_excel('TheoryCapacity.xlsx')
df_capacity = df_capacity[df_capacity['cycle'] > 19]
df_capacity_sorted = df_capacity.sort_values('cycle')

# === Plot 1: Approach Capacity ===
plt.figure(figsize=(12, 8))
box = plt.boxplot(data, positions=cycles_sim, widths=3, whis=[5, 95],
    patch_artist=True,
    boxprops=dict(facecolor='#AED6F1', color='#2980B9', linewidth=1.5),
    medianprops=dict(color='#154360', linewidth=2),
    whiskerprops=dict(color='#2980B9', linewidth=1.2),
    capprops=dict(color='#2980B9', linewidth=1.2),
    flierprops=dict(markerfacecolor='#EC7063', marker='o', markersize=5, linestyle='none'))

plt.rcParams['font.family'] = 'Serif'
plt.plot(df_capacity_sorted['cycle'], df_capacity_sorted['C1'],
         color='#E74C3C', linewidth=3, linestyle='-', label='Theoretical Capacity')

plt.xlabel('Cycle Length (seconds)', fontsize=22, fontproperties='Serif')
plt.ylabel('Capacity of Approach with Short Lane (veh/hr)', fontsize=22, fontproperties='Serif')
plt.xticks(fontproperties=serif_font)
plt.yticks(fontproperties=serif_font)
plt.ylim(1400, 1675)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

handles = [box["boxes"][0], plt.Line2D([0], [0], color='#E74C3C', linewidth=3)]
labels = ['Vissim Simulation Results', 'Theoretical Capacity']
plt.legend(handles, labels, prop=serif_font)

plt.tight_layout()
plt.savefig('SimVsTheory-Approach.png')
plt.show()

# === Plot 2: Intersection Total Capacity ===
total = [df_sim[df_sim['cycle'] == cycle]['Total Cap'].values for cycle in cycles_sim]
plt.figure(figsize=(12, 8))
box = plt.boxplot(total, positions=cycles_sim, widths=3, whis=[5, 95],
    patch_artist=True,
    boxprops=dict(facecolor='#AED6F1', color='#2980B9', linewidth=1.5),
    medianprops=dict(color='#154360', linewidth=2),
    whiskerprops=dict(color='#2980B9', linewidth=1.2),
    capprops=dict(color='#2980B9', linewidth=1.2),
    flierprops=dict(markerfacecolor='#EC7063', marker='o', markersize=5, linestyle='none'))

plt.plot(df_capacity_sorted['cycle'], df_capacity_sorted['TotalC'],
         color='#E74C3C', linewidth=3, linestyle='-', label='Theoretical Capacity')

plt.xlabel('Cycle Length (seconds)', fontsize=22, fontproperties='Serif')
plt.ylabel('Capacity of the Intersection (veh/hr)', fontsize=22, fontproperties='Serif')
plt.xticks(fontproperties=serif_font)
plt.yticks(fontproperties=serif_font)
plt.ylim(2100, 2550)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

handles = [box["boxes"][0], plt.Line2D([0], [0], color='#E74C3C', linewidth=3)]
labels = ['Vissim Simulation Results', 'Theoretical Capacity']
plt.legend(handles, labels, prop=serif_font)

plt.tight_layout()
plt.savefig('SimVsTheory-Intersection.png')
plt.show()
