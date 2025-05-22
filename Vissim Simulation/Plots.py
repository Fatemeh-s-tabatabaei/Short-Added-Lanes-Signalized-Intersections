import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Load the simulation data and group by cycle for the \"Bottleneck\" values.
df_sim = pd.read_excel('Validation-SimulationVsTheory.xlsx')
cycles_sim = sorted(df_sim['cycle'].unique())
data = [df_sim[df_sim['cycle'] == cycle]['Bottleneck'].values for cycle in cycles_sim]

# Load the capacity data and sort by cycle.
df_capacity = pd.read_excel('TheoryCapacity.xlsx')
df_capacity = df_capacity[df_capacity['cycle'] > 19]
df_capacity_sorted = df_capacity.sort_values('cycle')


plt.figure(figsize=(12, 8))

# Enhanced boxplots
box = plt.boxplot(data, positions=cycles_sim, widths=3, whis=[5, 95],
            patch_artist=True,
            boxprops=dict(facecolor='#AED6F1', color='#2980B9', linewidth=1.5),
            medianprops=dict(color='#154360', linewidth=2),
            whiskerprops=dict(color='#2980B9', linewidth=1.2),
            capprops=dict(color='#2980B9', linewidth=1.2),
            flierprops=dict(markerfacecolor='#EC7063', marker='o', markersize=5, linestyle='none'))

# Enhanced theory plot
plt.plot(df_capacity_sorted['cycle'], df_capacity_sorted['C1'],
         color='#E74C3C', linewidth=3, linestyle='-', label='Theoretical Capacity')

# Titles and labels
plt.xlabel('Cycle Length (seconds)', fontsize=18, fontproperties='Times New Roman')
plt.ylabel('Capacity of Approach with Short Lane (veh/hr)', fontsize=18, fontproperties='Times New Roman')
# plt.title('Simulation vs. Theoretical Capacity', fontsize=22, fontproperties='Times New Roman', pad=10)

# Legend
handles = [box["boxes"][0], plt.Line2D([0], [0], color='#E74C3C', linewidth=3)]
labels = ['Vissim Simulation Results', 'Theoretical Capacity']
plt.legend(handles, labels, prop='Times New Roman', fontsize=25)
legend = plt.legend(prop='Times New Roman', fontsize=22)
plt.setp(legend.get_texts(), fontsize=25)

# Grid and layout
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(fontsize=18, fontproperties='Times New Roman')
plt.yticks(fontsize=18, fontproperties='Times New Roman')
plt.tight_layout()
plt.savefig('SimVsTheory-Approach.png')
plt.show()