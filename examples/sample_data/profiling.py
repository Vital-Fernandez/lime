import numpy as np
from matplotlib import pyplot as plt, rcParams
from lime.plots import STANDARD_PLOT

time_array = np.array([20.97, 162.74, 533.44, 1704.91, 1358.32, 1852.11])
lines_array = np.array([505, 3663, 10472, 24390, 11896, 10874])
voxels_array = np.array([11, 91, 382, 2108, 2073, 3109])
size_file_MB = 149.5

minutes_array = time_array / 60.0

print(f'Minutes array: {minutes_array} min')
print(f'Total time: {minutes_array.sum():0.2f} min = {minutes_array.sum()/60.0:0.2f} hours\n')

print(f'Lines per voxel: {lines_array/voxels_array} lines / voxels\n')

print(f'Lines speed: {lines_array/time_array} lines / second')
print(f'Voxel speed: {voxels_array/time_array} voxels / second\n')

print(f'Lines speed: {lines_array/time_array} lines / second')
print(f'Voxel speed: {voxels_array/time_array} voxels / second\n')

print(f'Lines computation: {time_array/lines_array} seconds/line, this is {np.mean(time_array/lines_array)}')
print(f'Voxel computation: {time_array/voxels_array} seconds/voxel, this is {np.mean(time_array/voxels_array)}\n')

print(f'Cumulative time: {np.cumsum(minutes_array)} minutes')
print(f'Cumulative lines: {np.cumsum(lines_array)} lines')
print(f'Cumulative voxels: {np.cumsum(voxels_array)} voxels')

# # Create the plot
# rcParams.update(STANDARD_PLOT)
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot()
# im = ax.plot(np.cumsum(minutes_array), np.cumsum(lines_array))
# plt.show()
#
# # Create the plot
# rcParams.update(STANDARD_PLOT)
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot()
# im = ax.plot(np.cumsum(minutes_array), np.cumsum(voxels_array))
# ax.update({'title': f'LiMe algorithms speed', 'xlabel': 'Time (minutes)', 'ylabel': 'Number of voxels'})
# plt.show()


# 'axes.titlesize': 18, 'axes.labelsize': 16, 'legend.fontsize': 12,
#                  'xtick.labelsize': 16, 'ytick.labelsize': 16

STANDARD_PLOT['axes.titlesize'] = 20
STANDARD_PLOT['axes.labelsize'] = 18

rcParams.update(STANDARD_PLOT)
color_lines = 'tab:blue'
color_voxels = 'tab:orange'
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot()
ax2 = ax1.twinx()

ax1.plot(np.cumsum(minutes_array), np.cumsum(lines_array), color=color_lines)
ax2.plot(np.cumsum(minutes_array), np.cumsum(voxels_array), color=color_voxels)

c_range = np.arange(time_array.size)

ax1.scatter(np.cumsum(minutes_array), np.cumsum(lines_array), c=c_range, cmap="viridis")
ax2.scatter(np.cumsum(minutes_array), np.cumsum(voxels_array), c=c_range, cmap="viridis")

ax1.set_ylabel('Number of lines', color=color_lines)
ax1.tick_params(axis='y', labelcolor=color_lines)

ax2.set_ylabel('Number of Voxels', color=color_voxels)
ax2.tick_params(axis='y', labelcolor=color_voxels)

ax1.update({'title': f'LiMe algorithms speed', 'xlabel': 'Time (minutes)'})
plt.show()
