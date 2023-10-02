from matplotlib import pyplot as plt, rc_context
from lime.plots import STANDARD_PLOT
import pandas as pd
import numpy as np


def plot_lines_minutes(df):

    spaxels = df.Spaxels.to_numpy()
    lines = df.Lines.to_numpy()
    min = df.Minutes.to_numpy()

    # fig, ax = plt.subplots(dpi=200)
    # ax.plot(np.cumsum(min), np.cumsum(lines), label='Lines')
    # ax.plot(np.cumsum(min), np.cumsum(spaxels), label='Spaxels')
    # ax.legend()
    # plt.show()

    """ALFA took about 20 h to analyse the cube, during which time 41 022 pixels were analysed and just 
    over two million emission lines were fitted"""

    print(f'\nLiMe MUSE Cube with {spaxels.sum()} spaxels, {lines.sum()} lines for {min.sum()} minutes')
    print(f'{np.round(lines / (min*60), 0)} lines per second')
    print(f'{np.round(spaxels / (min*60), 0)} spaxels per second')
    print(f'Mean speed {lines.sum()/(min.sum()*60):0.2f}')
    print('\nALFA')
    print(f'{np.round(2000000 / (20*60*60), 0)} lines per second')
    print(f'{np.round(41022 / (20*60*60), 0)} spaxels per second')

    STANDARD_PLOT['axes.titlesize'] = 25
    STANDARD_PLOT['axes.labelsize'] = 22
    STANDARD_PLOT['xtick.labelsize'] = 18
    STANDARD_PLOT['ytick.labelsize'] = 18

    with rc_context(STANDARD_PLOT):

        color_lines = 'tab:blue'
        color_voxels = 'tab:orange'
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot()
        ax2 = ax1.twinx()

        ax1.plot(np.cumsum(minutes_array), np.cumsum(lines_array), color=color_lines)
        ax2.plot(np.cumsum(minutes_array), np.cumsum(spaxels), color=color_voxels)

        c_range = np.arange(min.size)

        ax1.scatter(np.cumsum(minutes_array), np.cumsum(lines_array), c=color_lines)
        ax2.scatter(np.cumsum(minutes_array), np.cumsum(spaxels), c=color_voxels)

        ax1.set_ylabel('Number of lines', color=color_lines)
        ax1.tick_params(axis='y', labelcolor=color_lines)

        ax2.set_ylabel('Number of Spaxels', color=color_voxels)
        ax2.tick_params(axis='y', labelcolor=color_voxels)

        ax1.update({'title': r'$LiMe$ benchmarks', 'xlabel': 'Time (minutes)'})

        plt.show()
        # plt.savefig('../../docs/source/_static/benchmarks.png', bbox_inches='tight', transparent=True)
    # 
    return

# Test 3) Saving fits file log every 1000 spaxels + terminal coord display
spaxels_array = np.array([11, 91, 382, 2108, 2073, 3109])
lines_array = np.array([459, 3418, 3431, 17390, 9579, 12498])
minutes_array = np.array([0.08, 0.65, 0.76, 4.26, 3.67, 5.84])
data = {'Spaxels': spaxels_array, 'Lines': lines_array, 'Minutes': minutes_array}
df_test3 = pd.DataFrame(data)

plot_lines_minutes(df_test3)

# # Display the DataFrame
# print(df)
# In this example, we first import Pandas and NumPy. Then, we create three NumPy arrays (array1, array2, and array3). We create a dictionary data where the keys are the column names, and the values are the NumPy arrays. Finally, we use the pd.DataFrame constructor to create a DataFrame from the dictionary.
#
# You can customize the column names by changing the keys in the data dictionary or by setting the columns parameter when creating the DataFrame. For example:
#
# python
# Copy code
# df = pd.DataFrame(data, columns=['CustomName1', 'CustomName2', 'CustomName3'])
# This will create a DataFrame with custom column names.
#












# Test 3) Saving fits file log every 1000 spaxels + terminal coord display
#
# Spatial mask 1/6) MASK_0 (11 spaxels)
# [==========] 100% of mask (Coord. 169-170)
# 459 lines measured in 0.08 minutes.
#
# Spatial mask 2/6) MASK_1 (91 spaxels)
# [==========] 100% of mask (Coord. 173-168)
# 3418 lines measured in 0.65 minutes.
#
# Spatial mask 3/6) MASK_2 (382 spaxels)
# [==========] 100% of mask (Coord. 208-186)
# 3431 lines measured in 0.76 minutes.
#
# Spatial mask 4/6) MASK_3 (2108 spaxels)
# [==========] 100% of mask (Coord. 213-191)
# 17390 lines measured in 4.26 minutes.
#
# Spatial mask 5/6) MASK_4 (2073 spaxels)
# [==========] 100% of mask (Coord. 215-192)
# 9579 lines measured in 3.67 minutes.
#
# Spatial mask 6/6) MASK_5 (3109 spaxels)
# [==========] 100% of mask (Coord. 228-169)
# 12498 lines measured in 5.84 minutes.
