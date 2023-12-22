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

    STANDARD_PLOT['figure.figsize'] = (8, 7)
    STANDARD_PLOT['axes.titlesize'] = 25
    STANDARD_PLOT['axes.labelsize'] = 25
    STANDARD_PLOT['xtick.labelsize'] = 20
    STANDARD_PLOT['ytick.labelsize'] = 20
    # STANDARD_PLOT['font.family'] = 'Times New Roman'
    STANDARD_PLOT['mathtext.fontset'] = 'cm'

    with rc_context(STANDARD_PLOT):

        color_lines = 'tab:blue'
        color_voxels = 'tab:orange'
        fig = plt.figure()
        ax1 = fig.add_subplot()
        ax2 = ax1.twinx()

        ax1.plot(np.cumsum(minutes_array), np.cumsum(lines_array), color=color_lines)
        ax2.plot(np.cumsum(minutes_array), np.cumsum(spaxels), color=color_voxels)

        ax1.scatter(np.cumsum(minutes_array), np.cumsum(lines_array), c=color_lines)
        ax2.scatter(np.cumsum(minutes_array), np.cumsum(spaxels), c=color_voxels)

        ax1.set_ylabel('Number of lines', color=color_lines)
        ax1.tick_params(axis='y', labelcolor=color_lines)

        ax2.set_ylabel('Number of spaxels', color=color_voxels)
        ax2.tick_params(axis='y', labelcolor=color_voxels)

        ax1.update({'xlabel': 'Time (minutes)'})
        plt.tight_layout()
        plt.show()
        # plt.savefig('benchmarks.png', bbox_inches='tight')
    # 
    return

# Test 3) Saving fits file log every 1000 spaxels + terminal coord display
spaxels_array = np.array([11, 90, 383, 2108, 2073, 3109])
lines_array = np.array([459, 3381, 3440, 17390, 9579, 12498])
minutes_array = np.array([0.08, 0.62, 0.71, 3.78, 2.69, 3.89])


data = {'Spaxels': spaxels_array, 'Lines': lines_array, 'Minutes': minutes_array}
df_test3 = pd.DataFrame(data)

plot_lines_minutes(df_test3)












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
