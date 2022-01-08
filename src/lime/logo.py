# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib import rcParams
#
# from matplotlib import font_manager
#
# font_manager._rebuild()
# font_manager.findfont('MTF Saxy', rebuild_if_missing=True)
#
# print(font_manager.win32FontDirectory())
#
# # build a rectangle in axes coords
# left, width = .25, .5
# bottom, height = .25, .5
# right = left + width
# top = bottom + height
#
# fig = plt.figure()
# ax = fig.add_axes([0, 0, 1, 1])
#
# rcParams['font.family'] = ['MTF Saxy']
#
# ax.text(0.5*(left+right), 0.5*(bottom+top), 'LiMe',
#         horizontalalignment='center',
#         verticalalignment='center',
#         fontsize=200, color='red',
#         transform=ax.transAxes)
#
# ax.set_axis_off()
# plt.show()

import numpy as np
import lime as lm
from lime.model import gaussian_model
from matplotlib import pyplot as plt, rcParams
from matplotlib import font_manager
from pathlib import Path

np.random.seed(2)

rcParams['font.family'] = ['MTF Saxy']

curve_dict = {'comp1': {'amp':0.75, 'center':1.00, 'sigma': 2.0},
              'comp2': {'amp':0.65, 'center':6.80, 'sigma': 1.8}}

cont = 0.6
err = 0.025
wave = np.linspace(-30, 30, 100)
noise = np.random.normal(0.0, err, size=wave.size)

flux_dict = {}
for curve, params in curve_dict.items():
        flux_dict[curve] = gaussian_model(wave, **params)
flux_comb = flux_dict['comp1'] + flux_dict['comp2'] + cont + noise

fig, ax = plt.subplots(dpi=300)

w3, w4 = 0, -1
w3, w4 = np.searchsorted(wave, (-3.0, 11))


flux_comb = flux_dict['comp1'] + flux_dict['comp2'] + cont + noise
ax.step(wave[w3:w4], flux_comb[w3:w4], where='mid', color='black', linewidth=4)

for curve, flux in flux_dict.items():
        ax.plot(wave[w3:w4], flux[w3:w4] + cont, '--', linewidth=1)


dodge = - 0.1
residual = flux_comb - (flux_dict['comp1'] + flux_dict['comp2']) + dodge
ax.step(wave, residual, where='mid', color='black')

ax.fill_between(wave, -err+cont+dodge, err+cont+dodge, facecolor='tab:red', alpha=0.5)

# ax.text(-1, 0.9, 'LiMe',
#         horizontalalignment='center',
#         verticalalignment='center',
#         fontsize=200, color='red', alpha=0.5)

ax.text(-13.5, 0.9, 'L',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=200, color='black')

ax.text(-5.5, 0.9, 'i',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=200, color='black')

ax.text(14, 0.9, 'e',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=200, color='black')

ax.set_ylim(0.4, 1.4)
ax.set_xlim(-18, 17)

ax.axis('off')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

plt.tight_layout()
doc_images_folder = Path('../../docs/source/_static/')
plt.savefig(doc_images_folder/'logo_transparent.png', bbox_inches='tight', transparent=True)
plt.savefig(doc_images_folder/'logo_white.png', bbox_inches='tight')
plt.show()
