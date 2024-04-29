import numpy as np
from lime.model import gaussian_model
from matplotlib import pyplot as plt, rcParams
from matplotlib import font_manager
from pathlib import Path
import matplotlib

matplotlib.font_manager._load_fontmanager(try_read_cache=False)

np.random.seed(2)

rcParams['font.family'] = ['MTF Saxy']

curve_dict = {'comp1': {'amp':0.75, 'center':1.00, 'sigma': 2.0},
              'comp2': {'amp':0.65, 'center':6.80, 'sigma': 1.8}}

cont = 0.6
err = 0.025
wave = np.linspace(-30, 30, 100)
wave_g = np.linspace(-30, 30, 1000)
noise = np.random.normal(0.0, err, size=wave.size)

# color_fg = 'black'
color_fg = np.array((179, 199, 216))/255.0

flux_dict = {}
for curve, params in curve_dict.items():
        flux_dict[curve] = gaussian_model(wave, **params)
flux_comb = flux_dict['comp1'] + flux_dict['comp2'] + cont + noise

flux_dict_g = {}
for curve, params in curve_dict.items():
        flux_dict_g[curve] = gaussian_model(wave_g, **params)

dpi = 200 #600
fig, ax = plt.subplots(dpi=200)

w3, w4 = np.searchsorted(wave, (-3.0, 11))
w_cross1, w_cross2 = np.searchsorted(wave_g, (4.352, 3.942))

ax.step(wave[w3:w4], flux_comb[w3:w4], where='mid', color=color_fg, linewidth=3)

# for curve, flux in flux_dict.items():
#         ax.plot(wave[w3:w4], flux[w3:w4] + cont, '--', linewidth=1)
ax.plot(wave_g[0:w_cross1], flux_dict_g['comp1'][0:w_cross1] + cont, '--', linewidth=1.5)
ax.plot(wave_g[w_cross2:-1], flux_dict_g['comp2'][w_cross2:-1] + cont, '--', linewidth=1.5)

dodge = 0 #- 0.1
residual = flux_comb - (flux_dict['comp1'] + flux_dict['comp2']) + dodge
ax.step(wave, residual, where='mid', color=color_fg)

ax.fill_between(wave, -err+cont+dodge, err+cont+dodge, facecolor='tab:red', alpha=0.5)

# _ax.text(-1, 0.9, 'LiMe',
#         horizontalalignment='center',
#         verticalalignment='center',
#         fontsize=200, color='red', alpha=0.5)

ax.text(-13.5, 0.9, 'L',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=200, color=color_fg)

ax.text(-5.5, 0.9, 'i',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=200, color=color_fg)

ax.text(14, 0.9, 'e',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=200, color=color_fg)

ax.set_ylim(0.4, 1.4)
ax.set_xlim(-18, 17)

ax.axis('off')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

plt.tight_layout()
doc_images_folder = Path('../../docs/source/_static/')
plt.savefig(doc_images_folder/'logo_dark_transparent.png', bbox_inches='tight', transparent=True)
# plt.show()

# plt.savefig(doc_images_folder/'logo_transparent.png', bbox_inches='tight', transparent=True)
# plt.savefig(doc_images_folder/'logo_white.png', bbox_inches='tight')
# print(f'Saving {doc_images_folder}')
