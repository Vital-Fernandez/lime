import numpy as np
from lime.model import gaussian_model
from matplotlib import pyplot as plt, rcParams
from matplotlib import font_manager
from pathlib import Path
import shutil
import matplotlib

np.random.seed(2)

rcParams['font.family'] = ['Caveat Brush']
rcParams['hatch.linewidth'] = 0.2

curve_dict = {'comp1': {'amp':  0.55, 'center': 1.00, 'sigma': 2.0},
              'comp2': {'amp':  0.45, 'center': 6.80, 'sigma': 1.8}}

cont = 1
err = 0.025
wave = np.linspace(-30, 30, 100)
wave_g = np.linspace(-30, 30, 1000)
noise = np.random.normal(0.0, err, size=wave.size)

flux_dict = {}
for curve, params in curve_dict.items():
        flux_dict[curve] = gaussian_model(wave, **params)
flux_comb = flux_dict['comp1'] + flux_dict['comp2'] + cont + noise

flux_dict_g = {}
for curve, params in curve_dict.items():
        flux_dict_g[curve] = gaussian_model(wave_g, **params)


with plt.xkcd():

        dpi = 600
        fig, ax = plt.subplots(dpi=dpi)

        w0, w7 = np.searchsorted(wave, (-20, 26))
        w3, w4 = np.searchsorted(wave, (-3.8, 12))
        w1, w2 = np.searchsorted(wave, (-17, -10))
        w5, w6 = np.searchsorted(wave, (17, 23))

        ax.step(wave[w0:w7], flux_comb[w0:w7], where='mid', color='black', linewidth=3)

        ax.fill_between(wave[w3:w4], 0, flux_comb[w3:w4], facecolor='white', alpha=0.05, hatch='///')
        ax.fill_between(wave[w1:w2], 0, flux_comb[w1:w2], facecolor='white', alpha=0.05, hatch='///')
        ax.fill_between(wave[w5:w6], 0, flux_comb[w5:w6], facecolor='white', alpha=0.05, hatch='///')


        # Band boundaries
        fs = 25
        ax.text(-16.35, 1.10, 'w1', horizontalalignment='center', verticalalignment='center', fontsize=fs, color='black')
        ax.text(-10.55, 1.13, 'w2', horizontalalignment='center', verticalalignment='center', fontsize=fs, color='black')
        ax.text(-4.5, 1.15, 'w3', horizontalalignment='center', verticalalignment='center', fontsize=fs, color='black')
        ax.text(12.25, 1.10, 'w4', horizontalalignment='center', verticalalignment='center', fontsize=fs, color='black')
        ax.text(17.4, 1.10, 'w5', horizontalalignment='center', verticalalignment='center', fontsize=fs, color='black')
        ax.text(22.7, 1.10, 'w6', horizontalalignment='center', verticalalignment='center', fontsize=fs, color='black')

        ax.set_ylim(0.0, 2)
        # ax.set_xlim(-18, 17)

        ax.axis('off')
        # ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.tight_layout()
        doc_images_folder = Path('../../../docs/source/_static/')
        plt.savefig(doc_images_folder/'band_sketch_transparent.png', bbox_inches='tight', transparent=True)
        plt.savefig(doc_images_folder/'band_sketc_white.png', bbox_inches='tight')
        print(f'Saving {doc_images_folder}')
        # plt.show()
