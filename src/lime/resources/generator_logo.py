import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt, rc_context, font_manager

from lime.io import _LIME_FOLDER
from lime.plotting.plots import theme
from lime.fitting.lines import gaussian_model

LOGO_FOLDER = _LIME_FOLDER.parent.parent/'examples/images'
LOGO_ADDRESS = LOGO_FOLDER / "LiMe2_logo_white.png"
LOGO_ADDRESS_DARK = LOGO_FOLDER / "LiMe2_logo_dark.png"
transparent = False

def lime_log_function(output_path: Path = None, seed: int = 4, show: bool = False,
                  style: str = 'default'):

        rng = np.random.default_rng(seed)
        font_manager._load_fontmanager(try_read_cache=False)

        # Define curve parameters
        curve_dict = {'comp1': {'amp': 0.75, 'center': 1.00, 'sigma': 2.0},
                      'comp2': {'amp': 0.65, 'center': 6.80, 'sigma': 1.8}}

        cont = 0.6
        err = 0.025
        wave = np.linspace(-30, 30, 100)
        wave_g = np.linspace(-30, 30, 1000)
        noise = rng.normal(0.0, err, size=wave.size)

        flux_dict = {k: gaussian_model(wave, **v) for k, v in curve_dict.items()}
        flux_comb = sum(flux_dict.values()) + cont + noise

        flux_dict_g = {k: gaussian_model(wave_g, **v) for k, v in curve_dict.items()}

        theme.set_style(style)
        fig_cfg = theme.fig_defaults()
        fig_cfg['font.family'] = ['MTF Saxy']
        fig_cfg['figure.dpi'] = 300

        with rc_context(fig_cfg):
                fig, ax = plt.subplots()

                w3, w4 = np.searchsorted(wave, (-3.0, 11))
                w_cross1, w_cross2 = np.searchsorted(wave_g, (4.352, 3.942))

                ax.step(wave[w3:w4], flux_comb[w3:w4], where='mid', color=theme.colors['fg'], linewidth=3)

                ax.plot(wave_g[0:w_cross1], flux_dict_g['comp1'][0:w_cross1] + cont, '--', linewidth=1.5)
                ax.plot(wave_g[w_cross2:-1], flux_dict_g['comp2'][w_cross2:-1] + cont, '--', linewidth=1.5)

                dodge = 0
                residual = flux_comb - sum(flux_dict.values()) + dodge
                ax.step(wave, residual, where='mid', color=theme.colors['fg'])
                ax.fill_between(wave, -err + cont + dodge, err + cont + dodge, facecolor='tab:red', alpha=0.5)

                for letter, pos in zip('Lie', [-13.5, -5.5, 14]):
                        ax.text(pos, 0.9, letter,
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=150, color=theme.colors['fg'])

                ax.set_ylim(0.4, 1.4)
                ax.set_xlim(-18, 17)
                ax.axis('off')

                plt.tight_layout()

                if output_path:
                        plt.savefig(output_path, bbox_inches='tight', transparent=transparent)

                if show:
                        plt.show()

        return fig


if __name__ == "__main__":

        lime_log_function(LOGO_ADDRESS, show=True, style='default')
        lime_log_function(LOGO_ADDRESS_DARK, show=True, style='dark')