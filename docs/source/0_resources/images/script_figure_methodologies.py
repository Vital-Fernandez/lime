import numpy as np
import lime
from matplotlib import pyplot as plt, rc_context
from lime.fitting.lines import gaussian_model, c_KMpS


# Params
OBJECT_LABEL = 'SDSS J1540'
Z_OBJ = 0.29442                     # redshift
CONTINUUM = 40.0                    # flat continuum level (counts)
NOISE_SIGMA = 9.0                   # read-out like noise floor (counts)
NOISE_GAIN = 0.25                   # photon like term: var = sigma**2 + gain * model
SEED = 42                           # reproducible noise
WAVE_LIMITS = (8465.0, 8543.0)      # observed frame plot range (angstrom)
PIXEL_SIZE = 0.35                   # observed frame sampling (angstrom/pixel)
BAND_WIDTH = 120 / c_KMpS * 6565

# Transitions: rest wavelength, amplitude scaling and cosmetics
LINES = {'H1_6563A': dict(wave_rest=6569, scale=1.000, color='#D55E00', label=r'$HI6563\AA$', rotation=0),
         'N2_6548A': dict(wave_rest=6548, scale=0.145 / 2.94, color='#C99700', label=r'$[NII]6548\AA$', rotation=0),
         'N2_6584A': dict(wave_rest=6590, scale=0.145, color='#B00020' , label=r'$[NII]6583\AA$', rotation=0)}
FIT_COLOR = '#0072B2'   # muted purple, replaces 'darkviolet'

COMPONENTS = [dict(name='broad', amp=760.0, v_offset=15.0, sigma_v=150.0, linestyle='-.'),
              dict(name='narrow', amp=960.0, v_offset=0.0, sigma_v=31.0, linestyle=':'),
              dict(name='red_wing', amp=440.0, v_offset=72.0, sigma_v=29.0, linestyle=(0, (6, 1, 1, 1, 1, 1)))]

_NARROW_COMP = next(c for c in COMPONENTS if c['name'] == 'narrow')
SPAN_SIGMA_A = (_NARROW_COMP['sigma_v'] / c_KMpS) * LINES['H1_6563A']['wave_rest']


def component_profile(wave, wave_rest, z_obj, amp, v_offset, sigma_v):
    mu = wave_rest * (1 + z_obj) * (1 + v_offset / c_KMpS)
    sigma_w = (sigma_v / c_KMpS) * mu

    return gaussian_model(wave, amp, mu, sigma_w)


def line_components(wave, line_cfg, components, z_obj):

    profile_list = []
    for comp in components:
        profile = component_profile(wave, line_cfg['wave_rest'], z_obj, comp['amp'] * line_cfg['scale'],
                                    comp['v_offset'], comp['sigma_v'])
        profile_list.append(profile)

    return profile_list


def synthetic_spectrum(wave, lines, components, z_obj, continuum, noise_sigma, noise_gain=0.0, seed=None):
    profiles = {label: line_components(wave, cfg, components, z_obj) for label, cfg in lines.items()}
    model = continuum + np.sum([np.sum(comp_list, axis=0) for comp_list in profiles.values()], axis=0)

    rng = np.random.default_rng(seed)
    sigma_pixel = np.sqrt(noise_sigma ** 2 + noise_gain * np.abs(model))
    flux = model + rng.normal(0.0, 1.0, wave.size) * sigma_pixel

    return flux, model, profiles


def continuum_axes_fraction(ax, y_data):
    """Convert a y data value into the axes-fraction coordinate, honouring
    whatever y-scale (linear or log) is currently set on the axis. Requires
    the axis's data limits to already be finalised (call after the main
    data has been plotted)."""
    ax.relim()
    ax.autoscale_view()
    frac = ax.transAxes.inverted().transform(ax.transData.transform((0, y_data)))[1]

    return float(np.clip(frac, 0.0, 1.0))


def plot_left_panel(ax, wave_rest, flux):

    ax.set_yscale('log')

    # Main spectrum first, so the axes have real data limits to autoscale from
    ax.step(wave_rest, flux, color=lime.theme.colors['fg'], linewidth=0.9, zorder=3, where='mid')

    continuum_frac = continuum_axes_fraction(ax, CONTINUUM)

    for cfg in LINES.values():
        mu = cfg['wave_rest']
        ax.axvspan(mu - BAND_WIDTH, mu + BAND_WIDTH, ymin=continuum_frac, ymax=1.0, color=lime.theme.colors['line_band'], alpha=0.40, linewidth=0, zorder=1)

    ax.axhline(CONTINUUM, color=lime.theme.colors['fg'], linestyle='--', linewidth=1.1, zorder=10)

    for label, cfg in LINES.items():
        mu = cfg['wave_rest']
        ax.text(mu, 0.02, cfg['label'], color=cfg['color'],
                rotation=0, fontsize=17, ha='center', va='bottom',
                transform=ax.get_xaxis_transform())

    ax.set_ylabel('Flux (counts)', fontsize=18, labelpad=10)
    # ax.set_xlabel(r'Rest-frame wavelength $(\AA)$', fontsize=18, labelpad=10)
    ax.set_title('Integrated flux (uniform kinematics)', fontsize=18, pad=12)

    return


def plot_right_panel(ax, wave_rest, flux, model, profiles):

    for label, comp_list in profiles.items():
        color = LINES[label]['color']
        for comp, profile in zip(COMPONENTS, comp_list):
            ax.plot(wave_rest, CONTINUUM + profile, color=color, linewidth=1, linestyle=comp['linestyle'], zorder=2)

    ax.step(wave_rest, flux, color=lime.theme.colors['fg'], linewidth=1, zorder=3, where='mid')
    ax.plot(wave_rest, model, color=FIT_COLOR, linewidth=2.6, linestyle=(0, (7, 3)), zorder=4)
    ax.axhline(CONTINUUM, color=lime.theme.colors['fg'], linestyle='--', linewidth=1.1, zorder=10)

    for label, cfg in LINES.items():
        mu = cfg['wave_rest']
        ax.text(mu, 0.02, cfg['label'], color=cfg['color'],
                rotation=0, fontsize=17, ha='center', va='bottom',
                transform=ax.get_xaxis_transform())

    ax.set_title('Multi-component flux (resolved kinematics)', fontsize=18, pad=12)
    ax.set_yscale('log')

    return


if __name__ == '__main__':

    # Plot data
    wave = np.arange(WAVE_LIMITS[0], WAVE_LIMITS[1] + PIXEL_SIZE, PIXEL_SIZE)
    flux, model, profiles = synthetic_spectrum(wave, LINES, COMPONENTS, Z_OBJ, CONTINUUM, NOISE_SIGMA, NOISE_GAIN, SEED)

    # Generate figure
    lime.theme.set_style('dark')
    fig_cfg = {'font.size': 14, 'figure.figsize': (12, 5), 'figure.dpi': 300, 'axes.labelsize': 15, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
    with rc_context(lime.theme.fig_defaults(fig_cfg)):

        wave_rest = wave / (1 + Z_OBJ)

        fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey=True)

        plot_left_panel(ax_left, wave_rest, flux)
        plot_right_panel(ax_right, wave_rest, flux, model, profiles)
        fig.supxlabel(r'Rest-frame wavelength $(\AA)$', fontsize=18, y=0.06)

        fig.tight_layout()
        # fig.savefig(f'./flux_methodology.png', bbox_inches='tight')
        fig.savefig(f'./flux_methodology_dark.png', bbox_inches='tight')
        # plt.show()