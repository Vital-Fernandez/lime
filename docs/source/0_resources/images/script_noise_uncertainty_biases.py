#!/usr/bin/env python3
"""
Two complementary views of the pseudo-continuum noise in a line measurement.

Left panel: upper limit on the flux of an undetected line.  The limiting
transition is assumed to have a Gaussian profile with a width equal to the mean
width of the detected lines (<delta_v>) and a height equal to the
pseudo-continuum noise (sigma_c), so its integrated flux is

    F = sqrt(2 * pi) * sigma_c * <delta_v>

Right panel: the composition of an observed weak line as the sum of an
intrinsic Gaussian profile and the background noise.  For a low signal to noise
transition (1 < S/N < 5) the noise realisation decides whether the line is
recovered above the detection threshold, which biases the detected sample
towards the transitions the noise has enhanced.

Run as:
    python upper_limit_flux.py             # interactive window
    python upper_limit_flux.py plot.png    # save to file
"""

import sys

import numpy as np
from matplotlib import pyplot as plt, rc_context
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import lime

lime.theme.set_style('dark')
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WAVE_LIMITS = (4272.0, 4415.0)   # rest frame range (angstrom)
PIXEL_SIZE = 1.15                # sampling (angstrom/pixel)

CONTINUUM = 0.55                 # pseudo-continuum level (normalised flux)
SIGMA_C = 0.28                   # pseudo-continuum noise, sets the line height
MEAN_WIDTH = 2.50                # <delta_v>, mean sigma of the detected lines

DETECTED = dict(wave=4340.47, amp=1.20, sigma=MEAN_WIDTH,
                label=r'H$\gamma$', label_pos=(4340.47, 2.16))
UNDETECTED = dict(wave=4363.21, amp=SIGMA_C, sigma=MEAN_WIDTH,
                  label=r'[OIII] 4363$\AA$' + '\n upper limit', label_pos=(4364.0, 1.42))
PROFILE_WINDOW = 4.0             # half width of the drawn limit profile (sigmas)

SIGMA_MARKER_WAVE = 4312.0       # location of the sigma_c double arrow
SEED = 8                         # reproducible noise, left panel

# Right panel: weak line decomposition
WEAK_SNR = 2.0                   # intrinsic amplitude in units of sigma_c
DETECTION_SNR = 3.0              # amplitude a peak must reach to be detected
WEAK_SEED = 281                  # noise realisation which enhances the line
STACK_OFFSETS = (2.80, 1.40, 0.00)   # baselines of the three stacked traces
STACK_SCALE = 0.55               # display only scaling of the stacked traces
WEAK_CENTER = 0.5 * (WAVE_LIMITS[0] + WAVE_LIMITS[1])   # line at the panel centre
STACK_LABELS = (f'Intrinsic line (S/N = {WEAK_SNR:.0f})',
                r'Background noise ($\sigma_{c}$)',
                'Observed spectrum')

SPEC_COLOR = 'tab:orange'
FILL_COLOR = '0.75'              # trace baselines
BAND_FILL = '#EFC3A6'            # area under the limiting profile
SILVER = '#C0C0C0'

# Confidence band shades. They are flattened against the panel background once,
# at import time, so the bands and their legend patches are drawn with the very
# same opaque colour and cannot disagree.
BAND_ALPHA = 0.35
BAND_BACKGROUND = 'white'
SIGMA_TONES = ((3, '#EFC3A6'), (2, '#E09A6A'), (1, '#C9713A'))

def blend(color, alpha, background=BAND_BACKGROUND):
    """Flatten a translucent colour against an opaque background."""
    foreground = np.array(to_rgb(color))
    backdrop = np.array(to_rgb(background))

    return tuple(alpha * foreground + (1 - alpha) * backdrop)


SIGMA_BANDS = tuple((n_sigma, blend(tone, BAND_ALPHA))
                    for n_sigma, tone in SIGMA_TONES)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def gaussian_profile(wave, amp, mu, sigma):
    """Gaussian curve with peak amplitude ``amp`` centred at ``mu``."""
    return amp * np.exp(-0.5 * np.square((wave - mu) / sigma))


def synthetic_spectrum(wave, continuum, sigma_c, detected, seed=None):
    """Noisy spectrum with one detected line on a flat pseudo-continuum."""
    line = gaussian_profile(wave, detected['amp'], detected['wave'],
                            detected['sigma'])

    rng = np.random.default_rng(seed)
    flux = continuum + line + rng.normal(0.0, sigma_c, wave.size)

    return flux


def weak_line_decomposition(wave, sigma_c, snr, mu, width, seed=None, mu_noise=None):

    """Intrinsic profile, noise realisation and their sum for a weak line.

    The noise realisation is the same sequence of values regardless of where
    the line is placed. When ``mu_noise`` is provided, the array is rolled so
    that the pixels which were originally at ``mu_noise`` land on ``mu``, which
    keeps the enhancing fluctuation aligned with the line centre.
    """
    line = gaussian_profile(wave, snr * sigma_c, mu, width)
    noise = np.random.default_rng(seed).normal(0.0, sigma_c, wave.size)

    if mu_noise is not None:
        shift = np.argmin(np.abs(wave - mu)) - np.argmin(np.abs(wave - mu_noise))
        noise = np.roll(noise, shift)

    return line, noise, line + noise


def upper_limit_flux(sigma_c, mean_width):
    """Integrated flux of the limiting Gaussian profile."""
    return np.sqrt(2 * np.pi) * sigma_c * mean_width


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_upper_limit(ax, wave, flux):
    """Left panel: flux upper limit set by the pseudo-continuum noise."""

    ax.step(wave, flux, color=lime.theme.colors['fg'], linewidth=0.9, zorder=3, where='mid')

    # Pseudo-continuum and the sigma_c ceiling
    ax.axhline(CONTINUUM, color=SPEC_COLOR, linewidth=1.0, zorder=4)
    ax.axhline(CONTINUUM + SIGMA_C, color=SPEC_COLOR, linewidth=1.3,
               linestyle=(0, (7, 5)), zorder=4)

    # Limiting profile of the undetected transition
    half_width = PROFILE_WINDOW * UNDETECTED['sigma']
    idcs = np.abs(wave - UNDETECTED['wave']) <= half_width
    wave_limit = wave[idcs]
    limit_profile = gaussian_profile(wave_limit, UNDETECTED['amp'],
                                     UNDETECTED['wave'], UNDETECTED['sigma'])
    ax.fill_between(wave_limit, CONTINUUM, CONTINUUM + limit_profile,
                    color=BAND_FILL, zorder=2, alpha=0.5)
    ax.plot(wave_limit, CONTINUUM + limit_profile, color='black',
            linewidth=1.8, linestyle=':', zorder=5)

    # sigma_c double headed arrow
    ax.annotate('', xy=(SIGMA_MARKER_WAVE, CONTINUUM),
                xytext=(SIGMA_MARKER_WAVE, CONTINUUM + SIGMA_C),
                arrowprops=dict(arrowstyle='<->', color=SPEC_COLOR, linewidth=1.1),
                zorder=6)
    ax.text(SIGMA_MARKER_WAVE - 2.5, CONTINUUM + 0.5 * SIGMA_C, r'$\sigma_{c}$',
            fontsize=19, ha='right', va='center', color=SPEC_COLOR)

    # Transition labels with their pointers
    for cfg, gap in ((DETECTED, 0.14), (UNDETECTED, 0.42)):
        peak = CONTINUUM + cfg['amp']
        ax.annotate('', xy=(cfg['wave'], peak + gap/2),
                    xytext=(cfg['wave'], cfg['label_pos'][1] - 0.06),
                    arrowprops=dict(arrowstyle='-|>', color=SPEC_COLOR,
                                    linewidth=1.5, mutation_scale=18))
        ax.text(cfg['label_pos'][0], cfg['label_pos'][1], cfg['label'], fontsize=15, ha='center', va='bottom')

    ax.set_ylabel('Normalized flux', fontsize=16, labelpad=8)
    ax.set_ylim(0.0, 2.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_title('Line hidden by noise', fontsize=17, pad=12)

    return


def plot_weak_line(ax, wave, line, noise, observed):
    """Right panel: line + noise decomposition of a low S/N detection."""

    mu = WEAK_CENTER
    top, middle, bottom = STACK_OFFSETS
    label_kwargs = dict(fontsize=13, ha='left', va='bottom',
                        bbox=dict(facecolor=lime.theme.colors['bg'], edgecolor='none', pad=1.5))

    traces = ((top, line), (middle, noise), (bottom, observed))

    # Noise confidence bands around every baseline
    for offset, _ in traces:
        for n_sigma, shade in SIGMA_BANDS:
            half = STACK_SCALE * n_sigma * SIGMA_C
            ax.axhspan(offset - half, offset + half, color=shade, linewidth=0,
                       zorder=1)

    for (offset, trace), label in zip(traces, STACK_LABELS):
        ax.axhline(offset, color=FILL_COLOR, linewidth=0.8, zorder=2)
        ax.step(wave, offset + STACK_SCALE * trace, color='tab:blue', linewidth=1.1, zorder=3, where='mid')
        ax.text(WAVE_LIMITS[0] + 3.0, offset + 0.60, label, zorder=6, **label_kwargs)

    # Operators between the stacked traces
    for y_pos, symbol in (((top + middle) / 2, '+'), ((middle + bottom) / 2, '=')):
        ax.text(mu, y_pos, symbol, fontsize=22, ha='center', va='center', zorder=1, bbox=dict(facecolor=lime.theme.colors['bg'],
                                                                                              edgecolor='none', pad=2))

    handles = [Patch(facecolor=shade, edgecolor='none',
                     label=f'{n_sigma}' + r'$\sigma_{c}$')
               for n_sigma, shade in reversed(SIGMA_BANDS)]
    ax.legend(handles=handles, loc='upper right', fontsize=12, frameon=True,
              framealpha=1.0, edgecolor='0.8', handlelength=1.6, ncol=3,
              columnspacing=1.0)

    ax.set_ylim(-0.80, 3.80)
    ax.set_yticks([])
    ax.set_title('Line enhanced by noise', fontsize=17, pad=12)

    return


def plot_figure(wave, flux, line, noise, observed, output_file=None):

    fig_cfg = lime.theme.fig_defaults({'font.size': 13, "figure.figsize": [16, 6],  "figure.dpi": 400})
    with rc_context(fig_cfg):
        fig, (ax_left, ax_right) = plt.subplots(1, 2)

        plot_upper_limit(ax_left, wave, flux)
        plot_weak_line(ax_right, wave, line, noise, observed)

        for ax in (ax_left, ax_right):
            ax.set_xlim(WAVE_LIMITS)
            ax.xaxis.set_major_locator(MultipleLocator(50))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.tick_params(which='major', length=8, width=1.0, labelsize=14)
            ax.tick_params(which='minor', length=4, width=0.8)

        fig.supxlabel(r'Wavelength [ $\AA$ ]', fontsize=16, y=0.03)
        # fig.tight_layout(rect=[0, 0.055, 1, 1])

        if output_file is None:
            plt.tight_layout()
            plt.show()
        else:
            fig.savefig(output_file, bbox_inches='tight')
            plt.close(fig)

        return


if __name__ == '__main__':

    wave = np.arange(WAVE_LIMITS[0], WAVE_LIMITS[1] + PIXEL_SIZE, PIXEL_SIZE)

    flux = synthetic_spectrum(wave, CONTINUUM, SIGMA_C, DETECTED, SEED)

    line, noise, observed = weak_line_decomposition(wave, SIGMA_C, WEAK_SNR, WEAK_CENTER, MEAN_WIDTH, WEAK_SEED,
                                                    mu_noise=UNDETECTED['wave'])

    fname = f'./weak_line_bias_dark.png'
    plot_figure(wave, flux, line, noise, observed, output_file=fname)