import numpy as np
from typing import Callable, NamedTuple
from uncertainties import ufloat
from scipy.special import wofz

# ---------- Support constants
tiny_limit = 1.0e-15
k_GaussArea = np.sqrt(2 * np.pi)
sqrt2 = np.sqrt(2)
k_gFWHM = 2 * np.sqrt(2 * np.log(2))   # Gaussian FWHM = k_gFWHM * sigma
k_eFWHM = 2 * np.log(2)                # Exponential FWHM = k_eFWHM / alpha


# ---------- Curves theoretical formulas
def gaussian_model(x, amp, center, sigma):
    """G(x) = A exp[-(x-c)^2 / (2 s^2)]"""
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentz_model(x, amp, center, sigma):
    """L(x) = A s^2 / [(x-c)^2 + s^2]      (sigma = half width at half maximum)"""
    return amp * (sigma * sigma) / ((x - center) ** 2 + sigma * sigma)


def voigt_model(x, amp, center, sigma, gamma):
    """V(x) = A Re[w(z)] / (s sqrt(2 pi)),  z = (x-c + i g) / (s sqrt2)
    w = Faddeeva function; unit-area Voigt times A (A is the area);
    sigma = Gaussian std, gamma = Lorentzian HWHM."""
    z = (x - center + 1j * gamma) / max(tiny_limit, sigma * sqrt2)
    return amp * np.real(wofz(z)) / max(tiny_limit, sigma * k_GaussArea)


def exponential_model(x, amp, center, alpha):
    """E(x) = A exp(-alpha |x-c|)"""
    return amp * np.exp(-alpha * np.abs(x - center))


def pseudo_voigt_model(x, amp, center, sigma, frac):
    """PV(x) = f G(x) + (1-f) L(x)       (same A and sigma in both)"""
    return frac * gaussian_model(x, amp, center, sigma) + (1 - frac) * lorentz_model(x, amp, center, sigma)


def broken_powerlaw_model(x, a, b, c, alpha):
    """P(x) = a |x-c|^-alpha for |x-c| >= b, and 0 inside the break |x-c| < b"""
    wave_shift_abs = np.abs(x - c)
    y = a * np.power(wave_shift_abs, -alpha)
    y[wave_shift_abs < b] = 0.0
    return y


def pseudo_power_model(x, amp, center, sigma, alpha, frac):
    """PP(x) = f G(x) + (1-f) P(x; a=amp, b=sigma, c=center)"""
    return (frac * gaussian_model(x, amp, center, sigma)
            + (1 - frac) * broken_powerlaw_model(x, amp, sigma, center, alpha))


# ---------- FWHM theoretical formulas

def g_FWHM(p):
    return k_gFWHM * p['sigma']                     # 2 sqrt(2 ln2) s


def l_FWHM(p):
    return 2 * p['sigma']                           # 2 s


def v_FWHM(p):
    # Olivero & Longbothum / Kielkopf: 0.5346 fL + sqrt(0.2166 fL^2 + fG^2)
    f_g, f_l = k_gFWHM * p['sigma'], 2 * p['gamma']
    return 0.5346 * f_l + (0.2166 * f_l * f_l + f_g * f_g) ** 0.5


def pv_FWHM(p):
    # Linear mix of both widths; within ~1% of the numerical half-maximum width
    f_g, f_l = k_gFWHM * p['sigma'], 2 * p['sigma']
    return p['frac'] * f_g + (1 - p['frac']) * f_l


def pp_FWHM(p):
    return g_FWHM(p)                                # core assumed Gaussian


def p_FWHM(p):
    return ufloat(np.nan, np.nan)


def e_FWHM(p):
    return k_eFWHM / p['alpha']                     # 2 ln2 / alpha


# ---------- Profile theoretical areas
def gaussian_area(p):
    return k_GaussArea * p['amp'] * p['sigma']      # sqrt(2 pi) A s


def lorentz_area(p):
    return np.pi * p['amp'] * p['sigma']            # pi A s


def voigt_area(p):
    return p['amp']                                 # A multiplies a unit-area profile


def pseudo_voigt_area(p):
    return p['frac'] * gaussian_area(p) + (1 - p['frac']) * lorentz_area(p)


def power_area(p):
    return ufloat(np.nan, np.nan)


def pseudo_power_area(p):
    return pseudo_voigt_area(p)                     # placeholder: ignores the power-law wing


def exp_area(p):
    return 2 * p['amp'] / p['alpha']                # 2 A / alpha


# ---------- Compilation class

class LineProfile(NamedTuple):
    params: tuple[str, ...]
    area: Callable
    fwhm: Callable


PROFILES = {'g':  LineProfile(('amp', 'center', 'sigma'),                    gaussian_area,     g_FWHM),
            'l':  LineProfile(('amp', 'center', 'sigma'),                    lorentz_area,      l_FWHM),
            'v':  LineProfile(('amp', 'center', 'sigma', 'gamma'),           voigt_area,        v_FWHM),
            'pv': LineProfile(('amp', 'center', 'sigma', 'frac'),            pseudo_voigt_area, pv_FWHM),
            'pp': LineProfile(('amp', 'center', 'sigma', 'alpha', 'frac'),   pseudo_power_area, pp_FWHM),
            'p':  LineProfile(('a', 'b', 'c', 'alpha'),                      power_area,        p_FWHM),
            'e':  LineProfile(('amp', 'center', 'alpha'),                    exp_area,          e_FWHM)}