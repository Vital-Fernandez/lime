# import astropy
# from astropy import units as au
#
# from astropy.units.core import CompositeUnit, IrreducibleUnit, PrefixUnit, Unit
#
# flux_keyword_lists = ['F_lam', 'f_la', 'flam', 'Flam', 'f_lam', 'FLAM', 'F_LAM']
# for keyword in flux_keyword_lists:
#     try:
#         unit_variable = au.Unit(keyword)
#         print(f'{keyword} is a VALID unit')
#     except ValueError:
#         print(f'{keyword} is NOT a valid unit')

import astropy
from astropy import units as au
import synphot.units as su

# au.add_enabled_aliases({'F_lam': au.erg/au.s/au.cm**2/au.AA, 'F_nu': au.erg/au.s/au.cm**2/au.Hz,
#                         'f_E': au.photon/au.s/au.cm**2/au.keV, 'f_lam': au.photon/au.s/au.cm**2/au.AA,
#                         'WN': 1/au.cm})

flam = au.def_unit(['flam', 'FLAM'], au.erg/au.s/au.cm**2/au.AA,
                    format={"latex": r"erg\,cm^{-2}s^{-1}\AA^{-1}",
                            "generic": "FLAM", "console": "FLAM"})

fnu = au.def_unit(['fnu', 'FNU'], au.erg/au.s/au.cm**2/au.Hz,
                    format={"latex": r"erg\,cm^{-2}s^{-1}Hz^{-1}",
                            "generic": "FNU", "console": "FNU"})

photlam = au.def_unit(['photlam', 'PHOTLAM'], au.photon/au.s/au.cm**2/au.AA,
                        format={"latex": r"photon\,cm^{-2}s^{-1}\AA^{-1}",
                        "generic": "PHOTLAM", "console": "PHOTLAM"})

photnu = au.def_unit(['photnu', 'PHOTNU'], au.photon/au.s/au.cm**2/au.Hz,
                        format={"latex": r"photon\,cm^{-2}s^{-1}Hz^{-1}",
                        "generic": "PHOTNU", "console": "PHOTNU"})

au.add_enabled_units([flam, fnu, photlam, photnu])

# 'Flam': r'erg\,cm^{-2}s^{-1}\AA^{-1}',
# 'Fnu': r'erg\,cm^{-2}s^{-1}\Hz^{-1}',

for keyword in ['photnu', 'PHOTNU', 'photlam', 'PHOTLAM', 'f_lam', 'flam', '1e17*flam', 'AA', 'Angstrom', 'Pa', 'atm']:

    try:
        unit_variable = au.Unit(keyword)
        print(f'{keyword} is a VALID unit, {unit_variable:latex}')
        print(isinstance(unit_variable, au.UnitBase))
    except ValueError:
        print(f'{keyword} is NOT a valid unit')

# format={"latex": r"\mathring{A}", "unicode": "Å", "vounit": "Angstrom"},
su.FLAM
# {"latex": r"\mathring{A}", "unicode": "Å", "vounit": "Angstrom"}
# FLAM = u.def_unit(
#     'flam', u.erg / (u.cm**2 * u.s * u.AA),
#     format={'generic': 'FLAM', 'console': 'FLAM'})

# import numpy as np
# from matplotlib import pyplot as plt
#
#
# def gaussian_model(x, amp, center, sigma):
#     """1-d gaussian curve : gaussian(x, amp, cen, wid)"""
#     return amp * np.exp(-0.5 * (((x-center)/sigma) * ((x-center)/sigma)))
#
#
# def broken_powerlaw(x, a, b, c, alpha):
#
#     # Compute symmetric power law
#     wave_shift_abs = np.abs(x - c)
#     y = a * np.power(wave_shift_abs/b, alpha)
#
#     # Set broken region to zero
#     idcs_core = wave_shift_abs < b
#     y[idcs_core] = 0.0
#
#     return y
#
# def pseudo_power_law(x, amp, center, sigma, frac, a, b, c, alpha):
#
#     y = frac * gaussian_model(x, amp, center, sigma) + (1 - frac) * broken_powerlaw(x, a, b, c, alpha)
#
#     return y
#
#
# # Profile parameters
# amp = 125.0
# mu = 4862.55
# sigma = 1.55
# gamma = 2.55
# m_cont = 0
# n_cont = 10
# noise = 2
# frac = 0.9
# n_pixels = 500
# off = -2
# exp = -2
# x_break = 0.5
#
# # Arrays with the independent variables
# x_array = np.linspace(mu - 75, mu + 75, n_pixels)
#
# # gaussian_array = gaussian_model(x_array, amp, mu, sigma)
#
# # broken_power = broken_powerlaw(x_array, amp, 0.01*sigma, mu, exp)
# # broken_power2 = broken_powerlaw(x_array, amp, 0.1*sigma, mu, exp)
# # broken_power3 = broken_powerlaw(x_array, amp, 0.5*sigma, mu, exp)
#
# pseudoPower1 = pseudo_power_law(x_array, amp, mu, sigma, frac, amp, sigma, mu, -2)
# pseudoPower2 = pseudo_power_law(x_array, amp, mu, sigma, frac, amp, sigma/4, mu, -0.5)
# # gaussian1 = frac * gaussian_model(x_array, amp, mu, sigma)
# # power1 = (1 - frac) * broken_powerlaw(x_array, amp, sigma, mu, exp)
# # power2 = (1 - frac) * broken_powerlaw(x_array, amp, sigma/2, mu, exp)
#
#
# fig, ax = plt.subplots(dpi=200)
# # ax.plot(x_array, gaussian_array, label='Gaussian')
# # ax.plot(x_array, broken_power, linestyle=':', color='tab:blue', label='Law 1')
# # ax.plot(x_array, broken_power2, linestyle=':', color='tab:orange', label='Law 2')
# # ax.plot(x_array, broken_power3, linestyle=':', color='tab:green', label='Law 3')
# ax.plot(x_array, pseudoPower1, label='Pseudo', )
# ax.plot(x_array, pseudoPower2, label='Pseudo 2', linestyle=':')
# # ax.plot(x_array, gaussian1, label='Gaussian', linestyle=':')
# # ax.plot(x_array, power1, label='Power law', linestyle=':')
# # ax.plot(x_array, power2, label='Power law 2', linestyle=':')
# ax.legend()
# ax.set_ylabel('Flux')
# ax.set_xlabel('Wavelength')
# # ax.set_yscale('log')
# plt.show()