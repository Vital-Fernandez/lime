import matplotlib.pyplot as plt
import numpy as np
import lime
from lmfit.models import gaussian, lorentzian, voigt, pvoigt
from lime.model import AREA_FUNCTIONS, FWHM_FUNCTIONS, PROFILE_FUNCTIONS

# Profile parameters
amp = 125.0
mu = 4862.55
sigma = 1.55
gamma = 2.55
m_cont = 0.01
n_cont = 10
noise = 0.1
frac = 0.555
n_pixels = 500

# Arrays with the independent variables
x_array = np.linspace(mu - 75, mu + 75, n_pixels)
cont_array = m_cont * x_array + n_cont
noise_array = np.random.normal(0, noise, size=x_array.shape)
gaussian_array = gaussian(x_array, amp * np.sqrt(2*np.pi) * sigma, mu, sigma)
lorentzian_array = lorentzian(x_array, amp * np.pi * sigma, mu, sigma)
pseudo_voigt_array = frac * gaussian_array + (1 - frac) * lorentzian_array


def test_gaussian():

    y_array = gaussian_array + cont_array + noise_array
    spec = lime.Spectrum(x_array, y_array, redshift=0, norm_flux=1)
    spec.fit.bands('H1_4861A')

    assert np.allclose(spec.fit.line.amp, amp, rtol=0.01)
    assert np.allclose(spec.fit.line.center, mu, rtol=0.01)
    assert np.allclose(spec.fit.line.sigma, sigma, rtol=0.01)
    assert np.allclose(spec.fit.line.gamma, np.nan, equal_nan=True)
    assert np.allclose(spec.fit.line.frac, np.nan, equal_nan=True)

    assert np.allclose(spec.fit.line.m_cont, m_cont, rtol=0.05)
    assert np.allclose(spec.fit.line.n_cont, n_cont, rtol=spec.fit.line.n_cont_err)

    p_shape = spec.fit.line._p_shape[0]
    g_fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    g_area = amp * sigma * np.sqrt(2 * np.pi)
    assert p_shape == 'g'
    assert np.allclose(spec.fit.line.FWHM_p, g_fwhm, rtol=0.01)
    assert np.allclose(spec.fit.line.intg_flux, g_area, rtol=0.01)
    assert np.allclose(spec.fit.line.profile_flux, g_area, rtol=0.01)
    # assert np.allclose(spec.fit.line.FWHM_i, g_fwhm, rtol=0.01)   # TODO correct this calculation

    return


def test_lorentzian():

    y_array = lorentzian_array + cont_array + noise_array
    spec = lime.Spectrum(x_array, y_array, redshift=0, norm_flux=1)
    spec.fit.bands('H1_4861A_p-l', bands=np.array([4809.8, 4836.1, 4837.00, 4882.00, 4883.13, 4908.4]))

    assert np.allclose(spec.fit.line.amp, amp, rtol=0.01)
    assert np.allclose(spec.fit.line.center, mu, rtol=0.01)
    assert np.allclose(spec.fit.line.sigma, sigma, rtol=0.01)
    assert np.allclose(spec.fit.line.gamma, np.nan, equal_nan=True)
    assert np.allclose(spec.fit.line.frac, np.nan, equal_nan=True)

    # assert np.allclose(spec.fit.line.m_cont, m_cont, rtol=0.15)
    # assert np.allclose(spec.fit.line.n_cont, n_cont, rtol=0.15)

    p_shape = spec.fit.line._p_shape[0]
    l_fwhm = 2 * sigma
    l_area = np.pi * amp * sigma
    assert p_shape == 'l'
    assert np.allclose(spec.fit.line.FWHM_p, l_fwhm, rtol=0.01)
    # assert np.allclose(spec.fit.line.FWHM_i, g_fwhm, rtol=0.01)
    # assert np.allclose(spec.fit.line.intg_flux, l_area, rtol=0.01) # TODO need to check this
    assert np.allclose(spec.fit.line.profile_flux, l_area, rtol=0.01)


    return


def test_pseudo_voigt():

    y_array = pseudo_voigt_array + cont_array + noise_array
    spec = lime.Spectrum(x_array, y_array, redshift=0, norm_flux=1)
    spec.fit.bands('H1_4861A_p-pv', bands=np.array([4809.8, 4836.1, 4837.00, 4882.00, 4883.13, 4908.4]))

    assert np.allclose(spec.fit.line.amp, amp, rtol=0.01)
    assert np.allclose(spec.fit.line.center, mu, rtol=0.01)
    assert np.allclose(spec.fit.line.sigma, sigma, rtol=0.01)
    assert np.allclose(spec.fit.line.gamma, np.nan, equal_nan=True)
    assert np.allclose(spec.fit.line.frac, frac, rtol=0.05)

    assert np.allclose(spec.fit.line.m_cont, m_cont, rtol=0.10)
    # assert np.allclose(spec.fit.line.n_cont, n_cont, rtol=0.15)

    p_shape = spec.fit.line._p_shape[0]
    g_fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    l_fwhm = 2 * sigma
    pv_fwhm = 0.5346 * l_fwhm + np.sqrt(0.2166 * l_fwhm * l_fwhm + g_fwhm * g_fwhm)
    pv_area = frac * (np.sqrt(2 * np.pi) * amp * sigma) + (1 - frac) * (np.pi * amp * sigma)

    assert p_shape == 'pv'

    assert np.allclose(spec.fit.line.FWHM_p, pv_fwhm, rtol=0.01)
    # assert np.allclose(spec.fit.line.FWHM_i, pv_fwhm, rtol=0.01)
    assert np.allclose(spec.fit.line.intg_flux, pv_area, rtol=0.05)         # TODO need to check this
    assert np.allclose(spec.fit.line.profile_flux, pv_area, rtol=0.01)

    return




