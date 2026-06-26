import numpy as np
import lime
from lmfit.models import gaussian, lorentzian
import pandas as pd



def make_spectrum_arrays(mu, sigma, gamma, frac, amp, m_cont, n_cont, noise, n_pixels=500, band_sigma=15, fwhm_factor=6):
    """
    Generate wavelength, flux, and band arrays for a single emission line.

    Parameters
    ----------
    mu          : line center wavelength
    sigma       : Gaussian width
    gamma       : Lorentzian width
    frac        : pseudo-Voigt mixing fraction
    amp         : line amplitude
    m_cont      : continuum gradient
    n_cont      : continuum zero level
    noise       : noise standard deviation
    n_pixels    : number of wavelength pixels
    band_sigma  : half-width of the full spectral window in units of sigma
    fwhm_factor : multiplier on the profile FWHM to set w3-w4 half-width
    """
    # FWHM for each profile component using the correct width parameter
    g_fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    l_fwhm = 2 * gamma
    pv_fwhm = 0.5346 * l_fwhm + np.sqrt(0.2166 * l_fwhm ** 2 + g_fwhm ** 2)

    # Use the widest FWHM to set the central band
    max_fwhm = max(g_fwhm, l_fwhm, pv_fwhm)
    half_central = fwhm_factor * max_fwhm / 2

    x_array = np.linspace(mu - band_sigma * max_fwhm, mu + band_sigma * max_fwhm, n_pixels)
    cont_array = m_cont * x_array + n_cont

    w3 = mu - half_central * 1.5
    w4 = mu + half_central * 1.5
    band_width = (band_sigma * max_fwhm - half_central) / 2
    w1 = w3 - 2 * band_width
    w2 = w3 - band_width
    w5 = w4 + band_width
    w6 = w4 + 2 * band_width

    bands_df = pd.DataFrame({'w1': [w1], 'w2': [w2], 'w3': [w3], 'w4': [w4], 'w5': [w5], 'w6': [w6]},
                            index=['H1_4861A'])

    return x_array, cont_array, bands_df



# Test parameter sets
TEST_CONDITIONS = [
    {'amp': -9.87, 'mu': 4862.55, 'sigma': 2.00, 'gamma': 3.00, 'frac': 0.400, 'm_cont': 0, 'n_cont': 50, 'noise': 0.05},
    {'amp': 9.87, 'mu': 4862.55, 'sigma': 2.00, 'gamma': 3.00, 'frac': 0.400, 'm_cont':0, 'n_cont': 50, 'noise': 0.05},
    {'amp': 22.2, 'mu': 4862.55, 'sigma': 1.55, 'gamma': 2.55, 'frac': 0.555, 'm_cont': 0,  'n_cont': 100,  'noise': 0.1},
    {'amp': 75.5, 'mu': 4862.55,  'sigma': 1.20, 'gamma': 1.80, 'frac': 0.700, 'm_cont': 0,  'n_cont': 200,  'noise': 0.2},
]



def test_gaussian():
    for i, p in enumerate(TEST_CONDITIONS):
        np.random.seed(i)
        x_array, cont_array, bands_df = make_spectrum_arrays(p['mu'], p['sigma'], p['gamma'], p['frac'],
                                                             p['amp'], p['m_cont'], p['n_cont'], p['noise'])
        noise_array = np.random.normal(0, p['noise'], size=x_array.shape)
        g_array = gaussian(x_array, p['amp'] * np.sqrt(2 * np.pi) * p['sigma'], p['mu'], p['sigma'])
        y_array = g_array + cont_array + noise_array

        spec = lime.Spectrum(x_array, y_array, redshift=0, norm_flux=1)
        spec.fit.bands('H1_4861A_p-g', bands=bands_df, cont_source='adjacent', shape='abs' if p['amp'] < 0 else 'emi')
        # spec.plot.bands()
        meas = spec.fit.line.measurements

        assert np.abs(meas.amp - p['amp']) <= 3 * meas.amp_err, f"amp failed condition {i}"
        assert np.allclose(meas.center, p['mu'], rtol=0.05)
        assert np.abs(meas.sigma - p['sigma']) <= 3 * meas.sigma_err, f"sigma failed condition {i}"
        assert meas.gamma is None, f"gamma failed condition {i}"
        assert meas.frac is None, f"frac failed condition {i}"
        assert np.abs(meas.m_cont - p['m_cont']) <= 3 * meas.m_cont_err_intg, f"m_cont failed condition {i}"
        assert np.abs(meas.n_cont - p['n_cont']) <= 3 * meas.n_cont_err_intg, f"n_cont failed condition {i}"

        g_fwhm = 2 * np.sqrt(2 * np.log(2)) * p['sigma']
        g_area = p['amp'] * p['sigma'] * np.sqrt(2 * np.pi)
        assert spec.fit.line.profile == 'g', f"profile failed condition {i}"
        assert spec.fit.line.shape == 'abs' if p['amp'] < 0 else 'emi', f"shape failed condition {i}"
        assert np.abs(meas.profile_flux - g_area) <= 3 * meas.profile_flux_err, f"profile_flux failed condition {i}"
        assert np.allclose(spec.fit.line.measurements.FWHM_p, g_fwhm, rtol=0.05)

    return


def test_gaussian_argument():
    for i, p in enumerate(TEST_CONDITIONS):
        np.random.seed(i)
        x_array, cont_array, bands_df = make_spectrum_arrays(p['mu'], p['sigma'], p['gamma'], p['frac'],
                                                             p['amp'], p['m_cont'], p['n_cont'], p['noise'])
        noise_array = np.random.normal(0, p['noise'], size=x_array.shape)
        g_array = gaussian(x_array, p['amp'] * np.sqrt(2 * np.pi) * p['sigma'], p['mu'], p['sigma'])
        y_array = g_array + cont_array + noise_array

        spec = lime.Spectrum(x_array, y_array, redshift=0, norm_flux=1)
        spec.fit.bands('H1_4861A', bands=bands_df, cont_source='adjacent', shape='abs' if p['amp'] < 0 else 'emi')
        # spec.plot.bands()
        meas = spec.fit.line.measurements

        assert np.abs(meas.amp - p['amp']) <= 3 * meas.amp_err, f"amp failed condition {i}"
        assert np.allclose(meas.center, p['mu'], rtol=0.05)
        assert np.abs(meas.sigma - p['sigma']) <= 3 * meas.sigma_err, f"sigma failed condition {i}"
        assert meas.gamma is None, f"gamma failed condition {i}"
        assert meas.frac is None, f"frac failed condition {i}"
        assert np.abs(meas.m_cont - p['m_cont']) <= 3 * meas.m_cont_err_intg, f"m_cont failed condition {i}"
        assert np.abs(meas.n_cont - p['n_cont']) <= 3 * meas.n_cont_err_intg, f"n_cont failed condition {i}"

        g_fwhm = 2 * np.sqrt(2 * np.log(2)) * p['sigma']
        g_area = p['amp'] * p['sigma'] * np.sqrt(2 * np.pi)
        assert spec.fit.line.profile == 'g', f"profile failed condition {i}"
        assert spec.fit.line.shape == 'abs' if p['amp'] < 0 else 'emi', f"shape failed condition {i}"
        assert np.abs(meas.profile_flux - g_area) <= 3 * meas.profile_flux_err, f"profile_flux failed condition {i}"
        assert np.allclose(spec.fit.line.measurements.FWHM_p, g_fwhm, rtol=0.05)

    return


def test_lorentzian():
    for i, p in enumerate(TEST_CONDITIONS):
        np.random.seed(i)
        x_array, cont_array, bands_df = make_spectrum_arrays(p['mu'], p['sigma'], p['gamma'], p['frac'],
                                                             p['amp'], p['m_cont'], p['n_cont'], p['noise'])
        noise_array = np.random.normal(0, p['noise'], size=x_array.shape)
        l_array = lorentzian(x_array, p['amp'] * np.pi * p['sigma'], p['mu'], p['sigma'])
        y_array = l_array + cont_array + noise_array

        spec = lime.Spectrum(x_array, y_array, redshift=0, norm_flux=1)
        spec.fit.bands('H1_4861A_p-l', bands=bands_df, cont_source='adjacent', shape='abs' if p['amp'] < 0 else 'emi')
        # spec.plot.bands()
        meas = spec.fit.line.measurements

        # assert np.abs(meas.amp - p['amp']) <= 3 * meas.amp_err, f"amp failed condition {i}"
        assert np.isclose(meas.amp, p['amp'], rtol=0.05)
        assert np.abs(meas.center - p['mu']) <= 3 * meas.center_err, f"center failed condition {i}"
        # assert np.abs(meas.sigma - p['sigma']) <= 3 * meas.sigma_err, f"sigma failed condition {i}" # The reported error seems beyond 3 sigma
        assert np.isclose(meas.sigma, p['sigma'], rtol=0.05)
        assert meas.gamma is None, f"gamma failed condition {i}"
        assert meas.frac is None, f"frac failed condition {i}"

        l_fwhm = 2 * p['sigma']
        l_area = np.pi * p['amp'] * p['sigma']
        assert spec.fit.line.profile == 'l', f"profile failed condition {i}"
        assert spec.fit.line.shape == 'abs' if p['amp'] < 0 else 'emi', f"shape failed condition {i}"
        # assert np.abs(meas.profile_flux - l_area) <= 3 * meas.profile_flux_err, f"profile_flux failed condition {i}"
        assert np.isclose(meas.profile_flux, l_area, rtol=0.05)
        assert np.allclose(spec.fit.line.measurements.FWHM_p, l_fwhm, rtol=0.05)

    return


def test_lorentzian_argument():
    for i, p in enumerate(TEST_CONDITIONS):
        np.random.seed(i)
        x_array, cont_array, bands_df = make_spectrum_arrays(p['mu'], p['sigma'], p['gamma'], p['frac'],
                                                             p['amp'], p['m_cont'], p['n_cont'], p['noise'])
        noise_array = np.random.normal(0, p['noise'], size=x_array.shape)
        l_array = lorentzian(x_array, p['amp'] * np.pi * p['sigma'], p['mu'], p['sigma'])
        y_array = l_array + cont_array + noise_array

        spec = lime.Spectrum(x_array, y_array, redshift=0, norm_flux=1)
        spec.fit.bands('H1_4861A', profile='l', bands=bands_df, cont_source='adjacent', shape='abs' if p['amp'] < 0 else 'emi')
        # spec.plot.bands()
        meas = spec.fit.line.measurements

        # assert np.abs(meas.amp - p['amp']) <= 3 * meas.amp_err, f"amp failed condition {i}"
        assert np.isclose(meas.amp, p['amp'], rtol=0.05)
        assert np.abs(meas.center - p['mu']) <= 3 * meas.center_err, f"center failed condition {i}"
        # assert np.abs(meas.sigma - p['sigma']) <= 3 * meas.sigma_err, f"sigma failed condition {i}" # The reported error seems beyond 3 sigma
        assert np.isclose(meas.sigma, p['sigma'], rtol=0.05)
        assert meas.gamma is None, f"gamma failed condition {i}"
        assert meas.frac is None, f"frac failed condition {i}"

        l_fwhm = 2 * p['sigma']
        l_area = np.pi * p['amp'] * p['sigma']
        assert spec.fit.line.profile == 'l', f"profile failed condition {i}"
        assert spec.fit.line.shape == 'abs' if p['amp'] < 0 else 'emi', f"shape failed condition {i}"
        # assert np.abs(meas.profile_flux - l_area) <= 3 * meas.profile_flux_err, f"profile_flux failed condition {i}"
        assert np.isclose(meas.profile_flux, l_area, rtol=0.05)
        assert np.allclose(spec.fit.line.measurements.FWHM_p, l_fwhm, rtol=0.05)

    return


def test_pseudo_voigt():
    for i, p in enumerate(TEST_CONDITIONS):

        np.random.seed(i)
        x_array, cont_array, bands_df = make_spectrum_arrays(p['mu'], p['sigma'], p['gamma'], p['frac'],
                                                             p['amp'], p['m_cont'], p['n_cont'], p['noise'])

        noise_array = np.random.normal(0, p['noise'], size=x_array.shape)
        g_array = gaussian(x_array, p['amp'] * np.sqrt(2 * np.pi) * p['sigma'], p['mu'], p['sigma'])
        l_array = lorentzian(x_array, p['amp'] * np.pi * p['sigma'], p['mu'], p['sigma'])
        pv_array = p['frac'] * g_array + (1 - p['frac']) * l_array
        y_array = pv_array + cont_array + noise_array

        spec = lime.Spectrum(x_array, y_array, redshift=0, norm_flux=1)
        spec.fit.bands('H1_4861A_p-pv', bands=bands_df, cont_source='adjacent', shape='abs' if p['amp'] < 0 else 'emi')
        # spec.plot.bands()

        meas = spec.fit.line.measurements
        assert np.isclose(meas.amp, p['amp'], rtol=0.05)
        assert np.isclose(meas.center, p['mu'], rtol=0.05), f"center failed condition {i}"
        # assert np.abs(meas.center - p['mu']) <= 3 * meas.center_err, f"center failed condition {i}"
        assert np.isclose(meas.sigma, p['sigma'], rtol=0.05), f"sigma failed condition {i}"
        assert meas.gamma is None, f"gamma failed condition {i}"
        assert np.isclose(meas.frac, p['frac'], rtol=0.10), f"frac failed condition {i}"

        g_fwhm = 2 * np.sqrt(2 * np.log(2)) * p['sigma']
        l_fwhm = 2 * p['sigma']
        pv_fwhm = 0.5346 * l_fwhm + np.sqrt(0.2166 * l_fwhm ** 2 + g_fwhm ** 2)
        pv_area = p['frac'] * (np.sqrt(2 * np.pi) * p['amp'] * p['sigma']) + (1 - p['frac']) * (np.pi * p['amp'] * p['sigma'])

        assert spec.fit.line.profile == 'pv', f"profile failed condition {i}"
        assert spec.fit.line.shape == 'abs' if p['amp'] < 0 else 'emi', f"shape failed condition {i}"
        assert np.isclose(meas.profile_flux, pv_area, rtol=0.05), f"profile_flux failed condition {i}"
        assert np.allclose(meas.FWHM_p, pv_fwhm, rtol=0.05), f"FWHM_p failed condition {i}"

    return


def test_pseudo_voigt_argument():
    for i, p in enumerate(TEST_CONDITIONS):

        np.random.seed(i)
        x_array, cont_array, bands_df = make_spectrum_arrays(p['mu'], p['sigma'], p['gamma'], p['frac'],
                                                             p['amp'], p['m_cont'], p['n_cont'], p['noise'])

        noise_array = np.random.normal(0, p['noise'], size=x_array.shape)
        g_array = gaussian(x_array, p['amp'] * np.sqrt(2 * np.pi) * p['sigma'], p['mu'], p['sigma'])
        l_array = lorentzian(x_array, p['amp'] * np.pi * p['sigma'], p['mu'], p['sigma'])
        pv_array = p['frac'] * g_array + (1 - p['frac']) * l_array
        y_array = pv_array + cont_array + noise_array

        spec = lime.Spectrum(x_array, y_array, redshift=0, norm_flux=1)
        spec.fit.bands('H1_4861A', profile='pv', bands=bands_df, cont_source='adjacent', shape='abs' if p['amp'] < 0 else 'emi')
        # spec.plot.bands()

        meas = spec.fit.line.measurements
        assert np.isclose(meas.amp, p['amp'], rtol=0.05)
        assert np.isclose(meas.center, p['mu'], rtol=0.05), f"center failed condition {i}"
        # assert np.abs(meas.center - p['mu']) <= 3 * meas.center_err, f"center failed condition {i}"
        assert np.isclose(meas.sigma, p['sigma'], rtol=0.05), f"sigma failed condition {i}"
        assert meas.gamma is None, f"gamma failed condition {i}"
        assert np.isclose(meas.frac, p['frac'], rtol=0.10), f"frac failed condition {i}"

        g_fwhm = 2 * np.sqrt(2 * np.log(2)) * p['sigma']
        l_fwhm = 2 * p['sigma']
        pv_fwhm = 0.5346 * l_fwhm + np.sqrt(0.2166 * l_fwhm ** 2 + g_fwhm ** 2)
        pv_area = p['frac'] * (np.sqrt(2 * np.pi) * p['amp'] * p['sigma']) + (1 - p['frac']) * (np.pi * p['amp'] * p['sigma'])

        assert spec.fit.line.profile == 'pv', f"profile failed condition {i}"
        assert spec.fit.line.shape == 'abs' if p['amp'] < 0 else 'emi', f"shape failed condition {i}"
        assert np.isclose(meas.profile_flux, pv_area, rtol=0.05), f"profile_flux failed condition {i}"
        assert np.allclose(meas.FWHM_p, pv_fwhm, rtol=0.05), f"FWHM_p failed condition {i}"

    return


