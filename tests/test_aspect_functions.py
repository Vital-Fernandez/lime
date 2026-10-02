import logging
import numpy as np
import pandas as pd
import lime
import pytest
from pathlib import Path
from lime.fitting.redshift import (comp_counter, compile_Rmin_arr, compute_gaussian_ridges,
                                   compute_gaussian_ridges_orig, redshift_key_method)
from lime.fitting.lines import c_KMpS
from matplotlib import pyplot as plt

try:
    import aspect
    aspect_check = True
except ImportError:
    aspect_check = False

aspect_required = pytest.mark.skipif(not aspect_check, reason='ASPECT is not installed')


def synthetic_spectrum(z_obj, lines_df, amp_arr, emis_number, noise=0.05, sigma_pix=1.5, seed=1234):

    # Flat continuum with white noise
    rng = np.random.default_rng(seed)
    wave = np.arange(4000.0, 10500.0, 2.0)
    flux = 1 + rng.normal(0, noise, wave.size)
    pred = np.zeros(wave.size, dtype=int)

    # Add the Gaussian lines and flag their pixels as detections
    sigma = sigma_pix * (wave[1] - wave[0])
    for mu, amp in zip(lines_df.wavelength.to_numpy() * (1 + z_obj), amp_arr):
        flux += amp * np.exp(-0.5 * np.square((wave - mu) / sigma))
        pred[np.abs(wave - mu) <= 3 * sigma] = emis_number

    return wave, flux, np.full(wave.size, noise), pred


# Data for the tests
baseline_folder = Path(__file__).parent / 'baseline'
file_address = baseline_folder/'SHOC579_MANGA38-35.txt'
bands_file_address = baseline_folder/'SHOC579_MANGA38-35_bands.txt'

data_folder = Path(__file__).parent.parent/'examples/doc_notebooks/0_resources'
outputs_folder = data_folder/'results'
tolerance_rms = 5.5

# Synthetic observation
z_true = 0.523
vsigma = 70
emis_number = aspect.cfg['shape_number']['emission'] if aspect_check else 3
comps_arr = np.array([emis_number])

# bands = pd.DataFrame({'wavelength': [3727.0, 4861.0, 4959.0, 5007.0, 6563.0]},
#                      index=['O2_3727A', 'H1_4861A', 'O3_4959A', 'O3_5007A', 'H1_6563A'])
bands = lime.lines_frame(line_list=['O2_3726A', 'H1_4861A', 'O3_4959A', 'O3_5007A', 'H1_6563A'])
amp_array = np.array([8.0, 6.0, 7.0, 20.0, 18.0])

wave_array, flux_array, err_array, pred_array = synthetic_spectrum(z_true, bands, amp_array, emis_number)
delta_z_pix = np.median(np.diff(wave_array)) / np.median(wave_array)

spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=0)
spec.infer.pred_arr = pred_array

# Observed SHOC579 spectrum
redshift = 0.0475
norm_flux = 1e-17


class TestRedshiftTools:

    def test_compile_Rmin_arr(self):

        # No constrains
        assert compile_Rmin_arr(bands) is None
        assert compile_Rmin_arr(bands, None) is None

        # Lines without an entry have no requirement
        Rmin_arr = compile_Rmin_arr(bands, {1000: ['O3_4959A', 'O3_5007A'], 300: ['H1_6563A']})
        assert np.array_equal(Rmin_arr, np.array([0, 0, 1000, 1000, 300]))

        # Repeated lines keep the strictest value (independently of the order)
        Rmin_arr = compile_Rmin_arr(bands, {1000: ['O3_5007A'], 300: ['O3_5007A', 'H1_6563A']})
        assert np.array_equal(Rmin_arr, np.array([0, 0, 0, 1000, 300]))

        Rmin_arr = compile_Rmin_arr(bands, {300: ['O3_5007A', 'H1_6563A'], 1000: ['O3_5007A']})
        assert np.array_equal(Rmin_arr, np.array([0, 0, 0, 1000, 300]))

        # Lines missing from the bands table
        with pytest.raises(ValueError, match='not found in bands table'):
            compile_Rmin_arr(bands, {1000: ['O3_5007A', 'He2_4686A']})

        return

    def test_gaussian_ridges(self):

        theo_lambda = bands.wavelength.to_numpy()
        wave_matrix = np.tile(wave_array, (theo_lambda.size, 1))
        res_power = np.full(wave_array.size, 2000.0)

        gauss_arr = compute_gaussian_ridges(z_true, theo_lambda, wave_matrix, 1, vsigma, res_power)

        # Normalized key with the spectrum shape
        assert gauss_arr.shape == wave_array.shape
        assert gauss_arr.min() >= 0
        assert np.isclose(gauss_arr.max(), 1)

        # Ridges at the lines observed wavelengths and none away from them
        idcs_obs = np.searchsorted(wave_array, theo_lambda * (1 + z_true))
        assert np.allclose(gauss_arr[idcs_obs], 1)
        assert np.all(gauss_arr[~np.isin(pred_array, comps_arr)] < 0.5)
        assert comp_counter(gauss_arr > 0.5) == theo_lambda.size

        # Ridges width (isolated line): velocity dispersion plus instrumental broadening
        mu = wave_array[idcs_obs[0]]
        sigma = mu * (vsigma / c_KMpS) + mu / (2000.0 * 2 * np.sqrt(2 * np.log(2)))
        idx_wing = idcs_obs[0] + 2
        assert np.isclose(gauss_arr[idx_wing], np.exp(-0.5 * np.square((wave_array[idx_wing] - mu) / sigma)))

        # Same output as the original function without resolution constrains
        gauss_orig = compute_gaussian_ridges_orig(z_true, theo_lambda, wave_matrix, 1, vsigma, res_power)
        assert np.allclose(gauss_arr, gauss_orig)

        # Fewer than two lines in the wavelength range
        assert compute_gaussian_ridges(1.6, theo_lambda, wave_matrix, 1, vsigma, res_power) is None  # Only O2_3727A
        assert compute_gaussian_ridges(5.0, theo_lambda, wave_matrix, 1, vsigma, res_power) is None  # No lines

        return

    def test_gaussian_ridges_min_R(self):

        theo_lambda = bands.wavelength.to_numpy()
        wave_matrix = np.tile(wave_array, (theo_lambda.size, 1))
        res_power = np.full(wave_array.size, 2000.0)
        idcs_obs = np.searchsorted(wave_array, theo_lambda * (1 + z_true))

        # Requirements below the observation resolving power do not change the key
        gauss_ref = compute_gaussian_ridges(z_true, theo_lambda, wave_matrix, 1, vsigma, res_power)
        Rmin_arr = compile_Rmin_arr(bands, {1000: ['O3_4959A', 'O3_5007A']})
        gauss_arr = compute_gaussian_ridges(z_true, theo_lambda, wave_matrix, 1, vsigma, res_power, Rmin_arr=Rmin_arr)
        assert np.allclose(gauss_arr, gauss_ref)

        # Requirement above it removes the line from the key
        Rmin_arr = compile_Rmin_arr(bands, {5000: ['H1_6563A']})
        gauss_arr = compute_gaussian_ridges(z_true, theo_lambda, wave_matrix, 1, vsigma, res_power, Rmin_arr=Rmin_arr)
        assert np.isclose(gauss_arr[idcs_obs[-1]], 0)
        assert np.allclose(gauss_arr[idcs_obs[:-1]], 1)
        assert comp_counter(gauss_arr > 0.5) == theo_lambda.size - 1

        # The requirements follow the lines when some of them are outside the wavelength range
        z_i = 0.3  # H1_6563A observed, O2_3727A outside
        Rmin_arr = compile_Rmin_arr(bands, {5000: ['H1_6563A']})
        gauss_arr = compute_gaussian_ridges(z_i, theo_lambda, wave_matrix, 1, vsigma, res_power, Rmin_arr=Rmin_arr)
        idcs_z = np.searchsorted(wave_array, theo_lambda[1:] * (1 + z_i))
        assert np.isclose(gauss_arr[idcs_z[-1]], 0)
        assert np.allclose(gauss_arr[idcs_z[:-1]], 1)

        # Fewer than two lines fulfilling the requirements
        Rmin_arr = compile_Rmin_arr(bands, {5000: list(bands.index[1:])})
        assert compute_gaussian_ridges(z_true, theo_lambda, wave_matrix, 1, vsigma, res_power, Rmin_arr=Rmin_arr) is None

        return


class TestRedshiftKeyMethod:

    def test_key_method(self):

        z_flux, z_pixel = redshift_key_method(spec, bands, 0, 2, None, pred_array, comps_arr,
                                              vsigma, method='key', sig_digits=4)

        assert np.isclose(z_flux, z_true, atol=2 * delta_z_pix)
        assert np.isclose(z_pixel, z_true, atol=2 * delta_z_pix)

        return

    def test_key_method_delta_z(self):

        z_flux, z_pixel = redshift_key_method(spec, bands, 0.4, 0.6, 0.001,
                                              pred_array, comps_arr, vsigma, method='key',
                                              sig_digits=3)

        assert np.isclose(z_flux, z_true, atol=0.001)
        assert np.isclose(z_pixel, z_true, atol=0.001)

        return

    def test_key_method_sig_digits(self):

        for sig_digits in (1, 2, 3):
            z_flux, z_pixel = redshift_key_method(spec, bands, 0, 2, None, pred_array, comps_arr, vsigma, 'key',
                                                  sig_digits=sig_digits)

            assert np.isclose(z_flux, np.round(z_true, sig_digits))
            assert np.isclose(z_pixel, np.round(z_true, sig_digits))

        return

    def test_key_method_no_detection(self, caplog):

        # No components array
        with caplog.at_level(logging.WARNING, logger='LiMe'):
            output = redshift_key_method(spec, bands, 0, 2, None, None, comps_arr, vsigma, 'key')
        assert output == (None, None)
        assert 'does not have a components array' in caplog.text

        # No components reference
        assert redshift_key_method(spec, bands, 0, 2, None, pred_array, None, vsigma, 'key') == (None, None)

        # No lines in the components array
        pred_empty = np.zeros(wave_array.size, dtype=int)
        assert redshift_key_method(spec, bands, 0, 2, None, pred_empty, comps_arr, vsigma, 'key') == (None, None)

        # Components array without the requested shapes
        assert redshift_key_method(spec, bands, 0, 2, None, pred_array, comps_arr + 1, vsigma, 'key') == (None, None)

        return

    def test_key_method_single_line(self):

        # Only one detected line
        idx_line = np.searchsorted(wave_array, 5007.0 * (1 + z_true))
        pred_single = np.zeros(wave_array.size, dtype=int)
        pred_single[idx_line - 3:idx_line + 4] = emis_number

        z_flux, z_pixel = redshift_key_method(spec, bands, 0, 2, None, pred_single, comps_arr, vsigma, 'key')

        assert np.isnan(z_flux)
        assert np.isnan(z_pixel)

        return

    def test_key_method_pixel_mask(self):

        # Masking one of the lines should not change the measurement
        pixel_mask = np.abs(wave_array - 4861.0 * (1 + z_true)) < 20
        spec_mask = lime.Spectrum(wave_array, flux_array, err_array, redshift=0, pixel_mask=pixel_mask)

        z_flux, z_pixel = redshift_key_method(spec_mask, bands, 0, 2, None, pred_array, comps_arr, vsigma, 'key',
                                              sig_digits=3)

        assert np.isclose(z_flux, z_true, atol=0.001)
        assert np.isclose(z_pixel, z_true, atol=0.001)

        return

    def test_key_method_min_R(self):

        # Requirement fulfilled by the observation
        z_flux, z_pixel = redshift_key_method(spec, bands, 0, 2, None, pred_array, comps_arr, vsigma, 'key',
                                              map_band_R={10: ['O3_4959A', 'O3_5007A']}, sig_digits=3)
        assert np.isclose(z_flux, z_true, atol=0.001)
        assert np.isclose(z_pixel, z_true, atol=0.001)

        # Lines excluded from the key (remaining ones still constrain the redshift)
        z_flux, z_pixel = redshift_key_method(spec, bands, 0, 2, None, pred_array, comps_arr, vsigma, 'key',
                                              map_band_R={1e7: ['O2_3726A', 'H1_4861A']}, sig_digits=3)
        assert np.isclose(z_flux, z_true, atol=0.001)
        assert np.isclose(z_pixel, z_true, atol=0.001)

        # Lines missing from the bands table
        with pytest.raises(ValueError):
            redshift_key_method(spec, bands, 0, 2, None, pred_array, comps_arr, vsigma, 'key',
                                map_band_R={1000: ['He2_4686A']})

        return


@aspect_required
class TestRedshiftFit:

    def test_redshift(self):

        z_flux, z_pixel = spec.fit.redshift(bands, z_max=2, comps_list=['emission'], sig_digits=3)

        assert np.isclose(z_flux, z_true, atol=0.001)
        assert np.isclose(z_pixel, z_true, atol=0.001)

        return

    def test_redshift_default_params(self):

        z_flux, z_pixel = spec.fit.redshift(bands, z_max=2)

        assert np.isclose(z_flux, 0.52)
        assert np.isclose(z_pixel, 0.52)

        return

    def test_redshift_range(self):

        # Limits are included in the search
        z_flux, z_pixel = spec.fit.redshift(bands, z_min=0.4, z_max=z_true, delta_z=0.001, sig_digits=3)

        assert np.isclose(z_flux, z_true, atol=0.001)
        assert np.isclose(z_pixel, z_true, atol=0.001)

        return

    def test_redshift_components(self):

        # Detection labelled with a different shape
        doublet_number = aspect.cfg['shape_number']['doublet-em']
        spec0 = lime.Spectrum(wave_array, flux_array, err_array, redshift=0)
        spec0.infer.pred_arr = np.where(pred_array == emis_number, doublet_number, pred_array)

        assert spec0.fit.redshift(bands, z_max=2, comps_list=['emission']) == (None, None)

        z_flux, z_pixel = spec0.fit.redshift(bands, z_max=2, comps_list=['doublet-em'], sig_digits=3)
        assert np.isclose(z_flux, z_true, atol=0.001)
        assert np.isclose(z_pixel, z_true, atol=0.001)

        z_flux, z_pixel = spec0.fit.redshift(bands, z_max=2, sig_digits=3)
        assert np.isclose(z_flux, z_true, atol=0.001)
        assert np.isclose(z_pixel, z_true, atol=0.001)

        return

    def test_redshift_min_R(self):

        z_flux, z_pixel = spec.fit.redshift(bands, z_max=2, map_min_R={10: ['O3_4959A', 'O3_5007A']}, sig_digits=3)

        assert np.isclose(z_flux, z_true, atol=0.001)
        assert np.isclose(z_pixel, z_true, atol=0.001)

        with pytest.raises(ValueError):
            spec.fit.redshift(bands, z_max=2, map_min_R={1000: ['He2_4686A']})

        return

    def test_redshift_no_detection(self, caplog):

        spec0 = lime.Spectrum(wave_array, flux_array, err_array, redshift=0)

        with caplog.at_level(logging.WARNING, logger='LiMe'):
            output = spec0.fit.redshift(bands, z_max=2)

        assert output == (None, None)
        assert 'please run ASPECT' in caplog.text

        return

    def test_redshift_unknown_mode(self):

        with pytest.raises(KeyError):
            spec.fit.redshift(bands, z_max=2, mode='xor')

        return

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_redshift_plot(self, monkeypatch):
        plt.close('all')
        monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
        spec.fit.redshift(bands, z_max=2, sig_digits=3, plot_results=True)

        return plt.gcf()

    def test_redshift_observation(self):

        wave_obs, flux_obs, err_obs, mask_obs = np.loadtxt(file_address, unpack=True)
        spec_obs = lime.Spectrum(wave_obs, flux_obs, err_obs, redshift=redshift, norm_flux=norm_flux,
                                 pixel_mask=mask_obs, id_label='SHOC579_Manga38-35')

        # Components detection
        spec_obs.infer.components()

        bands_obs = lime.load_frame(bands_file_address)
        z_flux, z_pixel = spec_obs.fit.redshift(bands_obs, z_max=0.5, sig_digits=3)

        assert np.isclose(z_flux, redshift, atol=0.002)
        assert np.isclose(z_pixel, redshift, atol=0.002)

        return