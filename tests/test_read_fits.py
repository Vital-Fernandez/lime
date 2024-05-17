import pytest
import numpy as np
import lime

from pathlib import Path
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

# Plot tolerance
tolerance_rms = 3

# Data for the tests
baseline_folder = Path(__file__).parent / 'baseline'
outputs_folder = Path(__file__).parent / 'outputs'
spectra_folder = Path(__file__).parent.parent/'examples/sample_data/spectra'

# Fitting example for text fil
redshift_dict = {'SHOC579': 0.0475, 'Izw18': 0.00095, 'gp121903': 0.19531, 'ceers1027': 7.8189}


class TestOpenFits:

    def test_read_isis_params(self, file_name='IZW18_isis.fits'):

        izw18 = lime.Spectrum.from_file(spectra_folder/file_name, instrument='ISIS', redshift=redshift_dict['Izw18'],
                                        norm_flux=1e-19)

        assert izw18.redshift == redshift_dict['Izw18']
        assert izw18.units_wave == 'Angstrom'
        assert izw18.units_flux == 'FLAM'
        assert izw18.norm_flux == 1e-19

        return

    def test_read_sdss_params(self, file_name='sdss_dr18_0358-51818-0504.fits'):

        # Open with LiMe functions
        SHOC579 = lime.Spectrum.from_file(spectra_folder/file_name, instrument='sdss')
        assert np.isclose(SHOC579.redshift, 0.047232304)
        assert SHOC579.units_wave == 'Angstrom'
        assert SHOC579.units_flux.scale == 1e-17
        assert SHOC579.units_flux.bases[0] == 'erg'
        assert SHOC579.units_flux.bases[1] == 'Angstrom'
        assert SHOC579.units_flux.bases[2] == 's'
        assert SHOC579.units_flux.bases[3] == 'cm'
        assert SHOC579.norm_flux == 1

        # Open with traditional methods
        with fits.open(spectra_folder/file_name) as hdul:
            data1 = hdul[1].data
            data2 = hdul[2].data
            header = hdul[1].header

        # Recover the data
        flux_array = data1['flux']
        wave_array = np.power(10, data1['loglam'])
        units_flux = '1e-17*FLAM'
        ivar_array = data1['ivar']
        err_array = np.sqrt(1/ivar_array)
        pixel_mask = ivar_array == 0
        redshift = data2['Z'][0]

        # Manual opening
        SHOC579_manual = lime.Spectrum(wave_array, flux_array, err_array, pixel_mask=pixel_mask, units_flux=units_flux,
                                       redshift=redshift)

        # Compare both spectra
        assert SHOC579.units_wave == SHOC579_manual.units_wave
        assert SHOC579.units_flux == SHOC579_manual.units_flux
        assert SHOC579.redshift == SHOC579_manual.redshift
        assert SHOC579.norm_flux == SHOC579_manual.norm_flux
        assert np.all(SHOC579.flux == SHOC579_manual.flux)
        assert np.all(SHOC579.wave == SHOC579_manual.wave)
        assert np.all(SHOC579.flux.mask == SHOC579_manual.flux.mask)

        return

    def test_read_manga_params(self, file_name='manga-8626-12704-LOGCUBE.fits.gz'):

        SHOC579 = lime.Cube.from_file(spectra_folder/file_name, instrument='manga', redshift=redshift_dict['SHOC579'])
        assert SHOC579.redshift == redshift_dict['SHOC579']
        assert SHOC579.units_wave == 'Angstrom'
        assert SHOC579.units_flux.scale == 1e-17
        assert SHOC579.units_flux.bases[0] == 'erg'
        assert SHOC579.units_flux.bases[1] == 'Angstrom'
        assert SHOC579.units_flux.bases[2] == 's'
        assert SHOC579.units_flux.bases[3] == 'cm'
        assert SHOC579.norm_flux == 1

        # Open with traditional methods
        with fits.open(spectra_folder/file_name) as hdul:

            wave = hdul['WAVE'].data
            flux_cube = hdul['FLUX'].data
            ivar_cube = hdul['IVAR'].data

            units_flux = '1e-17*FLAM'

            ivar_cube[ivar_cube == 0] = np.nan

            err_cube = np.sqrt(1/ivar_cube)
            pixel_mask_cube = np.isnan(err_cube)
            hdr = hdul['FLUX'].header
            wcs = WCS(hdr)

        # Create the traditional fits
        SHOC579_manual = lime.Cube(wave, flux_cube, err_cube, units_flux=units_flux, pixel_mask=pixel_mask_cube, wcs=wcs,
                                   redshift=redshift_dict['SHOC579'])

        # Check both cubes
        assert SHOC579_manual.units_flux.scale == 1e-17
        assert SHOC579_manual.units_flux.bases[0] == 'erg'
        assert SHOC579.units_wave == SHOC579_manual.units_wave
        assert SHOC579.units_flux == SHOC579_manual.units_flux
        assert SHOC579.redshift == SHOC579_manual.redshift
        assert SHOC579.norm_flux == SHOC579_manual.norm_flux
        assert np.all(SHOC579.flux == SHOC579_manual.flux)
        assert np.all(SHOC579.wave == SHOC579_manual.wave)
        assert np.all(SHOC579.wcs.pixel_shape == SHOC579_manual.wcs.pixel_shape)
        assert np.all(SHOC579.wcs.pixel_scale_matrix == SHOC579_manual.wcs.pixel_scale_matrix)
        assert np.all(SHOC579.wcs.wcs == SHOC579_manual.wcs.wcs)
        assert np.all(SHOC579.flux.mask == SHOC579_manual.flux.mask)

        return

    def test_read_nirspec_params(self, file_name='hlsp_ceers_jwst_nirspec_nirspec10-001027_comb-mgrat_v0.7_x1d-masked.fits'):

        ceers1027 = lime.Spectrum.from_file(spectra_folder/file_name, instrument='nirspec', redshift=redshift_dict['ceers1027'],
                                            norm_flux=1)

        assert ceers1027.redshift == redshift_dict['ceers1027']
        assert ceers1027.units_wave == 'um'
        assert ceers1027.units_flux.scale == 1
        assert ceers1027.units_flux.bases[0] == 'MJy'
        assert ceers1027.norm_flux == 1

        return

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_read_isis(self, file_name='IZW18_isis.fits'):

        izw18 = lime.Spectrum.from_file(spectra_folder/file_name, instrument='ISIS', redshift=redshift_dict['Izw18'])

        fig = plt.figure()
        izw18.plot.spectrum(in_fig=fig, rest_frame=True)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_read_osiris(self, file_name='gp121903_osiris.fits'):

        gp121903 = lime.Spectrum.from_file(spectra_folder/file_name, instrument='Osiris', redshift=redshift_dict['gp121903'])

        fig = plt.figure()
        gp121903.plot.spectrum(in_fig=fig, rest_frame=True)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_read_sdss(self, file_name='sdss_dr18_0358-51818-0504.fits'):

        gp121903 = lime.Spectrum.from_file(spectra_folder/file_name, instrument='sdss', redshift=redshift_dict['gp121903'])

        fig = plt.figure()
        gp121903.plot.spectrum(in_fig=fig, rest_frame=True)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_read_nirspec(self, file_name='hlsp_ceers_jwst_nirspec_nirspec10-001027_comb-mgrat_v0.7_x1d-masked.fits'):

        ceers1027 = lime.Spectrum.from_file(spectra_folder/file_name, instrument='nirspec', redshift=redshift_dict['ceers1027'])

        fig = plt.figure()
        ceers1027.plot.spectrum(in_fig=fig, rest_frame=True)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_read_manga(self, file_name='manga-8626-12704-LOGCUBE.fits.gz'):

        shoc579 = lime.Cube.from_file(spectra_folder/file_name, instrument='manga', redshift=redshift_dict['SHOC579'])

        fig = plt.figure()
        shoc579.check.cube('H1_6563A', in_fig=fig, rest_frame=True)

        return fig

