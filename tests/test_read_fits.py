import pytest
import lime

from pathlib import Path
from matplotlib import pyplot as plt

# Plot tolerance
tolerance_rms = 3

# Data for the tests
baseline_folder = Path(__file__).parent / 'baseline'
outputs_folder = Path(__file__).parent / 'outputs'
spectra_folder = Path(__file__).parent.parent/'examples/sample_data/spectra'

# Fitting example for text fil
redshift_dict = {'SHOC579': 0.0475, 'Izw18': 0.00095, 'gp121903': 0.19531, 'ceers1027': 7.8189}


class TestOpenFits:

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

