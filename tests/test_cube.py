import numpy as np
import lime
from matplotlib import pyplot as plt
from pathlib import Path
import urllib.request
from astropy.wcs import WCS
from astropy.io import fits
import pytest


baseline_folder = Path(__file__).parent / 'baseline'
outputs_folder = Path(__file__).parent / 'outputs'

file_address = baseline_folder/'manga_spaxel.txt'
conf_file_address = baseline_folder/'manga.toml'
bands_file_address = baseline_folder/f'manga_line_bands.txt'
lines_log_address = baseline_folder/'manga_lines_log.txt'
spatial_mask_address = baseline_folder/'SHOC579_mask.fits'
cube_log_address = baseline_folder/'SHOC579_log.fits'

# cube_address = Path(__file__).parent/'outputs'/'manga-8626-12704-LOGCUBE.fits.gz'
cube_address = Path(__file__).parent.parent/'examples/sample_data/spectra/manga-8626-12704-LOGCUBE.fits.gz'
spatial_log_address = Path(__file__).parent/'outputs'/'SHOC579_log.fits'

RMS_tolerance = 3
redshift = 0.0475
norm_flux = 1e-17
spaxel_label = '38-35'
cfg = lime.load_cfg(conf_file_address)

# Spaxel data
wave_array, flux_array, err_array = np.loadtxt(file_address, unpack=True)
pixel_mask = np.isnan(err_array)

spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                     pixel_mask=pixel_mask)

spec.fit.frame(bands_file_address, cfg, id_conf_prefix=spaxel_label)

# MANGA cube web link and save file location
cube_url = 'https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/8626/stack/manga-8626-12704-LOGCUBE.fits.gz'

# Download the cube file if not available (this may take some time)
if cube_address.is_file() is not True:
    urllib.request.urlretrieve(cube_url, cube_address)
    print(' Download completed!')
else:
    print('Observation found in folder')

# Open the MANGA cube fits file
with fits.open(cube_address) as hdul:
    wave = hdul['WAVE'].data
    flux_cube = hdul['FLUX'].data * norm_flux
    hdr = hdul['FLUX'].header

wcs = WCS(hdr)

# Define a LiMe cube object
cube = lime.Cube(wave, flux_cube, redshift=redshift, norm_flux=norm_flux, wcs=wcs)


class TestCubeClass:

    def test_read_cube(self):

        assert cube.norm_flux == norm_flux
        assert cube.redshift == redshift
        assert np.allclose(wave_array, cube.wave.data)
        assert np.allclose(wave_array / (1 + redshift), cube.wave_rest.data)

        return

    def test_spatial_masking(self):

        cube.spatial_masking('O3_4363A', param='SN_line', contour_pctls=[93, 96, 99],
                             output_address=spatial_mask_address)

        mask_dict = lime.load_spatial_mask(spatial_mask_address)

        assert len(mask_dict) == 3
        assert mask_dict['MASK_0'][0].sum() == mask_dict['MASK_0'][1]['NUMSPAXE']
        assert mask_dict['MASK_1'][0].sum() == mask_dict['MASK_1'][1]['NUMSPAXE']
        assert mask_dict['MASK_2'][0].sum() == mask_dict['MASK_2'][1]['NUMSPAXE']

        assert mask_dict['MASK_0'][1]['PARAM'] == 'SN_line'
        assert mask_dict['MASK_0'][1]['PARAMIDX'] == 99
        assert np.isclose(mask_dict['MASK_0'][1]['PARAMVAL'], 266.62467)

        assert mask_dict['MASK_1'][1]['PARAM'] == 'SN_line'
        assert mask_dict['MASK_1'][1]['PARAMIDX'] == 96
        assert np.isclose(mask_dict['MASK_1'][1]['PARAMVAL'], 49.29912)

        assert mask_dict['MASK_2'][1]['PARAM'] == 'SN_line'
        assert mask_dict['MASK_2'][1]['PARAMIDX'] == 93
        assert np.isclose(mask_dict['MASK_2'][1]['PARAMVAL'], 13.97339)

        return

    def test_get_spaxel(self):

        idx_j, idx_x = spaxel_label.split('-')
        spax = cube.get_spectrum(int(idx_j), int(idx_x))

        assert np.allclose(spax.wave, spec.wave.data)
        assert np.allclose(spax.flux, spec.flux.data)
        assert np.allclose(spax.redshift, spec.redshift)
        assert np.allclose(spax.norm_flux, spec.norm_flux)

        return

    def test_fit_spatial_mask(self):

        cfg['MASK_0_line_fitting']['bands'] = bands_file_address.as_posix()
        cube.fit.spatial_mask(spatial_mask_address, fit_conf=cfg, line_detection=True, mask_list=['MASK_0'],
                              output_address=spatial_log_address)

        assert spatial_log_address.is_file()

        spax_log = lime.load_frame(spatial_log_address, page=f'{spaxel_label}_LINELOG')
        orig_log = lime.load_frame(lines_log_address)

        # Test 3 lines # TODO review these fluxes
        assert np.sum(spax_log.index.isin(['O3_5007A', 'O3_5007A_k-1', 'He1_5016A'])) == 3
        assert np.isclose(spax_log.loc['O3_5007A', 'profile_flux'], orig_log.loc['O3_5007A', 'profile_flux'])
        assert np.isclose(spax_log.loc['O3_5007A_k-1', 'profile_flux'], orig_log.loc['O3_5007A_k-1', 'profile_flux'])
        assert np.isclose(spax_log.loc['He1_5016A', 'profile_flux'], orig_log.loc['He1_5016A', 'profile_flux'])

        return

    @pytest.mark.mpl_image_compare(baseline_dir='baseline')
    def test_plot_cube(self):

        fig = plt.figure()
        cube.plot.cube('H1_6563A', line_fg=4363, in_fig=fig)

        return fig

    @pytest.mark.mpl_image_compare(baseline_dir='baseline')
    def test_plot_cube_mask(self):

        fig = plt.figure()
        cube.plot.cube('H1_6563A', masks_file=spatial_mask_address, in_fig=fig)

        return fig

    @pytest.mark.mpl_image_compare(baseline_dir='baseline')
    def test_check_cube(self):

        fig = plt.figure()
        cube.check.cube('H1_6563A', lines_file=spatial_log_address, masks_file=spatial_mask_address, in_fig=fig)

        return fig

    def test_save_paramter_maps(self):

        ouput_folder = Path(__file__).parent/'outputs'

        # Export the measurements log as maps:
        param_list = ['intg_flux', 'intg_flux_err', 'v_r', 'v_r_err']
        lines_list = ['H1_4861A', 'H1_6563A', 'O3_4363A', 'O3_4959A', 'O3_5007A']

        lime.save_parameter_maps(spatial_log_address, ouput_folder, param_list, lines_list,
                                 mask_file=spatial_mask_address, output_file_prefix='SHOC579_', wcs=wcs)

        param_list = ['profile_flux', 'profile_flux_err']
        lines_list = ['H1_4861A', 'H1_6563A', 'O3_4363A', 'O3_4959A', 'O3_5007A']
        lime.save_parameter_maps(spatial_log_address, ouput_folder, param_list, lines_list,
                                 mask_file=spatial_mask_address, wcs=wcs)

        intg_flux_file = ouput_folder/f'SHOC579_intg_flux.fits'
        gauss_flux_file = ouput_folder/f'profile_flux.fits'

        assert intg_flux_file.is_file()
        assert gauss_flux_file.is_file()

        int_flux_map = fits.getdata(intg_flux_file, extname='O3_5007A')
        gauss_flux_map = fits.getdata(gauss_flux_file, extname='O3_5007A')

        assert np.isnan(int_flux_map).sum() == 5443
        assert np.isnan(gauss_flux_map).sum() == 5443

        orig_log = lime.load_frame(lines_log_address)

        idx_j, idx_x = [int(item) for item in spaxel_label.split('-')]

        assert np.isclose(orig_log.loc['O3_5007A', 'intg_flux'], int_flux_map[idx_j, idx_x])
        assert np.isclose(orig_log.loc['O3_5007A', 'profile_flux'], gauss_flux_map[idx_j, idx_x])

        return

