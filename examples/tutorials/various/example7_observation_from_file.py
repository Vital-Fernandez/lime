import numpy as np
from astropy.io import fits
import lime
from mpdaf.obj import Cube


# def read_muse_cube(file_address):
#
#     cube_obj = Cube(filename=str(file_address))
#     header = cube_obj.data_header
#
#     dw = header['CD3_3']
#     w_min = header['CRVAL3']
#     nPixels = header['NAXIS3']
#     w_max = w_min + dw * nPixels
#     wave_array = np.linspace(w_min, w_max, nPixels, endpoint=False)
#
#     return wave_array, cube_obj, header
#
# State the data files
obsFitsFile = '../../sample_data/spectra/gp121903_osiris.fits'
lineBandsFile = '../../sample_data/osiris_bands.txt'
cfgFile = '../../sample_data/osiris.toml'

# Load line bands
bands = lime.load_frame(lineBandsFile)

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
z_obj = obs_cfg['sample_data']['z_array'][2]
norm_flux = obs_cfg['sample_data']['norm_flux']
#
# gp_spec = lime.Spectrum.from_file(obsFitsFile, 'Osiris', redshift=z_obj, norm_flux=norm_flux)
# gp_spec.plot.spectrum(label='GP121903', rest_frame=True)
#
# manga_cube = '../../sample_data/spectra/manga-8626-12704-LOGCUBE.fits.gz'
# shoc579 = lime.Cube.from_file(manga_cube, instrument='Manga', redshift=0.0475)
# shoc579.plot.cube('H1_6563A')

megara_cube = '../../sample_data/spectra/NGC5471_datacube_LR-R_900_scale03_drp_nosky.fits'
# data = fits.getdata(megara_cube, ext=0)
# data1 = fits.getdata(megara_cube, ext=1)
# data2 = fits.getdata(megara_cube, ext=2)
#
# hdr = fits.getheader(megara_cube, ext=0)
# hdr1 = fits.getheader(megara_cube, ext=1)
# hdr2 = fits.getheader(megara_cube, ext=2)

ngc5471 = lime.Cube.from_file(megara_cube, instrument='megara', redshift=0.00091)
ngc5471.check.cube('H1_6563A')



# muse_cube = 'D:\AstroData\MUSE - Amorin\CGCG007.fits'
# cgcg007 = lime.Cube.from_file(muse_cube, instrument='Muse', redshift=0.004691)
# cgcg007.check.cube('H1_6563A')


# specprod_dir = '.'
# fits_path = f'{specprod_dir}/coadd-sv1-other-27256.fits'

# spectra_dict = open_desi_spectra(fits_path, obj_idtarget=special_ID, obj_idrows=None)
# wave = spectra_dict[special_ID]['B']['wave']
# flux = spectra_dict[special_ID]['B']['flux']
# spec = lime.Spectrum(wave, flux)
# spec.plot.spectrum()

# fits_url = 'https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/sv1/other/272/27256/coadd-sv1-other-27256.fits'
# spectra_dict = lime.OpenFitsSurvey.desi(fits_url, data_ext_list=special_ID)

# special_ID = 39627835576420141
# gp_spec = lime.Spectrum.from_survey(special_ID, 'desi', program='dark')
# gp_spec.plot.spectrum()

# manga_cube = '../../sample_data/manga-8626-12704-LOGCUBE.fits.gz'
#
# shoc579 = lime.Cube.from_file(manga_cube, instrument='Manga', redshift=0.0475)
# shoc579.plot.cube('H1_6563A')

# print(fits.getheader(manga_cube, extname='PRIMARY'))





# fits.info(muse_cube)
# fits.getheader(muse_cube, extname='PRIMARY')
#
# mpdaf_wave, mpdaf_flux, mpdaf_hdr = read_muse_cube(muse_cube)
#
# with fits.open(muse_cube) as hdul:
#
#     print(hdul[0].header)
#     header_list = [hdul[1].header]
#     w_min, dw, pixels = header_list[0]['CRVAL3'], header_list[0]['CD3_3'], header_list[0]['NAXIS3']
#     w_max = w_min + dw * pixels
#
#     wave_array = np.linspace(w_min, w_max, pixels, endpoint=False)
#     flux_array = hdul[1].data
#     err = np.sqrt(hdul[2].data)
#     wcs = WCS(header_list[0])
#
#     np.array_equal(flux_array, mpdaf_flux.data.data, equal_nan=True)

# # wave = spectra_dict['B']['wave']
# # flux = spectra_dict['B']['flux']
# # spec = lime.Spectrum(wave, flux)
# # spec.plot.spectrum()
#
# flux = None
# wave = None
# err_flux = None
#
# tolerance = 0.0001  # A , tolerance
# for b in ('B', 'R', 'Z'):
#
#     band = spectra_dict[b]
#     wave_b, flux_b, ivar_b = band['wave'], band['flux'], band['ivar']
#
#     if wave is None:
#         wave, flux, ivar = wave_b, flux_b, ivar_b
#     else:
#         idcs_match = (wave_b < wave[-1]) & (wave_b < wave_b[-1])
#         median_idx = int(np.sum(idcs_match)/2)
#         wave = np.append(wave, wave_b[median_idx:])
#         flux = np.append(flux, flux_b[median_idx:])
#         ivar = np.append(ivar, ivar_b[median_idx:])
#
#
# # wave = spectra_dict['B']['wave']
# # flux = spectra_dict['B']['flux']
# spec = lime.Spectrum(wave, flux)
# spec.plot.spectrum()

# # create wavelength array
# wave = None
# tolerance = 0.0001  # A , tolerance
# for b in sbands:
#     if wave is None:
#         wave = spectra.wave[b]
#     else:
#         wave = np.append(wave, spectra.wave[b][spectra.wave[b] > wave[-1] + tolerance])
# nwave = wave.size
