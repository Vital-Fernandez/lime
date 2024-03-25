import numpy as np
import astropy
from astropy.io import fits
import lime

program ='dark'
release ='fuji'
catalogue ='healpix'
# target_ID = 39628443939243162
target_ID = 39628517750604871
# target_ID = 39633440282250119
fits_url = 'https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/zcatalog/zall-pix-fuji.fits'

# with fits.open(fits_url, use_fsspec=True) as hdul:
#
#     zCatalogBin = hdul['ZCATALOG']
#     targetID_data = zCatalogBin.data['TARGETID']
#     program_data = zCatalogBin.data['PROGRAM']
#     idx_target = np.where((targetID_data == target_ID) & (program_data == program))[0]
#
#     # Get healpix, survey and redshift
#     hpx = zCatalogBin.data['HEALPIX'][idx_target]
#     survey = zCatalogBin.data['SURVEY'][idx_target]
#     redshift = zCatalogBin.data['Z'][idx_target]
#
#     # Compute the url address
#     url_list = []
#     for i, idx in enumerate(idx_target):
#         hpx_number = hpx[i]
#         hpx_ref = f'{hpx_number}'[:-2]
#         target_dir = f"/healpix/{survey[i]}/{program}/{hpx_ref}/{hpx_number}"
#         coadd_fname = f"coadd-{survey[i]}-{program}-{hpx_number}.fits"


data_file = f'/home/vital/Downloads/zall-pix-fuji.fits'
fits_url = 'https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/sv1/other/272/27256/coadd-sv1-other-27256.fits'
# special_ID = 39627848784286243

# spectra_dict = lime.OpenFitsSurvey.desi(fits_url, data_ext_list=special_ID)

gp_spec = lime.Spectrum.from_survey(target_ID,  'desi', program='dark', release='fuji', catalogue='healpix', ref_fits=data_file)
gp_spec.plot.spectrum(rest_frame=True)
gp_spec.fit.bands('O3_5007A')
gp_spec.plot.bands(rest_frame=True, y_scale='linear')
gp_spec.fit.bands('H1_6563A', )
gp_spec.save_log('desi_line_measurements.txt')

data = np.column_stack((gp_spec.wave, gp_spec.flux))
np.savetxt('/home/vital/Astrodata/LiMe_ml/desi_spectrum.txt', np.c_[data])

# np.savetxt('desi_comb.txt', np.c_[gp_spec.wave, gp_spec.flux])
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
