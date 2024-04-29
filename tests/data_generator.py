import numpy as np
import lime
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

baseline_folder = Path(__file__).parent / 'baseline'

# Inputs
cube_file = Path('../examples/sample_data/spectra/manga-8626-12704-LOGCUBE.fits.gz')
conf_file = baseline_folder/'manga.toml'
line_bands_file = baseline_folder/'manga_line_bands.txt'
spatial_mask_address = baseline_folder/'SHOC579_mask.fits'

# Outputs
file_address = baseline_folder/'manga_spaxel.txt'
lines_log_file = baseline_folder/'manga_lines_log.txt'
cube_log = baseline_folder/'manga_lines_log.txt'
cube_log_address = baseline_folder/'SHOC579_log.fits'

# Configuration
fit_cfg = lime.load_cfg(conf_file)

# Parameters
redshift = 0.0475
norm_flux = 1e-17
spaxel_coords = (38, 35)

# Open the MANGA cube fits file
with fits.open(cube_file) as hdul:
    wave = hdul['WAVE'].data
    flux_cube = hdul['FLUX'].data * norm_flux
    hdr = hdul['FLUX'].header

    # Convert inverse variance cube to standard error, masking 0-value pixels first
    ivar_cube = hdul['IVAR'].data
    ivar_cube[ivar_cube == 0] = np.nan
    err_cube = np.sqrt(1/ivar_cube) * norm_flux

# WCS from the obsevation header
wcs = WCS(hdr)

# ---------------- Cube
shoc579 = lime.Cube(wave, flux_cube, err_cube, redshift=redshift, norm_flux=norm_flux, wcs=wcs,
                    pixel_mask=np.isnan(err_cube))

# ---------------- Spectrum
spax = shoc579.get_spectrum(spaxel_coords[0], spaxel_coords[1], id_label='SHOC579-Manga38-35')
spax.plot.spectrum(rest_frame=True)
wave_array, flux_array, err_array = spax.wave.data, spax.flux.data * norm_flux, spax.err_flux.data * norm_flux
np.savetxt(file_address, np.c_[wave_array, flux_array, err_array])

# Frame fitting
spax.fit.frame(line_bands_file, fit_cfg, id_conf_prefix='38-35', progress_output=None)
spax.save_frame(lines_log_file)

spax.plot.velocity_profile('H1_4861A')

# Plots
spax.plot.spectrum(include_fits=True)
spax.plot.bands('Fe3_4658A_p-g-emi')

# Cube fitting
# shoc579.fit.spatial_mask(spatial_mask_address, cube_log_address, fit_conf=fit_cfg, line_detection=True, mask_list=['MASK_0'])