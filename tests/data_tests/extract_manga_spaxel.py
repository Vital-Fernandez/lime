import numpy as np
import lime
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

# Inputs
cube_file = Path('../../examples/sample_data/manga-8626-12704-LOGCUBE.fits.gz')
conf_file = Path('manga.toml')

# Outputs
file_address = 'manga_spaxel.txt'
cube_plot_address = 'cube_manga_plot.png'
spectrum_plot_address = 'spectrum_manga_spaxel.png'
line_plot_address = 'Fe3_4658A_manga_spaxel.png'
line_bands_file = f'manga_line_bands.txt'
lines_log_file = f'manga_lines_log.txt'

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
shoc579.plot.cube('H1_6563A', output_address=cube_plot_address)

# ---------------- Spectrum
spax = shoc579.get_spectrum(spaxel_coords[0], spaxel_coords[1])
wave_array, flux_array, err_array = spax.wave.data, spax.flux.data * norm_flux, spax.err_flux.data * norm_flux
np.savetxt(file_address, np.c_[wave_array, flux_array, err_array])

# Plot Spectrum
# spax.plot.spectrum(output_address=spectrum_plot_address)

# # Line fitting
# spax.fit.bands('Fe3_4658A')
# spax.plot.bands(output_address=line_plot_address)
# spax.check.bands(line_bands_file, n_cols=10)
# spax.log = spax.log[0:0]

# Frame fitting
spax.fit.frame(line_bands_file, fit_cfg, id_conf_prefix='38-35')
spax.plot.spectrum(include_fits=True, rest_frame=True)
spax.save_log(lines_log_file)

# Save iron line plot
spax.plot.bands('Fe3_4658A_p-g_emis', output_address=line_plot_address)

# Sample with 3 (repeated) observations
log_dict = {}
for i in range(3):
    log_dict[f'obj_{i}'] = spax.log.copy()

obs = lime.Sample()
obs.add_log_list(list(log_dict.keys()), list(log_dict.values()))

# spax.fit.bands('O3_4959A_b', line_bands_file, fit_conf=fit_cfg['38-35_line_fitting'])
# spax.fit.bands('O3_5007A_b', line_bands_file, fit_conf=fit_cfg['38-35_line_fitting'])
# spax.fit.bands('Fe3_4658A_b', line_bands_file, fit_conf=fit_cfg['38-35_line_fitting'])
# spax.fit.report()
# spax.plot.bands(rest_frame=True)

# S3_9530A_b