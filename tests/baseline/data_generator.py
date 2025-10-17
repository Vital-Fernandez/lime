import numpy as np
import lime
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

data_folder = Path('../../examples/doc_notebooks/0_resources/spectra')
baseline_folder = Path(__file__).parent

# Inputs
conf_file = baseline_folder/'lime_tests.toml'
line_bands_file = baseline_folder/'SHOC579_MANGA38-35_bands.txt'
spatial_mask_address = baseline_folder/'SHOC579_MANGA38-35_mask.fits'
cube_file = data_folder/'manga-8626-12704-LOGCUBE.fits.gz'

# Outputs
spec_text_address = baseline_folder/'SHOC579_MANGA38-35.txt'
lines_log_file = baseline_folder/'SHOC579_MANGA38-35_log.txt'
latex_log_file = baseline_folder/'SHOC579_MANGA38-35_log.tex'
cube_log = baseline_folder/'SHOC579_MANGA38-35_log.txt'
cube_log_address = baseline_folder/'SHOC579_log.fits'

# Configuration
fit_cfg = lime.load_cfg(conf_file)
redshift = fit_cfg['object_properties']['shoc579']['redshift']
norm_flux = fit_cfg['object_properties']['shoc579']['norm_flux']
spax_cords = fit_cfg['object_properties']['shoc579']['spaxel_coords']

# Open the MANGA cube fits file
remake_spec = True

if remake_spec:
    with fits.open(cube_file) as hdul:
        wave = hdul['WAVE'].data
        flux_cube = hdul['FLUX'].data * norm_flux
        hdr = hdul['FLUX'].header

        # Convert inverse variance cube to standard error, masking 0-value pixels first
        ivar_cube = hdul['IVAR'].data
        ivar_cube[ivar_cube == 0] = np.nan
        err_cube = np.sqrt(1/ivar_cube) * norm_flux

    # WCS from the observation header
    wcs = WCS(hdr)

    # ---------------- Cube
    shoc579 = lime.Cube(wave, flux_cube, err_cube, redshift=redshift, norm_flux=norm_flux, wcs=wcs,
                        pixel_mask=np.isnan(err_cube))

    # ---------------- Spectrum
    spax = shoc579.get_spectrum(spax_cords[0], spax_cords[1], id_label='SHOC579_Manga38-35')
    spax.retrieve.spectrum(spec_text_address)
    spax.retrieve.spectrum(data_folder/spec_text_address.name)


# Load the text file
spax = lime.Spectrum.from_file(spec_text_address, instrument='text')

# Generate the bands
bands = spax.retrieve.lines_frame(band_vsigma=90, fit_cfg=fit_cfg, obj_cfg_prefix='38-35', automatic_grouping=True)

# Frame fitting
spax.fit.frame(bands, fit_cfg, obj_cfg_prefix='38-35')
spax.plot.spectrum(rest_frame=True, log_scale=False)

# Save the results
lime.save_frame(line_bands_file, bands)
spax.save_frame(lines_log_file)
spax.save_frame(latex_log_file, param_list=['particle', 'wavelength', 'group_label', 'latex_label'])