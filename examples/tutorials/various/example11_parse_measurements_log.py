import numpy as np
from astropy.io import fits
import lime


def import_osiris_fits(file_address, ext=0):

    # Open fits file
    with fits.open(file_address) as hdul:
        data, hdr = hdul[ext].data, hdul[ext].header

    w_min, dw, n_pix = hdr['CRVAL1'],  hdr['CD1_1'], hdr['NAXIS1']
    w_max = w_min + dw * n_pix
    wavelength = np.linspace(w_min, w_max, n_pix, endpoint=False)

    return wavelength, data, hdr


# State the data files
obsFitsFile = './sample_data/gp121903_osiris.fits'
lineMaskFile = './sample_data/osiris_bands.txt'
cfgFile = './sample_data/config_file.cfg'

# Load spectrum
wave, flux, header = import_osiris_fits(obsFitsFile)

# Load mask
mask = lime.load_log(lineMaskFile)

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)
fit_cfg = obs_cfg['gp121903_line_fitting']

# Declare line measuring object
z_obj = obs_cfg['sample_data']['z_array'][2]
norm_flux = obs_cfg['sample_data']['norm_flux']
gp_spec = lime.Spectrum(wave, flux, redshift=z_obj, norm_flux=norm_flux)

# Find lines
peaks_table, matched_masks_DF = gp_spec.match_line_mask(mask, obs_cfg['sample_data']['noiseRegion_array'])

# Measure the emission lines
for i, lineLabel in enumerate(matched_masks_DF.index.values):
    wave_regions = matched_masks_DF.loc[lineLabel, 'w1':'w6'].values
    gp_spec.fit_from_wavelengths(lineLabel, wave_regions, user_cfg=fit_cfg)

# Save the results
lime.save_log(gp_spec.log, './sample_data/example_3.txt')

# Add new parameters to the log
parameters = ['eqw_gaussian',
              'eqw_gaussian_err']

formulation = ['profile_flux/cont',
               '(profile_flux/cont) * sqrt((profile_flux_err/profile_flux)**2 + (std_cont/cont)**2)']

lime.log_parameters_calculation('./sample_data/example_3.txt', parameters, formulation)
