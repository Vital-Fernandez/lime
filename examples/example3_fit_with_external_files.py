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
obsFitsFile = './sample_data/gp121903_BR.fits'
lineMaskFile = './sample_data/osiris_mask.txt'
cfgFile = './sample_data/config_file.cfg'

# Load spectrum
wave, flux, header = import_osiris_fits(obsFitsFile)

# Load mask
mask = lime.load_lines_log(lineMaskFile)

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)

# Declare line measuring object
z_obj = obs_cfg['sample_data']['z_array'][2]
norm_flux = obs_cfg['sample_data']['norm_flux']
gp_spec = lime.Spectrum(wave, flux, redshift=z_obj, norm_flux=norm_flux)
gp_spec.plot_spectrum(spec_label=f'GP121903 spectrum', frame='rest')

# Find lines
peaks_table, matched_masks_DF = gp_spec.match_line_mask(mask, obs_cfg['sample_data']['noiseRegion_array'])
gp_spec.plot_spectrum(peaks_table=peaks_table, match_log=matched_masks_DF, spec_label=f'GP121903 spectrum', log_scale=True, frame='rest')

# Correct line region
corrected_mask_file = './sample_data/gp121903_BR_mask_corrected.txt'
lime.save_line_log(matched_masks_DF, corrected_mask_file)

# Object line fitting configuration
fit_cfg = obs_cfg['gp121903_line_fitting']

# Measure the emission lines
for i, lineLabel in enumerate(matched_masks_DF.index.values):
    wave_regions = matched_masks_DF.loc[lineLabel, 'w1':'w6'].values
    gp_spec.fit_from_wavelengths(lineLabel, wave_regions, user_cfg=fit_cfg)
    # if '_b' in lineLabel:
    #     gp_spec.display_results(log_scale=True, frame='rest')

# Display fits in grid
gp_spec.plot_line_grid(gp_spec.log, frame='rest')

# Display fits along the spectrum
gp_spec.plot_spectrum(include_fits=True, frame='rest')

# Save the results
lime.save_line_log(gp_spec.log, './sample_data/gp121903_linelog.txt')
lime.save_line_log(gp_spec.log, './sample_data/gp121903_flux_table.pdf', parameters=['eqw', 'intg_flux', 'intg_err'])
lime.save_line_log(gp_spec.log, './sample_data/gp121903_linelog.fits', ext='GP121903')
lime.save_line_log(gp_spec.log, './sample_data/gp121903_linelog.xlsx', ext='GP121903')
lime.save_line_log(gp_spec.log, './sample_data/gp121903_linelog.asdf', ext='GP121903')

