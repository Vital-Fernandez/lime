import numpy as np
import lime
from astropy.io import fits


# Example load function for osiris spectra file
def osiris_load_function(log_df, obs_idx, data_folder, **kwargs):

    # Open fits file
    ext = 0
    file_address = f'{data_folder}/{obs_idx[log_df.index.names.index("file")]}'
    with fits.open(file_address) as hdul:
        data, hdr = hdul[ext].data, hdul[ext].header

    # Reconstruct the wavelength array
    w_min, dw, n_pix = hdr['CRVAL1'], hdr['CD1_1'], hdr['NAXIS1']
    w_max = w_min + dw * n_pix
    wavelength = np.linspace(w_min, w_max, n_pix, endpoint=False)

    # Compute the redshift from the mean line centroid redshift
    log_obj = log_df.xs(obs_idx[0], level='id', drop_level=False)
    redshift = np.nanmean(log_obj.z_line.to_numpy())

    # Multiplicative factor to distinguish the spectra on the plots
    ids = log_df.index.get_level_values('id').unique()
    id_idx = np.where(ids == obs_idx[0])[0][0]
    flux = data + np.nanmean(data) * 1.10 * id_idx

    # Recover the normalization from the sample input parameters
    norm_flux = kwargs['norm_flux']

    # Create the spectrum object with its line measurements
    spec = lime.Spectrum(wavelength, flux, redshift=redshift, norm_flux=norm_flux)

    return spec


# 2nd example load_function
def osiris_compliment(log_df, obs_idx, root_address, **kwargs):

    log_obj = log_df.xs(obs_idx, level='id')
    redshift = np.nanmean(log_obj.z_line.to_numpy())
    norm_flux = 1e-17

    return {'redshift': redshift, 'norm_flux': norm_flux}


# Declaring the name of the observations
id_list = ['GP121903_A', 'GP121903_B', 'GP121903_C']

# Declaring the observations root folder and the individual .fits files
folder_obs = f'../sample_data'
obs_list = ['spectra/gp121903_osiris.fits'] * 3

# We declare the line measurements logs
log_list = [f'{folder_obs}/example3_linelog.txt'] * 3

# We create the sample using the list of objects and files
sample1 = lime.Sample.from_file(id_list, log_list, obs_list, folder_obs=folder_obs, load_function=osiris_load_function,
                                norm_flux=1e-17)

# Get an individual observations:
specA = sample1.get_observation('GP121903_A')

# Review the measurements:
specA.plot.spectrum(include_fits=True)

# We can save the combiened sample log, so it can construct the Sample variable in the future.
sample_log_address = f'{folder_obs}/sample_log.txt'
sample1.save_frame(sample_log_address)

# Just with the combined log and the load function
sample2 = lime.Sample(sample_log_address, load_function=osiris_load_function, folder_obs=folder_obs, norm_flux=1e-17)
sample2.plot.spectra(rest_frame=True)

