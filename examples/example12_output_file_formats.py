import numpy as np
from astropy.io import fits
import lime

# Specify the file structure
spec_file = '/user/sample_folder/obj1.fits'
config_file = '/user/treatment/sample_cfg.txt'
outputs_folder = '/user/treatment/outputs'

# Load your data
sample_cfg = lime.load_cfg(config_file)
z_obj, norm_obj = sample_cfg['obj1']['z'], sample_cfg['obj1']['norm_flux']
wave, flux = fits.getdata(spec_file, extname='WAVE'), fits.getdata(spec_file, extname='FLUX')
err = fits.getdata(spec_file, extname='SIGMA')

# Import default line bands table (pandas dataframe)
bands = lime.spectral_mask_generator(wave_interval=(3300, 9500))

# Define the LiMe spectrum
spec = lime.Spectrum(wave, flux, err, redshift=z_obj, pixel_mask=np.isnan(flux))

# Convert the units (default values are angstroms and Flam = erg cm-2 s-1 A-1)
spec.convert_units('A', 'Flam', norm_flux=norm_obj)

# Fit a line
line_label = 'N2_6583.3513A'
line_band = bands.loc[line_label, 'w1':'w6'].values
fit_conf = sample_cfg['line_fitting']
spec.fit_from_wavelengths(line_label, line_band, fit_conf)

# Save the results to an external file
lime.save_line_log(spec.log, f'{outputs_folder}/results_log.txt')
lime.save_line_log(spec.log, f'{outputs_folder}/results_log.pdf', parameters=['eqw', 'intg_flux', 'intg_err'])
lime.save_line_log(spec.log, f'{outputs_folder}/results_log.xlsx', ext='obj1')
lime.save_line_log(spec.log, f'{outputs_folder}/results_log.fits', ext='obj1')


# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)
#
# print(lime.tools.air_to_vacuum_function([5008.240, 6549.86, 6585.27]))
#
# vac_waves = np.array([5008.240, 6549.86, 6585.27])
# n_array = lime.tools.refraction_index_air_vacuum(vac_waves)
# air_vac = vac_waves/n_array
# print(air_vac)
#
# # Load the parents mask
# master_mask_dir = f'{lime._dir_path}/resources'
# mask_df = lime.load_lines_log(f'{master_mask_dir}/parent_mask.txt')
#
# # Converting the vacuum wavelengths to air
# vac_waves = mask_df.wave_vac.values
# n_array = lime.tools.refraction_index_air_vacuum(vac_waves)
# air_vac = vac_waves/n_array
# mask_df.wave_obs = np.round(air_vac, 4)
# lime.save_line_log(mask_df, f'{master_mask_dir}/parent_mask.txt')
#
# # Setting the line label using the standard notation (vacum wavelentghs for < 2000 or > 20000 angstrons and rest air)
# idcs_vac = (mask_df.wave_vac < 2000) | (mask_df.wave_vac > 20000)
# mask_df.loc[idcs_vac, 'mixed_waves'] = mask_df.loc[idcs_vac, 'wave_vac']
# mask_df.loc[~idcs_vac, 'mixed_waves'] = mask_df.loc[~idcs_vac, 'wave_obs']
# mixed_waves = np.round(mask_df.mixed_waves.values, 0).astype(int)
# ion_array, wave_array, latex_array = lime.label_decomposition(mask_df.index.values)
# output_array = np.core.defchararray.add(ion_array, '_')
# output_array = np.core.defchararray.add(output_array, mixed_waves.astype(str))
# output_array = np.core.defchararray.add(output_array, 'A')
# mask_df.rename(index=dict(zip(mask_df.index.values, output_array)), inplace=True)
# mask_df.drop('mixed_waves', inplace=True, axis=1)
# lime.save_line_log(mask_df, f'{master_mask_dir}/parent_mask.txt')
# print(mask_df)pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)
#
# print(lime.tools.air_to_vacuum_function([5008.240, 6549.86, 6585.27]))
#
# vac_waves = np.array([5008.240, 6549.86, 6585.27])
# n_array = lime.tools.refraction_index_air_vacuum(vac_waves)
# air_vac = vac_waves/n_array
# print(air_vac)
#
# # Load the parents mask
# master_mask_dir = f'{lime._dir_path}/resources'
# mask_df = lime.load_lines_log(f'{master_mask_dir}/parent_mask.txt')
#
# # Converting the vacuum wavelengths to air
# vac_waves = mask_df.wave_vac.values
# n_array = lime.tools.refraction_index_air_vacuum(vac_waves)
# air_vac = vac_waves/n_array
# mask_df.wave_obs = np.round(air_vac, 4)
# lime.save_line_log(mask_df, f'{master_mask_dir}/parent_mask.txt')
#
# # Setting the line label using the standard notation (vacum wavelentghs for < 2000 or > 20000 angstrons and rest air)
# idcs_vac = (mask_df.wave_vac < 2000) | (mask_df.wave_vac > 20000)
# mask_df.loc[idcs_vac, 'mixed_waves'] = mask_df.loc[idcs_vac, 'wave_vac']
# mask_df.loc[~idcs_vac, 'mixed_waves'] = mask_df.loc[~idcs_vac, 'wave_obs']
# mixed_waves = np.round(mask_df.mixed_waves.values, 0).astype(int)
# ion_array, wave_array, latex_array = lime.label_decomposition(mask_df.index.values)
# output_array = np.core.defchararray.add(ion_array, '_')
# output_array = np.core.defchararray.add(output_array, mixed_waves.astype(str))
# output_array = np.core.defchararray.add(output_array, 'A')
# mask_df.rename(index=dict(zip(mask_df.index.values, output_array)), inplace=True)
# mask_df.drop('mixed_waves', inplace=True, axis=1)
# lime.save_line_log(mask_df, f'{master_mask_dir}/parent_mask.txt')
# print(mask_df)

# # Converting the vacuum wavelengths to air
# vac_waves = mask_df.wave_vac.values
# n_array = lime.tools.refraction_index_air_vacuum(vac_waves)
# air_vac = vac_waves/n_array
# mask_df.wave_obs = np.round(air_vac, 4)
# lime.save_line_log(mask_df, f'{master_mask_dir}/parent_mask.txt')
# lime.save_line_log(mask_df, f'{master_mask_dir}/parent_mask.txt')




# from numpy import exp, linspace, random
# from lmfit import Model
#
#
# def gaussian(x, amp, cen, sigma):
#     return amp * exp(-(x-cen)**2 / sigma)
#
#
# gmodel = Model(gaussian, prefix='H1_6563A_')
# print(f'parameter names: {gmodel.param_names}')
# print(f'independent variables: {gmodel.independent_vars}')
#
# param_conf = {'expr': 'H1_6563A_sigma'}
# gmodel.set_param_hint('N2_6584A_sigma', **param_conf)
#
# gmodel += Model(gaussian, prefix='N2_6584A_')
# print(f'parameter names: {gmodel.param_names}')
# print(f'independent variables: {gmodel.independent_vars}')
#
# gmodel.make_params()
# import asdf
# import numpy as np
# import pandas as pd
#
# import lime
#
#
# def asdf_to_log(file_address, ext_name):
#
#     with asdf.open(file_address) as af:
#         asdf_RA = af[ext_name]
#         df_RA = pd.DataFrame.from_records(asdf_RA, columns=asdf_RA.dtype.names)
#         df_RA.set_index('index', inplace=True)
#
#     return df_RA
#
#
# columns_dtypes = {'wavelength': '<f8', 'intg_flux': '<f8', 'intg_err': '<f8', 'gauss_flux': '<f8', 'gauss_err': '<f8',
#                 'eqw': '<f8', 'eqw_err': '<f8', 'ion': '<U50', 'latex_label': '<U100', 'profile_label': '<U120',
#                 'pixel_mask': '<U100', 'w1': '<f8', 'w2': '<f8', 'w3': '<f8', 'w4': '<f8', 'w5': '<f8', 'w6': '<f8',
#                 'w_i': '<f8', 'w_f': '<f8', 'peak_wave': '<f8', 'peak_flux': '<f8', 'cont': '<f8', 'std_cont': '<f8',
#                 'm_cont': '<f8', 'n_cont': '<f8', 'snr_line': '<f8', 'snr_cont': '<f8', 'z_line': '<f8', 'amp': '<f8',
#                 'center': '<f8', 'sigma': '<f8', 'amp_err': '<f8', 'center_err': '<f8', 'sigma_err': '<f8', 'v_r': '<f8',
#                 'v_r_err': '<f8', 'sigma_vel': '<f8', 'sigma_vel_err': '<f8', 'sigma_thermal': '<f8', 'sigma_instr': '<f8',
#                 'pixel_vel': '<f8', 'FWHM_intg': '<f8', 'FWHM_g': '<f8', 'FWZI': '<f8', 'v_med': '<f8', 'v_50': '<f8',
#                 'v_5': '<f8', 'v_10': '<f8', 'v_90': '<f8', 'v_95': '<f8', 'chisqr': '<f8', 'redchi': '<f8',
#                 'aic': '<f8', 'bic': '<f8', 'observations': '<U50', 'comments': '<U50'}
#
# # # Reading the log from a .txt file
# # log_txt_file = './gp121903_linelog.txt'
# # log_dfFull = pd.read_csv(log_txt_file, delim_whitespace=True, header=0, index_col=0)
# # log_df = log_dfFull[['wavelength', 'eqw', 'intg_flux', 'eqw_err', 'sigma_instr']]
# #
# # # idcs_m = (log_df.index.str.contains('_m')) & (log_df.profile_label != 'no')
# # # fit_conf = dict(zip(log_df.loc[idcs_m].index.values, log_df.loc[idcs_m].profile_label.values))
# # # log_pdf_file = './sample_data/gp121903_linelog.pdf'
# # #
# # # columns = ['wavelength', 'eqw', 'intg_flux', 'eqw_err', 'sigma_instr']
# # # lime.save_line_log(log_df, log_pdf_file, parameters=columns)
# #
# # # Saving the log to a .asdf file
# # log_asdf_file = './gp121903_linelog.asdf'
# # tree = {'log0': log_df.to_records(index=True, column_dtypes=columns_dtypes, index_dtypes='<U50')}
# # af = asdf.AsdfFile(tree)
# # af.write_to(log_asdf_file)
# #
# # # Reading the log in a .asdf file
# # # ext = 'log0'
# # # with asdf.open(log_asdf_file) as af:
# # #     logRA = af[ext]
# # #     logASDF = pd.DataFrame.from_records(logRA, columns=logRA.dtype.names)
# # #     logASDF.set_index('index', inplace=True)
# #
# # ext = 'log0'
# # log_asdf = asdf_to_log(log_asdf_file, ext)
# #
# # print(log_asdf.equals(log_df))
# # print(ext, log_asdf['intg_flux'].values)
# #
# # # Adding several extensions
# # for i in [0.0, 1.0, 2.0, 3.0]:
# #     ext = f'log{i:.0f}'
# #     log_df['intg_flux'] = i
# #     tree = {ext: log_df.to_records(index=True, column_dtypes=columns_dtypes, index_dtypes='<U50')}
# #
# #     with asdf.open(log_asdf_file, mode='rw') as af:
# #         af.tree.update(tree)
# #         af.update()
# #
# # # Reading the extensions
# # for i in [0.0, 1.0, 2.0, 3.0]:
# #     ext = f'log{i:.0f}'
# #     log_asdf = asdf_to_log(log_asdf_file, ext)
# #     # print(ext, log_asdf['intg_flux'].values)
# #     print(log_asdf)
#
# # Reading the log from a .txt file
# log_txt_file = './gp121903_linelog.txt'
# log_df = pd.read_csv(log_txt_file, delim_whitespace=True, header=0, index_col=0)
#
# # Saving the log to a .asdf file
# log_asdf_file = './gp121903_linelog.asdf'
# lime.save_line_log(log_df, log_asdf_file)
# log_asdf = lime.load_lines_log(log_asdf_file)
# print(log_asdf['intg_flux'].values)
#
# # Adding several extensions
# for i in [0.0, 1.0, 2.0, 3.0]:
#     ext = f'log{i:.0f}'
#     log_df['intg_flux'] = i
#     lime.save_line_log(log_df, log_asdf_file, ext=ext)
#
# # Reading several extensions
# for i in [0.0, 1.0, 2.0, 3.0]:
#     ext = f'log{i:.0f}'
#     log_asdf = lime.load_lines_log(log_asdf_file, ext=ext)
#     print(log_asdf['intg_flux'].values)
