from numpy import exp, linspace, random
from lmfit import Model


def gaussian(x, amp, cen, sigma):
    return amp * exp(-(x-cen)**2 / sigma)


gmodel = Model(gaussian, prefix='H1_6563A_')
print(f'parameter names: {gmodel.param_names}')
print(f'independent variables: {gmodel.independent_vars}')

param_conf = {'expr': 'H1_6563A_sigma'}
gmodel.set_param_hint('N2_6584A_sigma', **param_conf)

gmodel += Model(gaussian, prefix='N2_6584A_')
print(f'parameter names: {gmodel.param_names}')
print(f'independent variables: {gmodel.independent_vars}')

gmodel.make_params()
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
