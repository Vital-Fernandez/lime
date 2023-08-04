# import lime
# from pathlib import Path
# from astropy.io import fits
# from astropy.wcs import WCS
# import pandas as pd
# pd.set_option('display.max_columns', 8)
#
# log_address = Path(f'/home/usuario/Documents/fluxes_log.txt')
#
# log = lime.load_log(log_address, sample_levels=['sample', 'id', 'line'])
#
# lime.extract_fluxes(log, column_names=['line_flux', 'line_flux_err'], column_positions=[0, 1])
#
# lime.normalize_fluxes(log, column_name='line_flux_rel', norm_list='H1_4862A', column_normalization_name='Norm_line',
#                       sample_levels=['sample', 'id', 'line'])
#
# print(log)
# lime.save_log(log, f'que_locura.txt')

# import pandas as pd
#
# # Create individual pandas DataFrame.
# df1 = pd.DataFrame({'Col0': ['Gut', 'Gut', 'Gut', 'Gut'], 'Col1': [1, 2, 3, 4], 'Col2': [99, 98, 95, 90]}, index=['A', 'B', 'C', 'D'])
# df2 = pd.DataFrame({'Col0': ['Gut', 'Gut'], 'Col1': [1, 2], 'Col2': [99, 98]}, index=['A', 'B'])
# df3 = pd.DataFrame({'Col0': ['Fer', 'Fer'], 'Col1': [3, 4], 'Col2': [95, 90]}, index=['C', 'D'])
# df4 = pd.DataFrame({'Col0': ['Fer', 'Fer'], 'Col1': [3, 4], 'Col2': [95, 90]}, index=['B', 'C'])
#
# # Combine into one multi-index dataframe
# df_dict = dict(obj1=df1, obj2=df2, obj3=df3, obj4=df4)
#
# # Assign multi-index labels
# mDF = pd.concat(list(df_dict.values()), keys=list(df_dict.keys()))
# mDF.set_index(['Col0'], append=True, inplace=True)
# mDF.rename_axis(index=["ID", "property", 'family'], inplace=True)
#
# print(mDF, '\n')
#
# bools = mDF.index.get_level_values('property').isin(['A','B'])
# grouper = mDF.index.get_level_values('ID')
# # there should be a minimum of two (`A`, `B`)
# bools2 = pd.Series(bools).groupby(grouper).transform('sum').ge(2).array
# df_slice = mDF.loc[bools2]
#
# print(bools.sum())
# print(bools2.sum())
# print(mDF.loc[bools2].loc['obj1'])
# print(mDF.loc[bools2].loc['obj2'])
# print(mDF.loc[bools2].loc['obj3'])


# import pandas as pd
#
# # Create individual pandas DataFrame.
# df1 = pd.DataFrame({'Col1': [1, 2, 3, 4], 'Col2': [99, 98, 95, 90]}, index=['A', 'B', 'C', 'D'])
# df2 = pd.DataFrame({'Col1': [1, 2], 'Col2': [99, 98]}, index=['A', 'B'])
# df3 = pd.DataFrame({'Col1': [3, 4], 'Col2': [95, 90]}, index=['C', 'D'])
# df4 = pd.DataFrame({'Col1': [3, 4], 'Col2': [95, 90]}, index=['B', 'C'])
#
# # Combine into one multi-index dataframe
# df_dict = dict(obj1=df1, obj2=df2, obj3=df3, obj4=df4)
#
# # Assign multi-index labels
# mDF = pd.concat(list(df_dict.values()), keys=list(df_dict.keys()))
# mDF.rename_axis(index=["ID", "property"], inplace=True)
# print(mDF, '\n')
#
# bools = mDF.index.get_level_values('property').isin(['A','B'])
# grouper = mDF.index.get_level_values('ID')
# # there should be a minimum of two (`A`, `B`)
# bools = pd.Series(bools).groupby(grouper).transform('sum').ge(2).array
# print(mDF.loc[bools])


# log_address = Path('../sample_data/example3_linelog.txt')
#
# log = lime.load_log(log_address)
#
# lime.extract_fluxes(log, column_names=['line_flux', 'line_flux_err'], column_positions=[0, 1])
#
# lime.relative_fluxes(log, column_name='line_flux_rel', norm_list='H1_4861A')
#
# lime.relative_fluxes(log, line_list=['H1_6563A', 'O3_5007A'], column_name='line_flux2_rel', norm_list=['H1_4861A', 'O3_4959A'],
#                      column_normalization_name=None)
#
# lime.relative_fluxes(log, line_list=['H1_6563A/H1_4861A', 'O3_5007A/O3_4959A'], column_name='line_flux3_rel', column_normalization_name=None)
#
# print(log)

# # State the data location
# cfg_file = '../sample_data/manga.toml'
# cube_file = Path('../sample_data/manga-8626-12704-LOGCUBE.fits.gz')
# spatial_mask_file = Path('../sample_data/SHOC579_mask.fits')
# output_sample_file = Path('../sample_data/SHOC579_masked_spectra.fits')
#
# # Load the configuration file:
# obs_cfg = lime.load_cfg(cfg_file)
#
# # Observation properties
# z_obj = obs_cfg['SHOC579']['redshift']
# norm_flux = obs_cfg['SHOC579']['norm_flux']
#
# # Open the MANGA cube fits file
# with fits.open(cube_file) as hdul:
#     wave = hdul['WAVE'].data
#     flux_cube = hdul['FLUX'].data * norm_flux
#     hdr = hdul['FLUX'].header
#
# # World coordinate system from the observation
# wcs = WCS(hdr)
#
# # Define a LiMe cube object
# shoc579 = lime.Cube(wave, flux_cube, redshift=z_obj, norm_flux=norm_flux, wcs=wcs)
#
# shoc579.export_spaxels(output_sample_file, spatial_mask_file)
#
# print(fits.info(output_sample_file))