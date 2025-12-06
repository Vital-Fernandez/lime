from pathlib import Path
import lime

# State the data files
data_folder = Path('../doc_notebooks/0_resources/')
cfgFile = f'{data_folder}/long_slit.toml'
osiris_gp_df_path =  f'{data_folder}/bands/osiris_green_peas_linesDF.txt'

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)

# Instrument - file dictionary
files_dict = {'isis': 'IZW18_isis.fits',
              'nirspec':'hlsp_ceers_jwst_nirspec_nirspec10-001027_comb-mgrat_v0.7_x1d-masked.fits',
              'sdss':'SHOC579_SDSS_dr18.fits'}

# Instrument - object dictionary
object_dict = {'nirspec':'ceers1027',
               # 'isis':'Izw18',
               'sdss':'SHOC579'}

# Loop through the observations
for i, items in enumerate(object_dict.items()):

    inst, obj = items
    file_path = f'{data_folder}/spectra/{files_dict[inst]}'
    redshift = obs_cfg[inst][obj]['z']
    print('\n', obj, inst, redshift)

    # Create the observation object
    spec = lime.Spectrum.from_file(file_path, inst, redshift=redshift)

    # Unit conversion for NIRSPEC object
    if spec.units_wave != 'AA':
        spec.unit_conversion('AA', 'FLAM')

    # Detect the components
    spec.infer.components(exclude_continuum=False)

    # Show the components
    spec.plot.spectrum(show_components=True)

    # # Revised bands for every object
    # bands_df = spec.retrieve.lines_frame(band_vsigma = 100, map_band_vsigma = {'O2_3726A': 200, 'O2_3729A': 200,
    #                                                                            'H1_4861A': 200, 'H1_6563A': 200,
    #                                                                            'N2_6548A': 200, 'N2_6583A': 200,
    #                                                                            'O3_4959A': 250, 'O3_5007A': 250},
    #                                        fit_cfg=obs_cfg, obj_cfg_prefix=f'{obj}_{inst}',
    #                                        automatic_grouping=True, ref_bands=osiris_gp_df_path)
    #
    # # Fit the lines and plot the measurements
    # spec.fit.frame(bands_df, fit_cfg=obs_cfg, obj_cfg_prefix=f'{obj}_{inst}', line_detection=True)
    #
    # # Save the measurements
    # spec.save_frame(f'{data_folder}/results/{obj}_{inst}_line_frame.txt')
    #
    # # Plot the profiles.plot.grid()
    # spec.plot.grid()
