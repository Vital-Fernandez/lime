from pathlib import Path
import lime

data_folder = Path('../0_resources/spectra')
output_folder = Path('../0_resources/results')
file_address = data_folder/'SHOC579_manga_spaxel.txt'

# Create the spectrum object
spec = lime.Spectrum.from_file(file_address, instrument='text', redshift=0.0475)

# # Get the object spectrum bands
# bands_default = spec.retrieve.lines_frame(band_vsigma=200)
#
# # Default database in angstrom and with emission profiles
# spec.fit.bands('Fe3_4658A', bands_default, cont_from_bands=True)
# spec.plot.bands(rest_frame=True, show_continua=False)
# spec.fit.bands('Fe3_4658A', bands_default, shape='abs', fit_cfg={'Fe3_4658A_center': {'value':4651.0}}, cont_from_bands=True)
# spec.plot.bands(rest_frame=True, show_continua=False)

# # Switch the spectrum and database units to nm, and set the default shape to absorption
# spec.unit_conversion(wave_units_out='nm')
# lime.lineDB.set_database(units_wave='nm', default_shape='abs', update_latex=True)
# bands_um = spec.retrieve.lines_frame(band_vsigma=200, update_latex=True)
# print(bands_um)
# spec.fit.bands('Fe3_4658A', bands_um)
# spec.plot.bands(rest_frame=True, show_continua=False)


# spec.fit.bands('Fe3_4658A', bands_um, shape='emi')
# spec.plot.bands(rest_frame=True, show_continua=False)
# spec.fit.bands('Fe3_4658A', bands_um, fit_cfg={'Fe3_4658A_center': {'value':465.1}}, cont_from_bands=True)
# spec.plot.bands(rest_frame=True, show_continua=False)
#
# # Reset the database and spectrum to the original values
# lime.lineDB.reset()
# spec.unit_conversion(wave_units_out='AA')
#
# # Add a profile and shape columns to the input lines table to adjust the fitting
# bands_obj = spec.retrieve.lines_frame(band_vsigma=200)
# bands_obj[['shape', 'profile']] = 'emi', 'g'
# bands_obj.loc['Fe3_4658A', 'wavelength'] = 4651.0
# bands_obj.loc['Fe3_4658A', 'shape'] = 'abs'
# spec.fit.bands('Fe3_4658A', bands_obj, cont_from_bands=True)
# spec.plot.bands(show_continua=False, rest_frame=True)
#
# # Using the shape argument does not work now because the table input has preference over the argument
# spec.fit.bands('Fe3_4658A', bands_obj, shape='emi', fit_cfg={'Fe3_4658A_center': {'value': 4658}}, cont_from_bands=True)
# spec.plot.bands(show_continua=False, rest_frame=True)
#
# To overwrite the shape value on the input tables you need to set it the fit_cfg or use the line label suffix.
# fit_cfg = {'transitions': {"Fe3_4658A": {"wavelength": 4658,
#                                          "shape": "emi"}
#                            }}
#
# spec.fit.bands('Fe3_4658A', bands_obj, fit_cfg=fit_cfg,  cont_from_bands=True)
# spec.plot.bands(show_continua=False, rest_frame=True)

# fit_cfg =  {"Fe3_4658A_b"   : 'Fe3_4658A+Fe3_4658A_s-emi',
#             'transitions'   : {"Fe3_4658A_s-emi"    : {"wavelength": 4658, "shape": "emi"}}
#             }
#
# spec.fit.bands('Fe3_4658A_b', bands_obj, fit_cfg=fit_cfg, cont_from_bands=True)
# spec.plot.bands(show_continua=False, rest_frame=True)

# # Or using the profile suffix
# spec.fit.bands('Fe3_4658A_s-emi', bands_obj, fit_cfg={'Fe3_4658A_s-emi_center': {'value': 4658}}, cont_from_bands=True)
# spec.plot.bands(show_continua=False, rest_frame=True)
#
# # In addition to the shape argument, you can also use set the profile from the lines frame table
# bands_obj.loc['H1_4861A', 'profile'] = 'l'
# spec.fit.bands('H1_4861A', bands_obj, cont_from_bands=True)
# spec.plot.bands(show_continua=False, rest_frame=True)
#
# To change the default database for certain scripts you can use the .reset command.
lines_db_nm = lime.lines_frame(units_wave='nm', update_latex=True, update_labels=True)
lime.save_frame(output_folder/'lines_db_nm.txt', lines_db_nm)

lime.lineDB.reset(frame_address=output_folder/'lines_db_nm.txt', default_profile='l', default_shape='emi')
print(lime.lineDB.frame[['wavelength', 'units_wave', 'latex_label']])
spec.unit_conversion(wave_units_out='nm')
spec.fit.bands('H1_486.1nm')
spec.plot.bands()



# # Finally, you can change the default database permanently at this location
# lime_database_address = lime.transitions._DATABASE_FILE


