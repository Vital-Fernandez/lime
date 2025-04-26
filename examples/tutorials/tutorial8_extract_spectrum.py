from pathlib import Path
import lime

# Data folder
data_folder = Path('../sample_data')
ref_bands = '/home/vital/PycharmProjects/lime/src/lime/resources/lines_database_v2.0.0.txt'

# Configuration file
cfg = lime.load_cfg(data_folder/'long_slit.toml')

# Spectra list
object_dict = {'osiris':'gp121903', 'nirspec':'ceers1027', 'isis':'Izw18', 'sdss':'SHOC579'}

# File list
files_dict = {'osiris': 'gp121903_osiris.fits',
              'nirspec':'hlsp_ceers_jwst_nirspec_nirspec10-001027_comb-mgrat_v0.7_x1d-masked.fits',
              'isis': 'IZW18_isis.fits',
              'sdss':'SHOC579_SDSS_dr18.fits'}



inst, obj = 'sdss', 'SHOC579'
file_path = data_folder/'spectra'/files_dict[inst]
redshift = cfg[inst][obj]['z']

# Create the observation object
spec = lime.Spectrum.from_file(file_path, inst, redshift=redshift)

# spec.plot.spectrum()

# Output
SHOC579_txt_address = data_folder/'spectra'/f'{obj}_data.txt'

spec.fit.frame(data_folder /'bands' /f'{obj}_{inst}_bands.txt', cfg, obj_cfg_prefix=f'{obj}_{inst}')
spec_arrays = spec.retrieve.spectrum('H1_6563A', output_address=SHOC579_txt_address, split_components=True)

lime.line_bands()

spec.plot.spectrum()
print(spec.plot.frame)
# spec.plot.bands('H1_6563A')


#     # Create the observation object
#     spec = lime.Spectrum.from_file(file_path, inst, redshift=redshift)
#

# # Loop through the files and measure the lines
# for i, items in enumerate(object_dict.items()):
#
#     inst, obj = items
#     file_path = data_folder/'spectra'/files_dict[inst]
#     redshift = cfg[inst][obj]['z']
#
#     # Create the observation object
#     spec = lime.Spectrum.from_file(file_path, inst, redshift=redshift)
#
#     if spec.units_wave != 'AA':
#         spec.unit_conversion('AA', 'FLAM')
#
#     # Bands the results
#     bands = spec.retrieve.line_bands(band_vsigma=100)
#     # spec.check.bands(data_folder/'bands'/f'{obj}_{inst}_bands.txt', ref_bands=bands, exclude_continua=False)
#
#     spec.fit.frame(data_folder/'bands'/f'{obj}_{inst}_bands.txt', cfg, id_conf_prefix=f'{obj}_{inst}')
#     # spec.plot.grid()
#
#     # spec.plot.spectrum(rest_frame=False)


