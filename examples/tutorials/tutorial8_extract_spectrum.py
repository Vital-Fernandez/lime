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

# spec.plot.bands('H1_6563A')


# # Read text file
# import numpy as np
# import lime
# fname = '../sample_data/spectra/SHOC579_data.txt'
# fname = '/home/vital/Desktop/sfg_template_zmin0.002_zmax0.960_flux_nanmean.txt'
# fname = '/home/vital/Desktop/59.txt'
#
# # Load the data with whitespace separation
# data = np.loadtxt(fname)
#
# n_rows = data.shape[0]
# nan_columns = np.full((n_rows, 3), np.nan)
# data = np.hstack((nan_columns, data))
#
# np.savetxt('/home/vital/Desktop/59.csv', data, delimiter=';', fmt='%.6f')
#
# # Save the data using ';' as the delimiter
# spec = lime.Spectrum.from_file('/home/vital/Desktop/59.csv', instrument='text', delimiter=';', usecols=(3,4), redshift=0)
# spec.plot.spectrum(rest_frame=True)