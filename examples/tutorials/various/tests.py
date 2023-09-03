import pandas as pd
import lime
from pathlib import Path

# Load the lines log as a dataframe
ceers_log_file = Path(f'/home/usuario/PycharmProjects/lime_online/data/tables/fluxes_log.txt')
log_df = lime.load_log(ceers_log_file, levels=['sample', 'id', 'line'])

# Rename gauss_flux_err to match the new lime format
log_df.rename(columns={'gauss_err': 'gauss_flux_err'}, inplace=True)

# Create new column with relative fluxes (depending on the user inputs)
line_list_norm = ['O3_5008A/H1_4862A', 'N2_6550A/H1_6565A']
lime.normalize_fluxes(log_df, line_list=line_list_norm, column_name='line_flux')

# Get observations which contain all the lines
target_lines = ['O3_5008A', 'H1_4862A', 'N2_6550A', 'H1_6565A']
idcs_slice = log_df.index.get_level_values('line').isin(target_lines)
grouper = log_df.index.droplevel('line')
idcs_slice = pd.Series(idcs_slice).groupby(grouper).transform('sum').ge(len(target_lines)).array
slice_log = log_df.loc[idcs_slice]

# Compute the ratios
O3_ratio = slice_log.xs('O3_5008A', level='line').line_flux
N2_ratio = slice_log.xs('N2_6550A', level='line').line_flux


# with asdf.open(log_path, mode='rw') as af:
#     af.tree.update(tree)
#     af.update()

# import fiasco
# import numpy as np
# from fiasco import Ion
# import astropy.units as u
#
# temperature = np.logspace(5, 7, 100) * u.K
# ion = Ion('O 3', temperature)
# print(ion)
#
# Te = np.geomspace(0.1, 100, 51) * u.MK
# ne = 1e8 * u.cm**-3
#
# ion = Ion('O 5+', Te)
# print(ion)
#
# contribution_func = ion.contribution_function(ne)
# wlen = 1031.93 * u.Angstrom

# class Person:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#
#     @classmethod
#     def create_from_birth_year(cls, name, birth_year):
#         current_year = 2023  # Assuming the current year is 2023
#         age = current_year - birth_year
#         return cls(name, age)
#
#     @classmethod
#     def create_from_dict(cls, person_dict):
#         name = person_dict['name']
#         age = person_dict['age']
#         return cls(name, age)
#
#     def display(self):
#         print(f"Name: {self.name}, Age: {self.age}")
#
#
# # Creating objects using different class methods
# person1 = Person.create_from_birth_year('John', 1990)
# person2 = Person.create_from_dict({'name': 'Alice', 'age': 25})
#
# # Displaying the objects
# person1.display()  # Output: Name: John, Age: 33
# person2.display()  # Output: Name: Alice, Age: 25


# os.chdir(f'../../../')
#
# def launch_jupyter_notebook():
#     subprocess.run(["jupyter", "notebook"])
#
# if __name__ == "__main__":
#     launch_jupyter_notebook()

# import lime
# from pathlib import Path
#
# conf_file_toml = Path(f'../sample_data/manga.toml')
#
# cfg_ini = lime.load_cfg(conf_file_ini)
# cfg_toml = lime.load_cfg(conf_file_toml)
#
# print(cfg_ini)
# print(cfg_toml)


import numpy as np
from astropy.io import fits
from pathlib import Path
from astropy.wcs import WCS
import lime

# fits_path = Path(f'../sample_data/SHOC579_SDSS_dr18.fits')
#
# # Open the fits file
# extension = 2
# with fits.open(fits_path) as hdul:
#     data = hdul[extension].data
#     header = hdul[extension].header
#
# flux_array = data['flux'] * 1e-17
#
# ivar_array = data['ivar']
# pixel_mask = ivar_array == 0
#
# wave_vac_array = np.power(10, data['loglam'])
#
# spectra_path = Path('../../sample_data')
# fits_path = spectra_path/'manga-8626-12704-LOGCUBE.fits.gz'
#
# # Open the MANGA cube fits file
# with fits.open(fits_path) as hdul:
#
#     # Wavelength 1D array
#     wave = hdul['WAVE'].data
#
#     # Flux 3D array
#     flux_cube = hdul['FLUX'].data * 1e-17
#
#     # Convert inverse variance cube to standard error masking 0-value pixels first
#     ivar_cube = hdul['IVAR'].data
#     pixel_mask_cube = ivar_cube == 0
#     pixel_mask_cube = pixel_mask_cube.reshape(ivar_cube.shape)
#     err_cube = np.sqrt(1/np.ma.masked_array(ivar_cube, pixel_mask_cube)) * 1e-17
#
#     # Header
#     hdr = hdul['FLUX'].header
#
#
# wcs = WCS(hdr)
#
# cube = lime.Cube(wave, flux_cube, err_cube, redshift=0.0475, norm_flux=1e-17, pixel_mask=pixel_mask_cube, wcs=wcs)
# spaxel = cube.get_spectrum(38, 35)
# spaxel.plot.spectrum(rest_frame=True)
# spaxel.fit.band('S3_6312A')
# spaxel.plot.band()
# spaxel.con