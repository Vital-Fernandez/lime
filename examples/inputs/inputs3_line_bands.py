import numpy as np
from astropy.io import fits
from pathlib import Path
import lime

# Complete lines database with the default format
bands_df = lime.line_bands()

# Limited selection from the default database with magnitudes conversion
print(lime.line_bands(wave_intvl=(300, 900), particle_list=('He1', 'O3', 'S2'),
                      units_wave='nm', decimals=None, vacuum=True))

# Recovering columns from a dataframe
print(bands_df.columns)

labels = bands_df.index.to_numpy()
ions = bands_df['particle'].to_numpy()
wave_array = bands_df.wavelength.to_numpy()
print(labels, ions, wave_array)

# Getting rows from this dataframe
print(bands_df.index.to_numpy())

H1_1215A_params = bands_df.iloc[0].to_numpy()
H1_4861A_params = bands_df.loc['H1_4861A'].to_numpy()
print(H1_1215A_params, H1_4861A_params)

# Getting cell values
print(bands_df.at['H1_1215A', 'wavelength'])
print(bands_df.loc['H1_1215A', 'wavelength'], bands_df.loc['H1_1215A'].wavelength)
print(bands_df.loc[['H1_1215A', 'H1_4861A'], 'wavelength'].to_numpy())
print(bands_df.loc['H1_1215A':'He2_1640A', 'wavelength'].to_numpy())
print(bands_df.loc['H1_1215A', 'w1':'w6'].to_numpy())

# Save to the current folder in several formats:
lime.save_log(bands_df, 'bands_frame.txt')
lime.save_log(bands_df, 'bands_frame.pdf', parameters=['wavelength', 'latex_label'])
lime.save_log(bands_df, 'bands_frame.xlsx', page='BANDS')
lime.save_log(bands_df, 'bands_frame.fits', page='BANDS')

# Load the database into a pandas daframe
bands_df = lime.load_log('bands_frame.txt')
bands_df = lime.load_log('bands_frame.xlsx', page='BANDS')
bands_df = lime.load_log('bands_frame.fits', page='BANDS')
