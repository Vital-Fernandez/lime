import lime
from pathlib import Path

# Folder to save the table
data_folder = Path('../sample_data/bands')
fname = data_folder/'lines_star_forming_galaxies_optical.txt'

# Get the default database limited to the optical range (3000, 10000) angstroms
lime_database_optical = lime.line_bands((3000, 10000))
print(lime_database_optical.index)

# Exclude some weak lines unlikely to be observed in emission line galaxies
idcs_exclude = lime_database_optical.rel_int == 1
lime_database_optical = lime_database_optical.loc[idcs_exclude]

# Save the table
lime.save_frame(fname, lime_database_optical)