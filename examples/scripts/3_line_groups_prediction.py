import lime
from pathlib import Path
import numpy as np
from lime.retrieve.line_bands import get_spectrum_line_groups, blend_merge_dict




fname = '/home/vital/Dropbox/Astrophysics/Tools/SpectralSynthesis/Online_example_data/sdss_dr18_0358-51818-0504.fits'
bands_file = '/home/vital/Dropbox/Astrophysics/Tools/SpectralSynthesis/Online_example_data/sdss_dr18_0358-51818-0504.fits_bands_df.txt'
bands = lime.load_frame(bands_file)

spec = lime.Spectrum.from_file(fname, instrument='sdss')

grouped_lines = get_spectrum_line_groups(spec.wave_rest.data, bands)
for key, value in grouped_lines.items():
    print(f'{key}={value}')

# relation_list = []
# for group in grouped_lines:
#     idcs = bands.index.isin(group)
#     w3_arr = idx3_arr[idcs]
#     w4_arr = idx4_arr[idcs]
#     mu_arr = np.searchsorted(spec.wave_rest.data, bands.loc[group, 'wavelength'].to_numpy())
#     obs_group_blend_chek = deblend_criteria(mu_arr=mu_arr,
#                                             sigma_arr=(w4_arr - w3_arr) / (n_sigma * 2),
#                                             Rayleigh_threshold=Rayleigh_threshold)
#     relation_list.append(obs_group_blend_chek)
#
# print(relation_list)
# for i, group in enumerate(grouped_lines):
#     print(i, group, '->', relation_list[i], '->', blend_merge_dict(group, relation_list[i]))



# Quick checks
lines = np.array(['H1_3722A', 'O2_3726A', 'O2_3729A', 'H1_3734A'])

print(blend_merge_dict(lines, np.array([True, True, True])))
# {'H1_3722A_b': 'H1_3722A+O2_3726A+O2_3729A+H1_3734A'}

print(blend_merge_dict(lines, np.array([False, False, False])))
# {'H1_3722A_m': 'H1_3722A+O2_3726A+O2_3729A+H1_3734A'}

print(blend_merge_dict(np.array(['A','B','C','D']), np.array([True, False, True])))
# {'A_b': 'A+B_m+D', 'B_m': 'B+C'}

print(blend_merge_dict(np.array(['A','B','C','D','E']), np.array([True, False, False, True])))
# {'A_b': 'A+B_m+E', 'B_m': 'B+C+D'}

'''
list_lines = ['H1_3722A', 'O2_3726A', 'O2_3729A', 'H1_3734A'], relation_list = [ True  True  True]

I need you to make me a function which will take two inputs such as the ones above:
this function will generate a dictionary with certain keys and a values based on the following logic

1) each item in the relation_list represent the connection between the lines in list_lines. True means the two subsequent two lines are blended and false
it meas they are merged. 
2) If all the entries are True I would like the output key to be the first item in the line_list with the '_b' suffix and the values is joining all the list lines into a string with a '+'
3) If all the entries are false I would like the output key to be the first item in the line_list with the '_m' suffix and the values is joining all the list lines into a string with a '+'
4) If it is a mixture of true and false values the dictionary will have several new entries:
    4.1) For every continuous false entries (merged) I want a key with the the value of the first line in the merged subgroup with the '_m' suffix and as the value I want the lines in the subgroug as a string joined by '+'
    4.2) Also you must add an additional entry to the dictionary where the key is the first line in the group with the suffix '_b'. 
    The value for this entry is the list of all the lines joined by '+' but replace the merged lines by the previous 
    merged group key definition with the '_m'. It is important that in the output dictionary this entry is before the others. 

following logic:
each entry in the relation list 

if the input is relation_list=[True, False, True] with 4 lines line_list=[A, B, C, D] the right output dictionary would
be = {A_b = "A+B_m+D", B_m = "B+C"}





'''