import lime
from lime.transitions import Line

# List of line strings
conf_dict = {'O3_5007A':                None,
             'O3_5007A_t-col':          None,
             'O3_5007A_b':              None,
             "O3_5007.89A_b":           "O3_5007A+O3_5007A_k-1+O3_5007A_k-2",

             "C3_1909A":                None,
             "C3_1909A_t-sem":          None,

             'H1_6563A':                None,
             'H1_6563A_b':              "H1_6563A+N2_6583A+N2_6548A",
             'H1_6563.356A_m':          'H1_6563.356A+N2_6583A+N2_6548A',
             'H1_6563.356A_p-g_mix':    'H1_6563.356A+N2_6583A+N2_6548A',
             'Halpha_6563.56A_b':       "H1_6563A+N2_6583A+N2_6548A",

             'sky_8600A':               None,
             'H1-PashAlpha_18751A':     None}

# Proceed to the conversion
line_list = list(conf_dict.keys())
for line in line_list:

    # Create LiMe line object
    print(f'\n- {line} = {conf_dict[line]}')
    line_obj = Line(line, fit_conf=conf_dict)

    # Create
    for i, comp in enumerate(line_obj.list_comps):
        print(f'{comp} ({line_obj.latex_label[i]}): {line_obj.particle[i]}, {line_obj.wavelength[i]}, {line_obj.units_wave[i]},'
              f' {line_obj.kinem[i]}, {line_obj.profile_comp[i]}, {line_obj.transition_comp[i]} ')

particle_array, wave_array, latex_array = lime.label_decomposition(line_list)
for i, particle in enumerate(particle_array):
    print(f'{i}) {particle}, {wave_array[i]}, {latex_array[i]}')

wave, latex = lime.label_decomposition('H1_6563A', scalar_output=True, params_list=('wavelength', 'latex_label'))
print(f'\nIndividual line components: {wave, latex}')
