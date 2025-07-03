from pathlib import Path
import numpy as np
import lime
from lime.transitions import Transition, Line, label_decomposition, Particle, format_line_mask_option


# Data for the tests
baseline_folder = Path(__file__).parent / 'baseline'
outputs_folder = Path(__file__).parent / 'outputs'

file_address = baseline_folder/'manga_spaxel.txt'
wave_array, flux_array, err_array = np.loadtxt(file_address, unpack=True)
pixel_mask = np.isnan(err_array)


def test_label_decomposition():

    # Components configuration
    fit_conf = {'O3_5007A_b': 'O3_5007A+O3_5007A_k-1+He1_5016A',
                'O3_5007A_m': 'O3_5007A+O3_5007A_k-1'}

    particle, wavelength, latex = label_decomposition('O3_5007A', fit_conf=fit_conf)
    assert particle[0] == 'O3'
    assert wavelength[0] == 5006.77
    assert latex[0] == '$[OIII]5007\\mathring{A}$'

    particle, wavelength, latex = label_decomposition('O3_5007A_b', fit_conf=fit_conf)
    assert particle[0] == 'O3'
    assert wavelength[0] == 5006.77
    assert latex[0] == '$[OIII]5007\\mathring{A}$'

    particle, wavelength, latex = label_decomposition('O3_5007A_m', fit_conf=fit_conf)
    assert particle[0] == 'O3'
    assert wavelength[0] == 5006.77
    assert latex[0] == '$[OIII]5007\\mathring{A}$+$[OIII]5007\\mathring{A}-k_1$'

    return


def test_format_line_mask_option():

    array1 = format_line_mask_option('5000-5009', wave_array)
    array2 = format_line_mask_option('5000-5009,5876,6550-6570', wave_array)

    assert np.all(array1[0] == np.array([5000, 5009.]))

    assert np.all(array2[0] == np.array([5000, 5009.]))
    assert np.allclose(array2[1], np.array([5875.26214276, 5876.73785724]))
    assert np.all(array2[2] == np.array([6550, 6570.]))

    return


def tests_bands_from_log():

    # Declare the data
    bands_df = lime.load_frame(baseline_folder/'manga_line_bands.txt')
    log_df = lime.load_frame(baseline_folder/'manga_lines_log.txt')

    # Make the transformation
    rename_dict = {'Fe3_4658A_p-g-emi_b': 'Fe3_4658A_b'}
    bands_new = lime.bands_from_measurements(log_df, index_dict=rename_dict)

    # Compare the rows and columns
    assert (set(bands_new.loc[:, 'w1':'w6'].columns) == set(bands_df.columns))  # True
    assert (set(bands_new.index) == set(bands_df.index))  # True

    return


class TestLineClass:

    def test_single_merged_blended(self):

        # Components configuration
        fit_conf = {'O3_5007A_b': 'O3_5007A+O3_5007A_k-1+He1_5016A',
                    'O3_5007A_m': 'O3_5007A+O3_5007A_k-1'}

        # Default band O3_5007A
        O3_band = np.array([4971.796688, 4984.514249, 5000.089685, 5013.450315, 5027.743260, 5043.797081])

        line = Line('O3_5007A', fit_conf=fit_conf)
        assert line.particle[0].label == 'O3'
        assert line.particle[0].symbol == 'O', line.particle[0].ionization == 3
        assert np.all(line.wavelength == np.array([5006.77]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == np.array(['col']))
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array(['$[OIII]5007\mathring{A}$']))
        assert np.all(line.list_comps == ['O3_5007A'])

        assert line.label == 'O3_5007A'
        assert line.group_label is None
        assert np.all(line.mask == O3_band)

        line = Line('O3_5007A_b', band=None, fit_conf=fit_conf)
        assert np.all(line.particle == [Particle.from_label('O3'), Particle.from_label('O3'), Particle.from_label('He1')])
        assert np.allclose(line.wavelength, np.array([5006.77, 5006.77, 5015.6]))
        assert np.all(line.kinem == np.array([0, 1, 0]))
        assert np.all(line.transition_comp == np.array(['col', 'col', 'rec']))
        assert np.all(line.profile_comp == np.array(['g-emi', 'g-emi', 'g-emi']))
        assert np.all(line.latex_label == np.array(['$[OIII]5007\mathring{A}$',
                                                    '$[OIII]5007\mathring{A}-k_1$',
                                                    '$HeI5016\mathring{A}$']))
        assert np.all(line.list_comps == ['O3_5007A', 'O3_5007A_k-1', 'He1_5016A'])

        assert line.label == 'O3_5007A_b'
        assert line.group_label == 'O3_5007A+O3_5007A_k-1+He1_5016A'
        assert np.all(line.mask == O3_band)

        line = Line('O3_5007A_m', band=np.array([1, 2, 3, 4, 5, 6]), fit_conf=fit_conf)
        assert np.all(line.particle == [Particle('O3', symbol='O', ionization=3)])
        assert np.all(line.wavelength == np.array([5007]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == np.array(['col']))
        assert np.all(line.profile_comp == np.array(['g-emi', 'g-emi']))
        assert np.all(line.latex_label == np.array([r'$[OIII]5007\mathring{A}$+$[OIII]5007\mathring{A}-k_1$']))
        assert np.all(line.list_comps == ['O3_5007A_m'])
        assert np.all(line.mask == np.array([1, 2, 3, 4, 5, 6]))

        assert line.label == 'O3_5007A_m'
        assert line.group_label == 'O3_5007A+O3_5007A_k-1'
        assert np.all(line.mask == np.array([1, 2, 3, 4, 5, 6]))

        return

    def test_special_particles(self):

        line = Line('OIII_5007A')
        assert line.particle[0].label == 'OIII'
        assert np.all(line.wavelength == np.array([5007.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == [None])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array(['OIII-$5007\\mathring{A}$']))
        assert np.all(line.list_comps == ['OIII_5007A'])

        assert line.label == 'OIII_5007A'
        assert line.group_label is None
        assert line.mask is None

        line = Line('Halpha_6563A')
        assert line.particle[0].label == 'Halpha'
        assert np.all(line.wavelength == np.array([6563.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == [None])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array(['Halpha-$6563\\mathring{A}$']))
        assert np.all(line.list_comps == ['Halpha_6563A'])

        assert line.label == 'Halpha_6563A'
        assert line.group_label is None
        assert line.mask is None

        line = Line('HIPas4-3_18751A')
        assert line.particle[0].label == 'HIPas4-3'
        assert np.all(line.wavelength == np.array([18751.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == [None])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array(['HIPas4-3-$18751\\mathring{A}$']))
        assert np.all(line.list_comps == ['HIPas4-3_18751A'])

        assert line.label == 'HIPas4-3_18751A'
        assert line.group_label is None
        assert line.mask is None

        line = Line('OIII_5007A')
        assert line.particle[0].label == 'OIII'
        assert np.all(line.wavelength == np.array([5007.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == [None])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array(['OIII-$5007\\mathring{A}$']))
        assert np.all(line.list_comps == ['OIII_5007A'])

        assert line.label == 'OIII_5007A'
        assert line.group_label is None
        assert line.mask is None

        line = Line('C3_1909A')
        assert line.particle[0].label == 'C3'
        assert np.all(line.wavelength == np.array([1908.734]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == ['sem'])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array(['$CIII]1909\\mathring{A}$']))
        assert np.all(line.list_comps == ['C3_1909A'])

        assert line.label == 'C3_1909A'
        assert line.group_label is None
        assert line.mask is not None

        line = Line('C3_1909A_t-sem', band=None)
        assert line.particle[0].label == 'C3'
        assert np.all(line.wavelength == np.array([1908.734]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == ['sem'])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array(['$CIII]1909\\mathring{A}$']))
        assert np.all(line.list_comps == ['C3_1909A_t-sem'])

        assert line.label == 'C3_1909A_t-sem'
        assert line.group_label is None

        assert np.allclose(line.mask, np.array([1870.000000, 1895.000000, 1906.187259, 1911.280741, 1930.000000, 1950.000000]))

        return


class TestTransitionClass:

    def test_single_merged_blended(self):

        # Components configuration
        fit_conf = {'O3_5007A_b': 'O3_5007A+O3_5007A_k-1+He1_5016A',
                    'O3_5007A_m': 'O3_5007A+O3_4959A',
                    'H1_6563A_m': 'H1_6563A+N2_6583A+N2_6548A+S2_6722A_m',
                    'S2_6722A_m': 'S2_6716A+S2_6731A',
                    'transitions': {'S2_6722A_m': {'particle': 'S2',
                                                   'wavelength': 6722.5,
                                                   'units_wave': 'AA'}}
                    }

        # Default band O3_5007A
        O3_band = np.array([4971.796688, 4984.514249, 5000.089685, 5013.450315, 5027.743260, 5043.797081])

        # line = Transition.from_db('O3_5007A', fit_cfg=fit_conf)
        # assert line.label == 'O3_5007A'
        # assert line.particle.label == 'O3'
        # assert line.particle.symbol == 'O', line.particle.ionization == 3
        # assert line.wavelength == 5006.77
        # assert line.kinem == 0
        # assert line.transition_comp == 'col'
        # assert line.profile_comp == 'g-emi'
        # assert line.latex_label == '$[OIII]5007\mathring{A}$'
        # assert line.list_comps == ['O3_5007A']
        # assert line.group_label is None
        # assert np.all(line.mask == O3_band)
        # assert line.measurements is None
        # assert line.list_comps[0].measurements is None

        # Blended line
        line = Transition.from_db('O3_5007A_b', fit_cfg=fit_conf)

        # The first component
        assert line.label == 'O3_5007A_b'
        assert line.particle.label == 'O3'
        assert line.particle.symbol == 'O', line.particle.ionization == 3
        assert line.wavelength == 5006.77
        assert line.kinem == 0
        assert line.transition_comp == 'col'
        assert line.profile_comp == 'g-emi'
        assert np.all(line.list_comps == ['O3_5007A', 'O3_5007A_k-1', 'He1_5016A'])
        assert line.group_label == 'O3_5007A+O3_5007A_k-1+He1_5016A'
        assert np.all(line.mask == O3_band)
        assert line.latex_label == '$[OIII]5007\mathring{A}$+$[OIII]5007\mathring{A}_k-1$+$HeI5016\mathring{A}$'

        # All components
        assert np.all(line.param_arr('particle') == [Particle.from_label('O3'), Particle.from_label('O3'), Particle.from_label('He1')])
        assert np.allclose(line.param_arr('wavelength'), np.array([5006.77, 5006.77, 5015.6]))
        assert np.all(line.param_arr('kinem') == np.array([0, 1, 0]))
        assert np.all(line.param_arr('transition_comp') == np.array(['col', 'col', 'rec']))
        assert np.all(line.param_arr('profile_comp') == np.array(['g-emi', 'g-emi', 'g-emi']))
        assert np.all(line.param_arr('latex_label') == np.array(['$[OIII]5007\mathring{A}$',
                                                                 '$[OIII]5007\mathring{A}_k-1$',
                                                                 '$HeI5016\mathring{A}$']))

        # Merged line
        line = Transition.from_db('O3_5007A_m', fit_cfg=fit_conf)
        assert line.label == 'O3_5007A_m'
        assert np.all(line.param_arr('particle') == [Particle.from_label('O3'), Particle.from_label('O3')])
        assert np.allclose(line.param_arr('wavelength'), np.array([5006.77, 4958.835]))
        assert np.all(line.param_arr('kinem') == np.array([0, 0]))
        assert line.group_label == 'O3_5007A+O3_4959A'
        assert np.all(line.param_arr('profile_comp') == np.array(['g-emi', 'g-emi']))

        line = Transition.from_db('S2_6722A_m', fit_cfg=fit_conf)
        assert line.label == 'S2_6722A_m'
        assert line.particle == Particle.from_label('S2')
        assert line.wavelength == 6722.5
        assert line.units_wave == 'AA'
        assert np.all(line.param_arr('wavelength') == [6716.33, 6730.71])
        assert line.latex_label == '$[SII]6716\\mathring{A}$+$[SII]6731\\mathring{A}$'
        assert np.all(line.param_arr('latex_label') == ["$[SII]6716\\mathring{A}$", "$[SII]6731\\mathring{A}$"])

        line = Transition.from_db('H1_6563A_m', fit_cfg=fit_conf)
        assert line.label == 'H1_6563A_m'
        assert line.particle == Particle.from_label('H1')
        assert line.wavelength == 6562.7
        assert np.all(line.param_arr('wavelength') == [6562.7 , 6583.36, 6547.94, 6722.5 ])
        assert line.latex_label == '$HI6563\\mathring{A}$+$[NII]6583\\mathring{A}$+$[NII]6548\\mathring{A}$+$[SII]6716\\mathring{A}$+$[SII]6731\\mathring{A}$'
        assert line.list_comps[3].wavelength == 6722.5
        assert line.list_comps[3].list_comps == ['S2_6716A', 'S2_6731A']
        assert line.list_comps[3].list_comps[0].wavelength == 6716.33

        # Line with measurements
        line = Transition.from_db('O3_5007A_b', fit_cfg=fit_conf, init_measurements=True)
        assert line.measurements is not None
        assert line.measurements.observations == 'no'
        assert line.list_comps[0].measurements is None
        assert line.list_comps[1].measurements is None

        return

    def test_special_particles(self):

        line = Line('OIII_5007A')
        assert line.particle[0].label == 'OIII'
        assert np.all(line.wavelength == np.array([5007.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == [None])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array(['OIII-$5007\\mathring{A}$']))
        assert np.all(line.list_comps == ['OIII_5007A'])

        assert line.label == 'OIII_5007A'
        assert line.group_label is None
        assert line.mask is None

        line = Line('Halpha_6563A')
        assert line.particle[0].label == 'Halpha'
        assert np.all(line.wavelength == np.array([6563.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == [None])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array(['Halpha-$6563\\mathring{A}$']))
        assert np.all(line.list_comps == ['Halpha_6563A'])

        assert line.label == 'Halpha_6563A'
        assert line.group_label is None
        assert line.mask is None

        line = Line('HIPas4-3_18751A')
        assert line.particle[0].label == 'HIPas4-3'
        assert np.all(line.wavelength == np.array([18751.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == [None])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array(['HIPas4-3-$18751\\mathring{A}$']))
        assert np.all(line.list_comps == ['HIPas4-3_18751A'])

        assert line.label == 'HIPas4-3_18751A'
        assert line.group_label is None
        assert line.mask is None

        line = Line('OIII_5007A')
        assert line.particle[0].label == 'OIII'
        assert np.all(line.wavelength == np.array([5007.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == [None])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array(['OIII-$5007\\mathring{A}$']))
        assert np.all(line.list_comps == ['OIII_5007A'])

        assert line.label == 'OIII_5007A'
        assert line.group_label is None
        assert line.mask is None

        line = Line('C3_1909A')
        assert line.particle[0].label == 'C3'
        assert np.all(line.wavelength == np.array([1908.734]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == ['sem'])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array(['$CIII]1909\\mathring{A}$']))
        assert np.all(line.list_comps == ['C3_1909A'])

        assert line.label == 'C3_1909A'
        assert line.group_label is None
        assert line.mask is not None

        line = Line('C3_1909A_t-sem', band=None)
        assert line.particle[0].label == 'C3'
        assert np.all(line.wavelength == np.array([1908.734]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == ['sem'])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array(['$CIII]1909\\mathring{A}$']))
        assert np.all(line.list_comps == ['C3_1909A_t-sem'])

        assert line.label == 'C3_1909A_t-sem'
        assert line.group_label is None

        assert np.allclose(line.mask, np.array([1870.000000, 1895.000000, 1906.187259, 1911.280741, 1930.000000, 1950.000000]))

        return


class TestParticleClass:

    def test_pyneb_items(self):

        particle = Particle.from_label('H1')

        assert particle.label == 'H1'
        assert particle.ionization == 1
        assert particle.symbol == 'H'

        particle2 = Particle('H1', 'H', ionization=1)
        assert particle2.label == 'H1'
        assert particle2.ionization == 1
        assert particle2.symbol == 'H'

        return
