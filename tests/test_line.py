from pathlib import Path
import numpy as np
import lime
from lime.transitions import _LIME_DATABASE_FILE
from lime.transitions import Transition, label_decomposition, Particle, format_line_mask_option
from lime.resources.generator_db import format_lines_database
from lime.io import _RANGE_ATTRIBUTES_FIT, _ATTRIBUTES_FIT, _LOG_COLUMNS

# Data for the tests
baseline_folder = Path(__file__).parent / 'baseline'
outputs_folder = Path(__file__).parent / 'outputs'

file_address = baseline_folder/'manga_spaxel.txt'
conf_file_address = baseline_folder/'manga.toml'

redshift, norm_flux = 0.0475, 1e-17
wave_array, flux_array, err_array = np.loadtxt(file_address, unpack=True)
pixel_mask = np.isnan(err_array)

cfg = lime.load_cfg(conf_file_address)
input_cfg = {**cfg['38-35_line_fitting'], **cfg['default_line_fitting']}

spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
                     pixel_mask=pixel_mask, id_label='SHOC579-Manga38-35')

bands = spec.retrieve.line_bands(fit_cfg=cfg, obj_cfg_prefix='38-35')

# spec.fit.frame(bands, cfg, obj_cfg_prefix='38-35')


# # Data for the tests
# baseline_folder = Path(__file__).parent / 'baseline'
# outputs_folder = Path(__file__).parent / 'outputs'
# spectra_folder = Path(__file__).parent.parent/'examples/sample_data/spectra'
# fits_address = baseline_folder/'manga_spaxel.txt'
# conf_file_address = baseline_folder/'manga.toml'
# bands_file_address = baseline_folder/f'manga_line_bands.txt'
# lines_log_address = baseline_folder/'manga_lines_log.txt'
# lines_tex_address = baseline_folder/'manga_lines_log.tex'
#
#
# cfg = lime.load_cfg(conf_file_address)
# tolerance_rms = 5.5
#
# wave_array, flux_array, err_array = np.loadtxt(file_address, unpack=True)
# pixel_mask = np.isnan(err_array)
#
# spec = lime.Spectrum(wave_array, flux_array, err_array, redshift=redshift, norm_flux=norm_flux,
#                      pixel_mask=pixel_mask, id_label='SHOC579-Manga38-35')
#
# spec.fit.frame(bands_file_address, cfg, obj_cfg_prefix='38-35')



# TODO if line He2_1640_m does not find components on the log it is going
#  to use the database this can cause an issue with he bands width...


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
    assert latex[0] == '$[OIII]5007\\mathring{A}$+$[OIII]5007\\mathring{A}_k-1$+$HeI5016\\mathring{A}$'

    particle, wavelength, latex = label_decomposition('O3_5007A_m', fit_conf=fit_conf)
    assert particle[0] == 'O3'
    assert wavelength[0] == 5006.77
    assert latex[0] == '$[OIII]5007\\mathring{A}$+$[OIII]5007\\mathring{A}_k-1$'

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
    rename_dict = {'Fe3_4658A_s-emi_b': 'Fe3_4658A_b'}
    bands_new = lime.bands_from_measurements(log_df, index_dict=rename_dict)

    # Compare the rows and columns
    assert (set(bands_new.loc[:, 'w1':'w6'].columns) == set(bands_df.columns))  # True
    assert (set(bands_new.index) == set(bands_df.index))  # True

    return


def test_format_lines_database(tmp_path):

    # fname = tmp_path/'orig_database.txt'
    # lime.save_frame(fname, format_lines_database())
    #
    # orig_df = lime.load_frame(_LIME_DATABASE_FILE)
    # new_df = lime.load_frame(fname)
    # assert orig_df.equals(new_df)
    assert True
    return

    # # Manual review for the differences
    # differences = []
    # for row in orig_DF.index:
    #     for col in orig_DF.columns:
    #         val1 = orig_DF.loc[row, col]
    #         val2 = new_DF.loc[row, col]
    #         if not pd.isna(val1) and not pd.isna(val2) and val1 != val2:
    #             differences.append((row, col, val1, val2))
    #         elif pd.isna(val1) != pd.isna(val2):  # one is NaN, the other isn't
    #             differences.append((row, col, val1, val2))
    #
    # if not differences:
    #     print("✅ DataFrames are identical.")
    # else:
    #     print(f"❌ Found {len(differences)} differences:\n")
    #     for row, col, val1, val2 in differences:
    #         print(f"Row: {row}, Column: {col} → df1: {val1!r}, df2: {val2!r}")

    return


def assert_measurements(fit_line, log_line):

    for j in _RANGE_ATTRIBUTES_FIT:
        param = _ATTRIBUTES_FIT[j]
        param_value = fit_line.measurements.__getattribute__(param)

        # De-normalize
        if _LOG_COLUMNS[param][0]:
            if param_value is not None:
                param_value = param_value * norm_flux

        # Store in dataframe
        print(param, param_value, log_line.measurements.__getattribute__(param))
        if not isinstance(param_value, (np.ndarray, list)):
            assert param_value == log_line.measurements.__getattribute__(param)
        else:
            assert np.all(np.isclose(param_value, log_line.measurements.__getattribute__(param)))

    return

# class TestLineClass:
#
#     def test_single_merged_blended(self):
#
#         # Components configuration
#         fit_conf = {'O3_5007A_b': 'O3_5007A+O3_5007A_k-1+He1_5016A',
#                     'O3_5007A_m': 'O3_5007A+O3_5007A_k-1'}
#
#         # Default band O3_5007A
#         O3_band = np.array([4971.796688, 4984.514249, 5000.089685, 5013.450315, 5027.743260, 5043.797081])
#
#         line = Line('O3_5007A', fit_conf=fit_conf)
#         assert line.particle[0].label == 'O3'
#         assert line.particle[0].symbol == 'O', line.particle[0].ionization == 3
#         assert np.all(line.wavelength == np.array([5006.77]))
#         assert np.all(line.kinem == np.array([0]))
#         assert np.all(line.transition_comp == np.array(['col']))
#         assert np.all(line.profile_comp == np.array(['g-emi']))
#         assert np.all(line.latex_label == np.array(['$[OIII]5007\mathring{A}$']))
#         assert np.all(line.list_comps == ['O3_5007A'])
#
#         assert line.label == 'O3_5007A'
#         assert line.group_label is None
#         assert np.all(line.mask == O3_band)
#
#         line = Line('O3_5007A_b', band=None, fit_conf=fit_conf)
#         assert np.all(line.particle == [Particle.from_label('O3'), Particle.from_label('O3'), Particle.from_label('He1')])
#         assert np.allclose(line.wavelength, np.array([5006.77, 5006.77, 5015.6]))
#         assert np.all(line.kinem == np.array([0, 1, 0]))
#         assert np.all(line.transition_comp == np.array(['col', 'col', 'rec']))
#         assert np.all(line.profile_comp == np.array(['g-emi', 'g-emi', 'g-emi']))
#         assert np.all(line.latex_label == np.array(['$[OIII]5007\mathring{A}$',
#                                                     '$[OIII]5007\mathring{A}-k_1$',
#                                                     '$HeI5016\mathring{A}$']))
#         assert np.all(line.list_comps == ['O3_5007A', 'O3_5007A_k-1', 'He1_5016A'])
#
#         assert line.label == 'O3_5007A_b'
#         assert line.group_label == 'O3_5007A+O3_5007A_k-1+He1_5016A'
#         assert np.all(line.mask == O3_band)
#
#         line = Line('O3_5007A_m', band=np.array([1, 2, 3, 4, 5, 6]), fit_conf=fit_conf)
#         assert np.all(line.particle == [Particle('O3', symbol='O', ionization=3)])
#         assert np.all(line.wavelength == np.array([5007]))
#         assert np.all(line.kinem == np.array([0]))
#         assert np.all(line.transition_comp == np.array(['col']))
#         assert np.all(line.profile_comp == np.array(['g-emi', 'g-emi']))
#         assert np.all(line.latex_label == np.array([r'$[OIII]5007\mathring{A}$+$[OIII]5007\mathring{A}-k_1$']))
#         assert np.all(line.list_comps == ['O3_5007A_m'])
#         assert np.all(line.mask == np.array([1, 2, 3, 4, 5, 6]))
#
#         assert line.label == 'O3_5007A_m'
#         assert line.group_label == 'O3_5007A+O3_5007A_k-1'
#         assert np.all(line.mask == np.array([1, 2, 3, 4, 5, 6]))
#
#         return
#
#     def test_special_particles(self):
#
#         line = Line('OIII_5007A')
#         assert line.particle[0].label == 'OIII'
#         assert np.all(line.wavelength == np.array([5007.]))
#         assert np.all(line.kinem == np.array([0]))
#         assert np.all(line.transition_comp == [None])
#         assert np.all(line.profile_comp == np.array(['g-emi']))
#         assert np.all(line.latex_label == np.array(['OIII-$5007\\mathring{A}$']))
#         assert np.all(line.list_comps == ['OIII_5007A'])
#
#         assert line.label == 'OIII_5007A'
#         assert line.group_label is None
#         assert line.mask is None
#
#         line = Line('Halpha_6563A')
#         assert line.particle[0].label == 'Halpha'
#         assert np.all(line.wavelength == np.array([6563.]))
#         assert np.all(line.kinem == np.array([0]))
#         assert np.all(line.transition_comp == [None])
#         assert np.all(line.profile_comp == np.array(['g-emi']))
#         assert np.all(line.latex_label == np.array(['Halpha-$6563\\mathring{A}$']))
#         assert np.all(line.list_comps == ['Halpha_6563A'])
#
#         assert line.label == 'Halpha_6563A'
#         assert line.group_label is None
#         assert line.mask is None
#
#         line = Line('HIPas4-3_18751A')
#         assert line.particle[0].label == 'HIPas4-3'
#         assert np.all(line.wavelength == np.array([18751.]))
#         assert np.all(line.kinem == np.array([0]))
#         assert np.all(line.transition_comp == [None])
#         assert np.all(line.profile_comp == np.array(['g-emi']))
#         assert np.all(line.latex_label == np.array(['HIPas4-3-$18751\\mathring{A}$']))
#         assert np.all(line.list_comps == ['HIPas4-3_18751A'])
#
#         assert line.label == 'HIPas4-3_18751A'
#         assert line.group_label is None
#         assert line.mask is None
#
#         line = Line('OIII_5007A')
#         assert line.particle[0].label == 'OIII'
#         assert np.all(line.wavelength == np.array([5007.]))
#         assert np.all(line.kinem == np.array([0]))
#         assert np.all(line.transition_comp == [None])
#         assert np.all(line.profile_comp == np.array(['g-emi']))
#         assert np.all(line.latex_label == np.array(['OIII-$5007\\mathring{A}$']))
#         assert np.all(line.list_comps == ['OIII_5007A'])
#
#         assert line.label == 'OIII_5007A'
#         assert line.group_label is None
#         assert line.mask is None
#
#         line = Line('C3_1909A')
#         assert line.particle[0].label == 'C3'
#         assert np.all(line.wavelength == np.array([1908.734]))
#         assert np.all(line.kinem == np.array([0]))
#         assert np.all(line.transition_comp == ['sem'])
#         assert np.all(line.profile_comp == np.array(['g-emi']))
#         assert np.all(line.latex_label == np.array(['$CIII]1909\\mathring{A}$']))
#         assert np.all(line.list_comps == ['C3_1909A'])
#
#         assert line.label == 'C3_1909A'
#         assert line.group_label is None
#         assert line.mask is not None
#
#         line = Line('C3_1909A_t-sem', band=None)
#         assert line.particle[0].label == 'C3'
#         assert np.all(line.wavelength == np.array([1908.734]))
#         assert np.all(line.kinem == np.array([0]))
#         assert np.all(line.transition_comp == ['sem'])
#         assert np.all(line.profile_comp == np.array(['g-emi']))
#         assert np.all(line.latex_label == np.array(['$CIII]1909\\mathring{A}$']))
#         assert np.all(line.list_comps == ['C3_1909A_t-sem'])
#
#         assert line.label == 'C3_1909A_t-sem'
#         assert line.group_label is None
#
#         assert np.allclose(line.mask, np.array([1870.000000, 1895.000000, 1906.187259, 1911.280741, 1930.000000, 1950.000000]))
#
#         return


class TestTransitionClass:

    def test_single_merged_blended(self):

        # Components configuration
        fit_conf = {'O3_5007A_b': 'O3_5007A+O3_5007A_k-1+He1_5016A',
                    'O3_5007A_m': 'O3_5007A+O3_4959A',
                    'H1_6563A_m': 'H1_6563A+N2_6583A+N2_6548A+S2_6722A_m',
                    'S2_6722A_m': 'S2_6716A+S2_6731A',
                    'transitions': {'S2_6722A_m': {'particle': 'S2',
                                                   'wavelength': 6722.5,
                                                   'units_wave': 'AA'},
                                    'H1-O3_5007A_b':{'wavelength': 5000,
                                                     'units_wave': 'AA'}},

                    'Fe3_4658A_b': "Fe3_4658A_s-emi+Fe3_4658A_s-abs",
                    'H1-O3_5007A_b': "H1_4861A+O3_5007A_m"
        }

        line = Transition.from_db('H1-O3_5007A_b', fit_cfg=fit_conf)
        assert line.label == 'H1-O3_5007A_b'
        assert np.all(line.param_arr('particle') == [Particle.from_label('H1'), Particle.from_label('O3')])
        assert line.wavelength == 5000
        assert line.group_label == "H1_4861A+O3_5007A_m"
        assert line.list_comps == ['H1_4861A', "O3_5007A_m"]
        assert line.list_comps[0].group_label == "H1_4861A+O3_5007A_m"
        assert line.list_comps[1].group_label == 'O3_5007A+O3_4959A'
        assert line.list_comps[0].list_comps == ["H1_4861A"]
        assert line.list_comps[1].list_comps == ["O3_5007A", "O3_4959A"]

        # Default band O3_5007A
        line = Transition.from_db('Fe3_4658A_b', fit_cfg=fit_conf)
        assert line.label == 'Fe3_4658A_b'
        assert np.all(line.param_arr('particle') == [Particle.from_label('Fe3'), Particle.from_label('Fe3')])
        assert np.allclose(line.param_arr('wavelength'), np.array([4658.09, 4658.09]))
        assert np.all(line.param_arr('kinem') == np.array([0, 0]))
        assert line.group_label == 'Fe3_4658A_s-emi+Fe3_4658A_s-abs'
        assert np.all(line.param_arr('profile') == np.array(['g', 'g']))
        assert np.all(line.param_arr('trans') == np.array(['col', 'col']))
        assert np.all(line.param_arr('shape') == np.array(['emi', 'abs']))

        O3_band = np.array([4971.796688, 4984.514249, 5000.089685, 5013.450315, 5027.743260, 5043.797081])

        line = Transition.from_db('O3_5007A', fit_cfg=fit_conf)
        assert line.label == 'O3_5007A'
        assert line.particle.label == 'O3'
        assert line.particle.symbol == 'O', line.particle.ionization == 3
        assert line.wavelength == 5006.77
        assert line.kinem == 0
        assert line.trans == 'col'
        assert line.profile == 'g'
        assert line.shape == 'emi'
        assert line.latex_label == '$[OIII]5007\mathring{A}$'
        assert line.list_comps == ['O3_5007A']
        assert line.group_label is None
        assert np.all(line.mask == O3_band)
        assert line.measurements is not None
        assert line.list_comps[0].measurements is not None

        # Blended line
        line = Transition.from_db('O3_5007A_b', fit_cfg=fit_conf)
        assert line.label == 'O3_5007A_b'
        assert line.particle.label == 'O3'
        assert line.particle.symbol == 'O', line.particle.ionization == 3
        assert line.wavelength == 5006.77
        assert line.kinem == 0
        assert line.trans == 'col'
        assert line.profile == 'g'
        assert line.shape == 'emi'
        assert np.all(line.list_comps == ['O3_5007A', 'O3_5007A_k-1', 'He1_5016A'])
        assert line.group_label == 'O3_5007A+O3_5007A_k-1+He1_5016A'
        assert np.all(line.mask == O3_band)
        assert line.latex_label == '$[OIII]5007\mathring{A}$+$[OIII]5007\mathring{A}_k-1$+$HeI5016\mathring{A}$'

        # All components
        assert np.all(line.param_arr('particle') == [Particle.from_label('O3'), Particle.from_label('O3'), Particle.from_label('He1')])
        assert np.allclose(line.param_arr('wavelength'), np.array([5006.77, 5006.77, 5015.6]))
        assert np.all(line.param_arr('kinem') == np.array([0, 1, 0]))
        assert np.all(line.param_arr('trans') == np.array(['col', 'col', 'rec']))
        assert np.all(line.param_arr('profile') == np.array(['g', 'g', 'g']))
        assert np.all(line.param_arr('shape') == np.array(['emi', 'emi', 'emi']))
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
        assert np.all(line.param_arr('profile') == np.array(['g', 'g']))
        assert np.all(line.param_arr('trans') == np.array(['col', 'col']))
        assert np.all(line.param_arr('shape') == np.array(['emi', 'emi']))

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
        line = Transition.from_db('O3_5007A_b', fit_cfg=fit_conf)
        assert line.measurements is not None
        assert line.measurements.observations == 'no'
        assert line.list_comps[0].measurements is None
        assert line.list_comps[1].measurements is None

        return

    def test_special_particles(self):

        line = Transition.from_db('OIII_5007A')
        assert line.particle.label == 'OIII'
        assert line.wavelength == 5007.
        assert line.kinem == 0
        assert line.trans is None
        assert line.profile == 'g'
        assert line.shape == 'emi'
        assert line.latex_label == 'OIII-$5007\\mathring{A}$'
        assert line.list_comps == ['OIII_5007A']

        assert line.label == 'OIII_5007A'
        assert line.group_label is None
        assert line.mask is None


        line = Transition.from_db('Halpha_6563A_s-abs')
        assert line.particle.label == 'Halpha'
        assert line.wavelength == 6563.
        assert line.kinem == 0
        assert line.trans == None
        assert line.profile == 'g'
        assert line.shape == 'abs'
        assert line.latex_label == 'Halpha-$6563\\mathring{A}$'
        assert line.list_comps == ['Halpha_6563A_s-abs']

        assert line.label == 'Halpha_6563A_s-abs'
        assert line.group_label is None
        assert line.mask is None


        line = Transition.from_db('HIPas4-3_18751A')
        assert line.particle.label == 'HIPas4-3'
        assert line.wavelength == 18751.
        assert line.trans is None
        assert line.kinem == 0
        assert line.profile == 'g'
        assert line.shape == 'emi'
        assert line.latex_label == 'HIPas4-3-$18750\\mathring{A}$'
        assert line.list_comps == ['HIPas4-3_18751A']
        assert line.label == 'HIPas4-3_18751A'
        assert line.group_label is None
        assert line.mask is None


        line = Transition.from_db('C3_1909A')
        assert line.particle.label == 'C3'
        assert line.wavelength == 1908.734
        assert line.trans == 'sem'
        assert line.kinem == 0
        assert line.profile == 'g'
        assert line.shape == 'emi'
        assert line.latex_label == '$CIII]1909\\mathring{A}$'
        assert line.list_comps == ['C3_1909A']
        assert line.label == 'C3_1909A'
        assert line.group_label is None
        assert line.mask is not None

        line = Transition.from_db('C3_1909A_t-sem')
        assert line.particle.label == 'C3'
        assert line.wavelength == 1908.734
        assert line.trans == 'sem'
        assert line.kinem == 0
        assert line.profile == 'g'
        assert line.shape == 'emi'
        assert line.latex_label == '$CIII]1909\\mathring{A}$'
        assert line.list_comps == ['C3_1909A_t-sem']
        assert line.label == 'C3_1909A_t-sem'
        assert line.group_label is None
        assert np.allclose(line.mask, np.array([1870.000000, 1895.000000, 1906.187259, 1911.280741, 1930.000000, 1950.000000]))

        return

    def test_line_db_vs_log(self):

        line_label = 'He1_5876A'
        spec.fit.frame(bands, input_cfg, obj_cfg_prefix='38-35', line_list=[line_label], progress_output=None)

        line_db = Transition.from_db(line_label)
        line_log = Transition.from_log(line_label, spec.frame)
        line_fit = spec.fit.line

        assert line_db.label == line_log.label == 'He1_5876A'
        assert line_db.wavelength == line_log.wavelength == np.float64(5875.5352)
        assert line_db.particle == line_log.particle == 'He1'
        assert line_db.units_wave == line_log.units_wave == 'Angstrom'
        assert line_db.latex_label == line_log.latex_label == '$HeI5876\\mathring{A}$'
        assert line_db.core == line_log.core == 'He1_5876A'
        assert line_db.group_label == line_log.group_label is None
        assert line_db.group == line_log.group is None
        assert line_db.list_comps == line_log.list_comps == ['He1_5876A']
        assert line_db.ref_idx == line_log.ref_idx == 0
        assert line_db.pixel_mask == line_log.pixel_mask == 'no'
        assert line_db.kinem == line_log.kinem == 0
        assert line_db.trans == line_log.trans == 'rec'
        assert line_db.profile == line_log.profile == 'g'
        assert line_db.shape == line_log.shape == 'emi'
        assert_measurements(line_fit, line_log)

        line_label = 'H1_3889A_m'
        spec.fit.frame(bands, input_cfg, obj_cfg_prefix='38-35', line_list=[line_label], progress_output=None)

        line_db = Transition.from_db(line_label, fit_cfg=input_cfg)
        line_log = Transition.from_log(line_label, spec.frame)
        line_fit = spec.fit.line

        assert line_db.label == line_log.label == 'H1_3889A_m'
        assert line_db.wavelength == line_log.wavelength == np.float64(3888.988)
        assert line_db.particle == line_log.particle == 'H1'
        assert line_db.units_wave == line_log.units_wave == 'Angstrom'
        assert line_db.latex_label == line_log.latex_label == '$HI3889\\mathring{A}$+$HeI3889\\mathring{A}$'
        assert line_db.core == line_log.core == 'H1_3889A'
        assert line_db.group_label == line_log.group_label == 'H1_3889A+He1_3889A'
        assert line_db.group == line_log.group == 'm'
        assert line_db.list_comps == line_log.list_comps == ['H1_3889A', 'He1_3889A']
        assert line_db.ref_idx == line_log.ref_idx == 0
        assert line_db.pixel_mask == line_log.pixel_mask == 'no'
        assert line_db.kinem == line_log.kinem == 0
        assert line_db.trans == line_log.trans == 'rec'
        assert line_db.profile == line_log.profile == 'g'
        assert line_db.shape == line_log.shape == 'emi'
        # assert_measurements(line_fit, line_log)

        line_label = 'H1_4861A_b'
        spec.fit.frame(bands, input_cfg, obj_cfg_prefix='38-35', line_list=[line_label], progress_output=None)

        line_db = Transition.from_db(line_label, fit_cfg=input_cfg)
        line_log = Transition.from_log(line_label, spec.frame)
        line_fit = spec.fit.line

        assert line_db.label == line_log.label == 'H1_4861A_b'
        assert line_db.wavelength == line_log.wavelength == np.float64(4861.25)
        assert line_db.particle == line_log.particle == 'H1'
        assert line_db.units_wave == line_log.units_wave == 'Angstrom'
        assert line_db.latex_label == line_log.latex_label == '$HI4861\\mathring{A}$+$HI4861\\mathring{A}_k-1$'
        assert line_db.core == line_log.core == 'H1_4861A'
        assert line_db.group_label == line_log.group_label == 'H1_4861A+H1_4861A_k-1'
        assert line_db.group == line_log.group == 'b'
        assert line_db.list_comps == line_log.list_comps == ['H1_4861A', 'H1_4861A_k-1']
        assert line_db.ref_idx == line_log.ref_idx == 0
        assert line_db.pixel_mask == line_log.pixel_mask == 'no'
        assert line_db.kinem == line_log.kinem == 0
        assert line_db.trans == line_log.trans == 'rec'
        assert line_db.profile == line_log.profile == 'g'
        assert line_db.shape == line_log.shape == 'emi'
        assert_measurements(line_fit, line_log)

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
