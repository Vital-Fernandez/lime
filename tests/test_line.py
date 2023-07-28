from lime.transitions import Line, label_decomposition, Particle
import numpy as np


def test_label_decomposition():

    # Components configuration
    fit_conf = {'O3_5007A_b': 'O3_5007A+O3_5007A_k-1+He1_5016A',
                'O3_5007A_m': 'O3_5007A+O3_5007A_k-1'}

    particle, wavelength, latex = label_decomposition('O3_5007A', fit_conf=fit_conf)
    assert particle[0] == 'O3'
    assert wavelength[0] == 5007.0
    assert latex[0] == r'$[OIII]5007\AA$'

    particle, wavelength, latex = label_decomposition('O3_5007A_b', fit_conf=fit_conf)
    assert particle[0] == 'O3'
    assert wavelength[0] == 5007.0
    assert latex[0] == r'$[OIII]5007\AA$'

    particle, wavelength, latex = label_decomposition('O3_5007A_m', fit_conf=fit_conf)
    assert particle[0] == 'O3'
    assert wavelength[0] == 5007.0
    assert latex[0] == r'$[OIII]5007\AA$+$[OIII]5007\AA-k_1$'

    return


class TestLineClass:

    def test_single_merged_blended(self):

        # Components configuration
        fit_conf = {'O3_5007A_b': 'O3_5007A+O3_5007A_k-1+He1_5016A',
                    'O3_5007A_m': 'O3_5007A+O3_5007A_k-1'}

        # Default band O3_5007A
        O3_band = np.array([4971.796688, 4984.514249, 4995.348943, 5024.303156, 5027.74326, 5043.797081])

        line = Line('O3_5007A', fit_conf=fit_conf)
        assert line.particle[0].label == 'O3'
        assert line.particle[0].symbol == 'O', line.particle[0].ionization == 3
        assert np.all(line.wavelength == np.array([5007.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == np.array(['col']))
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array([r'$[OIII]5007\AA$']))
        assert np.all(line.list_comps == ['O3_5007A'])

        assert line.label == 'O3_5007A'
        assert line.profile_label is np.nan
        assert np.all(line.mask == O3_band)

        line = Line('O3_5007A_b', band=None, fit_conf=fit_conf)
        assert np.all(line.particle == [Particle.from_label('O3'), Particle.from_label('O3'), Particle.from_label('He1')])
        assert np.all(line.wavelength == np.array([5007., 5007., 5016.]))
        assert np.all(line.kinem == np.array([0, 1, 0]))
        assert np.all(line.transition_comp == np.array(['col', 'col', 'rec']))
        assert np.all(line.profile_comp == np.array(['g-emi', 'g-emi', 'g-emi']))
        assert np.all(line.latex_label == np.array([r'$[OIII]5007\AA$', r'$[OIII]5007\AA-k_1$', r'$HeI5016\AA$']))
        assert np.all(line.list_comps == ['O3_5007A', 'O3_5007A_k-1', 'He1_5016A'])

        assert line.label == 'O3_5007A_b'
        assert line.profile_label == 'O3_5007A+O3_5007A_k-1+He1_5016A'
        assert np.all(line.mask == O3_band)

        line = Line('O3_5007A_m', band=np.array([1, 2, 3, 4, 5, 6]), fit_conf=fit_conf)
        assert np.all(line.particle == [Particle('O3', symbol='O', ionization=3)])
        assert np.all(line.wavelength == np.array([5007]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == np.array(['col']))
        assert np.all(line.profile_comp == np.array(['g-emi', 'g-emi']))
        assert np.all(line.latex_label == np.array([r'$[OIII]5007\AA$+$[OIII]5007\AA-k_1$']))
        assert np.all(line.list_comps == ['O3_5007A_m'])

        assert line.label == 'O3_5007A_m'
        assert line.profile_label == 'O3_5007A+O3_5007A_k-1'
        assert np.all(line.mask == np.array([1, 2, 3, 4, 5, 6]))

        return

    def test_special_particles(self):

        line = Line('OIII_5007A')
        assert line.particle[0].label == 'OIII'
        assert np.all(line.wavelength == np.array([5007.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == [None])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array([r'OIII-$5007\AA$']))
        assert np.all(line.list_comps == ['OIII_5007A'])

        assert line.label == 'OIII_5007A'
        assert line.profile_label is np.nan
        assert line.mask is None

        line = Line('Halpha_6563A')
        assert line.particle[0].label == 'Halpha'
        assert np.all(line.wavelength == np.array([6563.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == [None])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array([r'Halpha-$6563\AA$']))
        assert np.all(line.list_comps == ['Halpha_6563A'])

        assert line.label == 'Halpha_6563A'
        assert line.profile_label is np.nan
        assert line.mask is None

        line = Line('HIPas4-3_18751A')
        assert line.particle[0].label == 'HIPas4-3'
        assert np.all(line.wavelength == np.array([18751.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == [None])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array([r'HIPas4-3-$18751\AA$']))
        assert np.all(line.list_comps == ['HIPas4-3_18751A'])

        assert line.label == 'HIPas4-3_18751A'
        assert line.profile_label is np.nan
        assert line.mask is None

        line = Line('OIII_5007A')
        assert line.particle[0].label == 'OIII'
        assert np.all(line.wavelength == np.array([5007.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == [None])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array([r'OIII-$5007\AA$']))
        assert np.all(line.list_comps == ['OIII_5007A'])

        assert line.label == 'OIII_5007A'
        assert line.profile_label is np.nan
        assert line.mask is None

        line = Line('C3_1909A')
        assert line.particle[0].label == 'C3'
        assert np.all(line.wavelength == np.array([1909.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == ['sem'])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array([r'$CIII]1909\AA$']))
        assert np.all(line.list_comps == ['C3_1909A'])

        assert line.label == 'C3_1909A'
        assert line.profile_label is np.nan
        assert line.mask is not None

        line = Line('C3_1909A_t-sem', band=None)
        assert line.particle[0].label == 'C3'
        assert np.all(line.wavelength == np.array([1909.]))
        assert np.all(line.kinem == np.array([0]))
        assert np.all(line.transition_comp == ['sem'])
        assert np.all(line.profile_comp == np.array(['g-emi']))
        assert np.all(line.latex_label == np.array([r'$CIII]1909\AA$']))
        assert np.all(line.list_comps == ['C3_1909A_t-sem'])

        assert line.label == 'C3_1909A_t-sem'
        assert line.profile_label is np.nan
        assert line.mask is None

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
