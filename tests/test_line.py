from lime.transitions import Line, label_decomposition


def test_line_components():

    line = 'O3_5007A'
    particle, wavelength, latex = label_decomposition(line)

    assert particle[0] == 'O3'

    assert wavelength[0] == 5007.0

    assert latex[0] == r'$[OIII]5007\AA$'

    return

