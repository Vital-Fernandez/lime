import numpy as np
import lime
from lime.transitions import _DATABASE_FILE, LinesDatabase
from lime.rsrc_manager import lineDB
parent_bands = LinesDatabase(_DATABASE_FILE).frame


def test_line_bands():

    log0 = lime.lines_frame()
    assert np.all(log0.columns == parent_bands.columns)

    log1 = lime.lines_frame(wave_intvl=(3000, 7000))
    assert np.all((3000 <= log1.wavelength.to_numpy()) & (log1.wavelength.to_numpy() <= 7000))

    log2 = lime.lines_frame(wave_intvl=(3000, 7000), redshift=22)
    assert np.all((3000 * 2 <= log2.wavelength.to_numpy()) & (log2.wavelength.to_numpy() <= 7000 * 2))

    log3 = lime.lines_frame(line_list=['O3_4363A', 'O3_4959A', 'O3_5007A'])
    assert log3.index.isin(['O3_4363A', 'O3_4959A', 'O3_5007A']).sum() == log3.index.size

    log4 = lime.lines_frame(particle_list=['O3', 'S2'])
    assert log4.particle.isin(['O3', 'S2']).sum() == log4.index.size

    return


def test_database_modification():

    # Original database
    assert np.isclose(lime.lineDB.frame.loc['H1_1216A', 'wavelength'], 1215.67)
    assert np.isclose(lime.lineDB.frame.loc['H1_4861A', 'wavelength'], 4861.25)
    assert lime.lineDB._vacuum_check is False

    # Change to vacuum wavelength values
    lime.lineDB.set_database(vacuum_waves=True)
    assert np.isclose(lime.lineDB.frame.loc['H1_1216A', 'wavelength'], 1215.67)
    assert np.isclose(lime.lineDB.frame.loc['H1_4861A', 'wave_vac'], 4862.683)
    assert lime.lineDB._vacuum_check is True

    # Reset to original values
    lime.lineDB.reset()
    assert np.isclose(lime.lineDB.frame.loc['H1_1216A', 'wavelength'], 1215.67)
    assert np.isclose(lime.lineDB.frame.loc['H1_4861A', 'wavelength'], 4861.25)
    assert lime.lineDB._vacuum_check is False

    return