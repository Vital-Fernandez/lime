import numpy as np
import pandas as pd
import lime


def test_line_bands():

    log0 = lime.line_bands()
    parent_bands = lime.load_frame(lime._lines_database_path)

    # TODO rework on the master database
    # assert np.all(log0.index == parent_bands.index)
    # assert log0.equals(parent_bands)
    assert np.all(log0.columns == parent_bands.columns)

    log1 = lime.line_bands(wave_intvl=(3000, 7000))
    assert np.all((3000 <= log1.wavelength.to_numpy()) & (log1.wavelength.to_numpy() <= 7000))

    log2 = lime.line_bands(wave_intvl=(3000, 7000), z_intvl=(0, 2))
    assert np.all((3000 * (1 + 0) <= log2.wavelength.to_numpy()) & (log2.wavelength.to_numpy() <= 7000 * (1 + 2)))

    log3 = lime.line_bands(lines_list=['O3_4363A', 'O3_4959A', 'O3_5007A'])
    assert log3.index.isin(['O3_4363A', 'O3_4959A', 'O3_5007A']).sum() == log3.index.size

    log4 = lime.line_bands(particle_list=['O3', 'S2'])
    assert log4.particle.isin(['O3', 'S2']).sum() == log4.index.size

    return