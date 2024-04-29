import numpy as np
import pandas as pd
import lime
from pathlib import Path


baseline_folder = Path(__file__).parent / 'baseline'
outputs_folder = Path(__file__).parent / 'outputs'
lines_log_address = baseline_folder / 'manga_lines_log.txt'

# Data for the tests
lines_log = lime.load_frame(lines_log_address)


class TestSampleClass:

    def test_from_sample_creation(self):

        sample1 = lime.Sample.from_file(id_list=['spec1', 'spec2'],
                                        log_list=[lines_log_address, lines_log_address],
                                        file_list=['spec1.fits', 'spec2.fits'],
                                        instrument='isis')
        sample1.save_frame(outputs_folder / f'sample1_3indeces.txt')

        assert list(sample1.frame.index.names) == ['id', 'file', 'line']

        sample2 = lime.Sample.from_file(id_list=['spec1', 'spec2'], log_list=[lines_log_address, lines_log_address],
                                        instrument='isis')
        sample2.save_frame(outputs_folder / f'sample1_2indeces.txt')
        assert list(sample2.frame.index.names) == ['id', 'line']

        sample3 = lime.Sample(outputs_folder / f'sample1_3indeces.txt', instrument='isis')
        sample4 = lime.Sample(outputs_folder / f'sample1_2indeces.txt', levels=['id', 'line'], instrument='isis')

        assert list(sample3.frame.index.names) == ['id', 'file', 'line']
        assert list(sample4.frame.index.names) == ['id', 'line']

        assert sample3.frame.equals(sample1.frame)
        assert sample4.frame.equals(sample2.frame)

        return

    # def test_multindex_log(self):
    #
    #     # Default import
    #     assert isinstance(obs.log.index, pd.MultiIndex)
    #     assert obs.log.index.names == ['id', 'line']
    #     assert np.all(obs.log.index.get_level_values('id').unique() == ['obj_0', 'obj_1', 'obj_2'])
    #
    #     return
    #
    # def test_extract_fluxes(self):
    #
    #     lime.extract_fluxes(obs.log, )
    #
    #     return