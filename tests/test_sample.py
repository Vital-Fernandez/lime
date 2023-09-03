import numpy as np
import pandas as pd
import lime
from pathlib import Path

data_folder = Path(__file__).parent/'data_tests'
lines_log_address = data_folder/'manga_lines_log.txt'

# Data for the tests
lines_log = lime.load_log(lines_log_address)

class TestSampleClass:

    def test_from_sample_creation(self):

        sample1 = lime.Sample.from_file_list(id_list=['spec1', 'spec2'],
                                             log_list=[lines_log_address, lines_log_address],
                                             file_list=['spec1.fits', 'spec2.fits'])
        sample1.save_log(data_folder/f'sample1_3indeces.txt')

        assert list(sample1.log.index.names) == ['id', 'file', 'line']

        sample2 = lime.Sample.from_file_list(id_list=['spec1', 'spec2'], log_list=[lines_log_address, lines_log_address])
        sample2.save_log(data_folder/f'sample1_2indeces.txt')
        assert list(sample2.log.index.names) == ['id', 'line']

        sample3 = lime.Sample(data_folder/f'sample1_3indeces.txt')
        sample4 = lime.Sample(data_folder /f'sample1_2indeces.txt', levels=['id', 'line'])

        assert list(sample3.log.index.names) == ['id', 'file', 'line']
        assert list(sample4.log.index.names) == ['id', 'line']

        assert sample3.log.equals(sample1.log)
        assert sample4.log.equals(sample2.log)

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