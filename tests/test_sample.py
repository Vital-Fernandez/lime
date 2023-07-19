import numpy as np
import pandas as pd
import lime
from pathlib import Path

lines_log_address = Path(__file__).parent/'data_tests'/'manga_lines_log.txt'

# Data for the tests
lines_log = lime.load_log(lines_log_address)

# Sample with 3 (repeated) observations
log_dict = {}
for i in range(3):
    log_dict[f'obj_{i}'] = lines_log.copy()

obs = lime.Sample()
obs.add_log_list(list(log_dict.keys()), list(log_dict.values()))


class TestSampleClass:

    def test_multindex_log(self):

        # Default import
        assert isinstance(obs.log.index, pd.MultiIndex)
        assert obs.log.index.names == ['id', 'line']
        assert np.all(obs.log.index.get_level_values('id').unique() == ['obj_0', 'obj_1', 'obj_2'])

        return

    def test_extract_fluxes(self):

        lime.extract_fluxes(obs.log, )

        return