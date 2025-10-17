import numpy as np
import pandas as pd
import lime
import pytest
from pathlib import Path
from matplotlib import pyplot as plt

# Data for the tests
baseline_folder = Path(__file__).parent / 'baseline'
fits_address = baseline_folder/'sdss_dr18_0358-51818-0504.fits'
lines_log_address = baseline_folder / 'SHOC579_MANGA38-35_log.txt'
file_address = baseline_folder/'SHOC579_MANGA38-35.txt'


data_folder = Path(__file__).parent.parent/'examples/doc_notebooks/0_resources'
outputs_folder = data_folder/'results'
spectra_folder = data_folder/'spectra'

tolerance_rms = 5.5

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

    @pytest.mark.mpl_image_compare(tolerance=tolerance_rms)
    def test_check_redshift_spectrum(self, tmp_path):

        fig = plt.figure()

        # Declaring the name of the observations
        id_list = ['SHOC579_A', 'SHOC579_B', 'SHOC579_C']
        obs_list = ['sdss_dr18_0358-51818-0504.fits'] * 3

        # We declare the line measurements logs
        sample1 = lime.Sample.from_file(id_list, log_list=None, file_list=obs_list, instrument='sdss',
                                        folder_obs=baseline_folder, redshift=0.0475, norm_flux=1e-17)

        ref_lines = ['H1_4861A', 'O3_5007A', 'H1_6563A']
        sample_log_address = f'{tmp_path}/sample_log.txt'
        sample1.frame['z_line'] = 0
        sample1.check.redshift(sample1.frame.index, reference_lines=ref_lines, output_file_log=sample_log_address,
                               output_idcs=sample1.frame.index, redshift_column='z_line', initial_z=0.0475,
                               in_fig=fig)

        return fig

