import numpy as np
import pandas as pd
import lime
from pathlib import Path

spectrum_log_path = Path(f'sample_data/gp121903_linelog.txt')
sample_log_path = Path(f'sample_data/sample_log.txt')

# Load the logs
log_spec = lime.load_log(spectrum_log_path)
log_sample = lime.load_log(sample_log_path)

list_ids = ['p10_G395M_24_s00024_x1d', 'p4_PRISM_43440_s43440_x1d', 'p8_G395M_44804_s44804_x1d']


z_log = lime.redshift_calculation(log_spec)

idcs = log_sample.index.get_level_values('id').isin(list_ids)
ids

idcs_slice = log_sample.index.get_level_values('id').isin(list_ids)
log_sample.loc[idcs, 'point':'ext'].copy()



print(z_log.loc['spec_0'])
z_log = lime.redshift_calculation(log_spec, line_list=['O3_5007A', 'H1_6563A'])
print(z_log.loc['spec_0'])
z_log = lime.redshift_calculation(log_spec, line_list='O3_5007A')
print(z_log.loc['spec_0'])
z_log = lime.redshift_calculation(log_spec, weight_parameter='eqw')
print(z_log.loc['spec_0'])
z_log = lime.redshift_calculation(log_spec, weight_parameter='gauss_flux')
print(z_log.loc['spec_0'])
z_log = lime.redshift_calculation(log_sample, line_list=['O3_5008A', 'H1_6565A'], weight_parameter='gauss_flux')
print(z_log)
