from pathlib import Path
import numpy as np
import lime
from lime.io import check_fit_conf
from copy import deepcopy

baseline_folder = Path(__file__).parent / 'baseline'

data_folder = Path(__file__).parent.parent/'examples/doc_notebooks/0_resources'
outputs_folder = data_folder/'results'

conf_file_address = baseline_folder/'lime_tests.toml'
lines_log_address = baseline_folder/'SHOC579_MANGA38-35_log.txt'

cfg = lime.load_cfg(conf_file_address)


def test_load_cfg():

    # Read data is as expected
    assert 'object_properties' in cfg
    assert 'default_line_fitting' in cfg
    assert '38-35_line_fitting' in cfg
    assert 'MASK_0_line_fitting' in cfg

    assert 'H1_4861A_k-1_sigma' in cfg['MASK_0_line_fitting']
    assert isinstance(cfg['MASK_0_line_fitting']['H1_4861A_k-1_sigma'], dict)
    assert cfg['MASK_0_line_fitting']['H1_4861A_k-1_sigma']['expr'] == '>2.0*H1_4861A_sigma'

    assert 'H1_9546A_sigma' in cfg['38-35_line_fitting']
    assert isinstance(cfg['38-35_line_fitting']['H1_9546A_sigma'], dict)
    assert cfg['38-35_line_fitting']['H1_9546A_sigma']['min'] == 1.0
    assert cfg['38-35_line_fitting']['H1_9546A_sigma']['max'] == 2.0

    return


def test_cfg_levels():

    input_cfg = lime.load_cfg(conf_file_address)
    input_cfg_copy = deepcopy(input_cfg)

    # Configuration updated
    fig_cfg = check_fit_conf(input_cfg, default_key='default', obj_key='38-35')

    assert fig_cfg.get('continuum') == {'degree_list': [3, 6, 6], 'emis_threshold': [3, 2, 1.5]}
    assert fig_cfg['O3_5007A_b'] == "O3_5007A+O3_5007A_k-1+He1_5016A"
    assert fig_cfg['He1_5016A_center'] == {'min': 5014.0, 'max': 5018.0}
    assert fig_cfg['H1_6563A_k-1_amp'] == {'expr': '<10.0*H1_6563A_amp'}
    assert fig_cfg.get('N1_5198A_m') == "N1_5198A+N1_5200A"
    assert fig_cfg.get('O1_6300A_b') == "O1_6300A+S3_6312A"
    assert fig_cfg.get('H1_6563A_b') == "H1_6563A+H1_6563A_k-1+N2_6583A+N2_6548A"

    # Configuration custom
    fig_cfg = check_fit_conf(input_cfg, default_key='default', obj_key='38-35', update_default=False)
    assert fig_cfg.get('continuum') == {'degree_list': [3, 6, 6], 'emis_threshold': [3, 2, 1.5]}
    assert fig_cfg['O3_5007A_b'] == "O3_5007A+O3_5007A_k-1+He1_5016A"
    assert fig_cfg['He1_5016A_center'] == {'min': 5014.0, 'max': 5018.0}
    assert fig_cfg['H1_6563A_k-1_amp'] == {'expr': '<10.0*H1_6563A_amp'}
    assert fig_cfg.get('N1_5198A_m') is None
    assert fig_cfg.get('O1_6300A_b') is None
    assert fig_cfg.get('H1_6563A_b') == "H1_6563A+H1_6563A_k-1+N2_6583A+N2_6548A"

    # Configuration custom
    fig_cfg = check_fit_conf(input_cfg, default_key='default', obj_key='non_existing', update_default=False)
    assert fig_cfg.get('continuum') is None
    assert fig_cfg.get('O3_5007A_b') is None
    assert fig_cfg.get('He1_5016A_center') is None
    assert fig_cfg.get('H1_6563A_k-1_amp') is None
    assert fig_cfg.get('N1_5198A_m') == "N1_5198A+N1_5200A"
    assert fig_cfg.get('O1_6300A_b') == "O1_6300A+S3_6312A"
    assert fig_cfg.get('H1_6563A_b') == "H1_6563A+N2_6583A+N2_6548A"

    # File versus dictionary
    file_cfg = check_fit_conf(conf_file_address, default_key='MASK_0', obj_key='38-35')
    dict_cfg = check_fit_conf(input_cfg, default_key='MASK_0', obj_key='38-35')
    assert file_cfg == dict_cfg

    # Confirm the process does not affect the original dictionary
    assert input_cfg == input_cfg_copy

    return


def test_save_cfg():

    # Create new save
    save_file_address = outputs_folder/'new_manga.toml'
    copy_cfg = cfg.copy()
    copy_cfg['new_line_fitting'] = {'O3_5007A_kinem': "O3_4959A",
                                    'O3_5007A_k-1_kinem': "O3_4959A_k-1",
                                    'He1_5016A_center': "min:5014,max:5018",
                                    'He1_5016A_sigma': "min:1.0,max:2.0"}
    lime.save_cfg(save_file_address, copy_cfg)

    # Reload and test
    new_cfg = lime.load_cfg(save_file_address)
    assert 'object_properties' in cfg
    assert 'default_line_fitting' in cfg
    assert '38-35_line_fitting' in cfg
    assert 'MASK_0_line_fitting' in cfg

    assert 'H1_4861A_k-1_sigma' in new_cfg['MASK_0_line_fitting']
    assert isinstance(new_cfg['MASK_0_line_fitting']['H1_4861A_k-1_sigma'], dict)
    assert new_cfg['MASK_0_line_fitting']['H1_4861A_k-1_sigma']['expr'] == '>2.0*H1_4861A_sigma'

    assert 'H1_9546A_sigma' in new_cfg['38-35_line_fitting']
    assert isinstance(new_cfg['38-35_line_fitting']['H1_9546A_sigma'], dict)
    assert new_cfg['38-35_line_fitting']['H1_9546A_sigma']['min'] == 1.0
    assert new_cfg['38-35_line_fitting']['H1_9546A_sigma']['max'] == 2.0

    assert 'new_line_fitting' in new_cfg
    assert isinstance(new_cfg['new_line_fitting']['He1_5016A_center'], dict)
    assert isinstance(new_cfg['new_line_fitting']['He1_5016A_sigma'], dict)
    assert new_cfg['new_line_fitting']['He1_5016A_center']['min'] == 5014.0
    assert new_cfg['new_line_fitting']['He1_5016A_center']['max'] == 5018.0
    assert new_cfg['new_line_fitting']['He1_5016A_sigma']['min'] == 1.0
    assert new_cfg['new_line_fitting']['He1_5016A_sigma']['max'] == 2.0

    return


def test_log_parameters_calculation():

    log_lines = lime.load_frame(lines_log_address)
    lime.extract_fluxes(log_lines, flux_type='profile')

    parameters = ['eqw_new', 'eqw_new_err']

    formulation = ['line_flux/cont', '(line_flux/cont) * sqrt((line_flux_err/line_flux)**2 + (cont_err/cont)**2)']

    lime.log_parameters_calculation(log_lines, parameters, formulation)
    assert 'eqw_new' in log_lines.columns

    for label in log_lines.index:
        line = lime.Line.from_transition(label, data_frame=log_lines)
        if line.group == '_b':
            param_exp_value = log_lines.loc[label, 'eqw']
            param_value = log_lines.loc[label, 'eqw_new']
            param_exp_err = log_lines.loc[label, f'eqw_new_err']
            assert np.allclose(param_value, param_exp_value, atol=np.abs(param_exp_err * 2), equal_nan=True)

    return
