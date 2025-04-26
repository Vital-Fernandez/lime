from pathlib import Path
import numpy as np
import lime
from lime.io import check_fit_conf
from copy import deepcopy

baseline_folder = Path(__file__).parent / 'baseline'
outputs_folder = Path(__file__).parent / 'outputs'

conf_file_address = baseline_folder/'manga.toml'
lines_log_address = baseline_folder/'manga_lines_log.txt'

cfg = lime.load_cfg(conf_file_address)


def test_load_cfg():

    # Read data is as expected
    assert 'SHOC579' in cfg
    assert 'default_line_fitting' in cfg
    assert 'MASK_0_line_fitting' in cfg
    assert '38-35_line_fitting' in cfg

    assert 'H1_4861A_k-1_sigma' in cfg['MASK_0_line_fitting']
    assert isinstance(cfg['MASK_0_line_fitting']['H1_4861A_k-1_sigma'], dict)
    assert cfg['MASK_0_line_fitting']['H1_4861A_k-1_sigma']['expr'] == '>2.0*H1_4861A_sigma'

    assert 'H1_9548A_sigma' in cfg['MASK_0_line_fitting']
    assert isinstance(cfg['MASK_0_line_fitting']['H1_9548A_sigma'], dict)
    assert cfg['MASK_0_line_fitting']['H1_9548A_sigma']['min'] == 1.0
    assert cfg['MASK_0_line_fitting']['H1_9548A_sigma']['max'] == 2.0

    return


def test_cfg_levels():

    input_cfg = lime.load_cfg(conf_file_address)
    input_cfg_copy = deepcopy(input_cfg)

    # Configuration updated
    fig_cfg = check_fit_conf(input_cfg, default_key='MASK_0', obj_key='38-35')

    assert fig_cfg['bands'] == "./baseline/manga_line_bands.txt"
    assert fig_cfg['continuum']['degree_list'] == [3, 6, 6]
    assert fig_cfg['O3_5007A_b'] == "O3_5007A+O3_5007A_k-1+He1_5016A"
    assert fig_cfg['He1_5016A_center'] == {'min': 5014.0, 'max': 5018.0}
    assert fig_cfg['H1_6563A_k-1_amp'] == {'expr': '<10.0*H1_6563A_amp'}

    # Configuration custom
    fig_cfg = check_fit_conf(input_cfg, default_key='MASK_0', obj_key='38-35', update_default=False)

    assert fig_cfg.get('bands') is None
    assert fig_cfg.get('continuum') is None
    assert fig_cfg['O3_5007A_b'] == "O3_5007A+O3_5007A_k-1+He1_5016A"
    assert fig_cfg['He1_5016A_center'] == {'min': 5014.0, 'max': 5018.0}
    assert fig_cfg['H1_6563A_k-1_amp'] == {'expr': '<10.0*H1_6563A_amp'}

    # Configuration custom
    fig_cfg = check_fit_conf(input_cfg, default_key='MASK_0', obj_key='Not_existing', update_default=False)

    assert fig_cfg['bands'] == "./baseline/manga_line_bands.txt"
    assert fig_cfg['continuum']['degree_list'] == [3, 6, 6]
    assert fig_cfg['O3_5007A_b'] == "O3_5007A+O3_5007A_k-1"
    assert fig_cfg.get('He1_5016A_center') is None
    assert fig_cfg.get('H1_6563A_k-1_amp') is None

    # File versus dictionary
    file_cfg = check_fit_conf(conf_file_address, default_key='MASK_0', obj_key='38-35')
    dict_cfg = check_fit_conf(input_cfg, default_key='MASK_0', obj_key='38-35')
    assert file_cfg == dict_cfg

    # Confirm the process does not affect the original dictionary
    assert input_cfg == input_cfg_copy

    return


def test_save_cfg():

    save_file_address = outputs_folder/'new_manga.toml'
    copy_cfg = cfg.copy()
    copy_cfg['new_line_fitting'] = {'O3_5007A_kinem': "O3_4959A",
                                    'O3_5007A_k-1_kinem': "O3_4959A_k-1",
                                    'He1_5016A_center': "min:5014,max:5018",
                                    'He1_5016A_sigma': "min:1.0,max:2.0"}

    lime.save_cfg(copy_cfg, save_file_address)

    new_cfg = lime.load_cfg(save_file_address)

    assert 'SHOC579' in new_cfg
    assert 'default_line_fitting' in new_cfg
    assert 'MASK_0_line_fitting' in new_cfg
    assert '38-35_line_fitting' in new_cfg

    assert 'H1_4861A_k-1_sigma' in new_cfg['MASK_0_line_fitting']
    assert isinstance(new_cfg['MASK_0_line_fitting']['H1_4861A_k-1_sigma'], dict)
    assert new_cfg['MASK_0_line_fitting']['H1_4861A_k-1_sigma']['expr'] == '>2.0*H1_4861A_sigma'

    assert 'H1_9548A_sigma' in new_cfg['MASK_0_line_fitting']
    assert isinstance(new_cfg['MASK_0_line_fitting']['H1_9548A_sigma'], dict)
    assert new_cfg['MASK_0_line_fitting']['H1_9548A_sigma']['min'] == 1.0
    assert new_cfg['MASK_0_line_fitting']['H1_9548A_sigma']['max'] == 2.0

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
        line = lime.Line.from_log(label, log_lines)
        if line.blended_check is False:
            param_exp_value = log_lines.loc[label, 'eqw']
            param_value = log_lines.loc[label, 'eqw_new']
            param_exp_err = log_lines.loc[label, f'eqw_new_err']
            assert np.allclose(param_value, param_exp_value, atol=np.abs(param_exp_err * 2), equal_nan=True)

    return
