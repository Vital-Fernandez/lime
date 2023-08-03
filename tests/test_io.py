from pathlib import Path
import numpy as np
import lime

conf_file_address = Path(__file__).parent/'data_tests'/'manga.toml'
lines_log_address = Path(__file__).parent/'data_tests'/'manga_lines_log.txt'

cfg = lime.load_cfg(conf_file_address)

def test_load_cfg():

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


def test_save_cfg():

    save_file_address = Path(__file__).parent/'data_tests'/'new_manga.toml'
    copy_cfg = cfg.copy()
    copy_cfg['new_line_fitting'] = {'O3_5007A_kinem': "O3_4959A",
                                    'O3_5007A_k-1_kinem': "O3_4959A_k-1",
                                    'He1_5016A_center': "min:5014,max:5018",
                                    'He1_5016A_sigma': "min:1.0,max:2.0"
                                    }

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

    log_lines = lime.load_log(lines_log_address)
    lime.extract_fluxes(log_lines, flux_type='mixture')

    parameters = ['eqw_new',
                  'eqw_new_err']

    formulation = ['line_flux/cont',
                   '(line_flux/cont) * sqrt((line_flux_err/gauss_flux)**2 + (std_cont/cont)**2)']

    lime.log_parameters_calculation(log_lines, parameters, formulation)
    assert 'eqw_new' in log_lines.columns
    tol = log_lines['eqw_err'] * 2
    assert np.allclose(log_lines['eqw_new'], log_lines['eqw'], atol=tol)

    return