import numpy as np
import lime
from lime.io import load_fits
from pathlib import Path
from shutil import copy as shu_copy
from matplotlib import pyplot as plt


a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
idcs_searchsorted = np.searchsorted(a, (3, 7))
print(a[idcs_searchsorted[0]:idcs_searchsorted[1]])

idcs_limits = (a >= 5) & (a <= 5)
print(a[idcs_limits])


def format_line_mask_option(entry_value, wave_array):

    # Check if several entries
    formatted_value = entry_value.split(',') if ',' in entry_value else [f'{entry_value}']

    # Check if interval or single pixel mask
    for i, element in enumerate(formatted_value):
        if '-' in element:
            formatted_value[i] = element.split('-')
        else:
            element = float(element)
            pix_width = np.diff(wave_array).mean()/2
            formatted_value[i] = [element-pix_width, element+pix_width]

    formatted_value = np.array(formatted_value).astype(float)

    return formatted_value


def line_mask_indexing(wave_array, limits_array):

    # True values for masked pixels
    idcsMask = (wave_array[:, None] >= limits_array[:, 0]) & (wave_array[:, None] <= limits_array[:, 1])
    idcsMask = idcsMask.sum(axis=1).astype(bool)

    return idcsMask


conf_file = '/home/vital/PycharmProjects/vital_tests/astro/data/LzLCS_ISIS/LzLCS_ISIS_cfg.ini'
obsCfg = lime.load_cfg(conf_file)

dataFolder = Path(obsCfg['data_location']['data_folder'])

specNameList = obsCfg['sample_data']['specName_list']
zList = obsCfg['sample_data']['redshift_array']
arm_list = obsCfg['sample_data']['arm_list']
objList = obsCfg['sample_data']['objName_list']
refMask = '/home/vital/Dropbox/Astrophysics/Data/LzLCS_ISIS/data/reference_mask.txt'
norm_flux = obsCfg['sample_data']['norm_flux']
S2_6716A_b_mask = '8398-8399,8428.5-8430,8451.7'


for i, specName in enumerate(specNameList):
    for arm in arm_list:

        if (objList[i] == 'J105330') and (arm == 'Red'):

            # Load the spectra data
            file_name = f'{specName}_{arm}_f_w_e_flux_nearest.fits'
            wave, data, hdr = load_fits(dataFolder/objList[i]/file_name, instrument='ISIS', frame_idx=0)
            flux = data[0][0]
            fit_cfg = obsCfg[f'{objList[i]}_line_fitting']

            # Lime spectrum object
            print(f'- ({i}) {objList[i]}: {arm}')
            # spec.plot_spectrum(spec_label=objList[i])

            # # # Adjust mask to object
            obj_mask_file = dataFolder/objList[i]/f'{objList[i]}_{arm}_mask.txt'
            # # # shu_copy(refMask, obj_mask)
            # lime.MaskInspector(obj_mask_file, wave, flux, redshift=zList[i], norm_flux=norm_flux, y_scale='linear')
            mask = lime.load_lines_log(obj_mask_file)
            line_mask = mask.loc['S2_6716A_b', 'w1':'w6'].values * (1+zList[i])

            mask_limits = format_line_mask_option(S2_6716A_b_mask, wave)
            idcs_mask_limits = line_mask_indexing(wave, mask_limits)
            wave_ma = np.ma.masked_array(wave, mask=idcs_mask_limits)
            flux_ma = np.ma.masked_array(flux, mask=idcs_mask_limits)

            # spec = lime.Spectrum(wave, flux, redshift=zList[i], norm_flux=norm_flux)
            # spec.fit_from_wavelengths('S2_6716A_b', line_mask/(1+zList[i]), fit_cfg)
            # spec.display_results()
            # spec_mask = lime.Spectrum(wave_ma, flux_ma, redshift=zList[i], norm_flux=norm_flux)
            # spec_mask.fit_from_wavelengths('S2_6716A_b', line_mask/(1+zList[i]), fit_cfg)
            # spec_mask.display_results(log_scale=False)

            spec_new_mask = lime.Spectrum(wave, flux, redshift=zList[i], norm_flux=norm_flux)
            fit_cfg['S2_6716A_b_mask'] = S2_6716A_b_mask
            spec_new_mask.fit_from_wavelengths('S2_6716A_b', line_mask/(1+zList[i]), fit_cfg)
            spec_new_mask.display_results(log_scale=True, frame='rest')
            lime.save_line_log(spec_new_mask.log, 'testing_masks.txt')

            # fig, ax = plt.subplots(figsize=(12, 12))
            # ax.step(wave, flux, where='mid')
            # ax.step(wave_ma, flux_ma, where='mid')
            #
            # ax.scatter(wave[~idcs_mask_limits], flux[~idcs_mask_limits], color='red')
            # ax.set_xlim(line_mask[0], line_mask[-1])
            # # ax.set_yscale('log')
            # plt.show()