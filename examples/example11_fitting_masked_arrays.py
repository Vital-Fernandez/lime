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


def converting_entry(entry_value, wave_array):

    if ',' in entry_value:
        formated_value = entry_value.split(',')
    else:
        formated_value = [f'{entry_value}']

    for i, element in enumerate(formated_value):
        if '-' in element:
            formated_value[i] = element.split('-')
        else:
            element = float(element)
            pix_width = np.diff(wave_array).mean()/2
            formated_value[i] = [element-pix_width, element+pix_width]

    formated_value = np.array(formated_value).astype(float)

    return formated_value


def mask_indexing(wave_array, limits_array):

    # mask_array = np.ones(wave_array.size).astype(bool)
    #
    # for entry in limits_array:
    #     idcs_entry = np.searchsorted(wave_array, entry)
    #     mask_array[idcs_entry[0]:idcs_entry[1]] = False
    #
    # # Valid pixels are true
    # return mask_array

    mask_array = np.zeros(wave_array.size).astype(bool)

    for entry in limits_array:
        entry_array = (wave_array >= entry[0]) & (wave_array <= entry[1])
        mask_array += entry_array

    # Valid pixels are true
    return ~mask_array


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
            spec = lime.Spectrum(wave, flux, redshift=zList[i], norm_flux=norm_flux)
            # spec.plot_spectrum(spec_label=objList[i])

            # # Adjust mask to object
            obj_mask_file = dataFolder/objList[i]/f'{objList[i]}_{arm}_mask.txt'
            # # shu_copy(refMask, obj_mask)
            # lime.MaskInspector(obj_mask_file, wave, flux, redshift=zList[i], norm_flux=norm_flux, y_scale='linear')
            mask = lime.load_lines_log(obj_mask_file)
            line_mask = mask.loc['S2_6716A_b', 'w1':'w6'].values * (1+zList[i])

            mask_limits = converting_entry(S2_6716A_b_mask, wave)
            idcs_mask_limits = mask_indexing(wave, mask_limits)
            wave_ma = np.ma.masked_array(wave, mask=~idcs_mask_limits)
            flux_ma = np.ma.masked_array(flux, mask=~idcs_mask_limits)

            spec.fit_from_wavelengths('S2_6716A_b', line_mask/(1+zList[i]), fit_cfg)
            spec.display_results()

            spec_mask = lime.Spectrum(wave_ma, flux_ma, redshift=zList[i], norm_flux=norm_flux)
            spec_mask.fit_from_wavelengths('S2_6716A_b', line_mask/(1+zList[i]), fit_cfg)
            spec_mask.display_results()

            # fig, ax = plt.subplots(figsize=(12, 12))
            # ax.step(wave, flux, where='mid')
            # ax.step(wave_ma, flux_ma, where='mid')
            #
            # ax.scatter(wave[~idcs_mask_limits], flux[~idcs_mask_limits], color='red')
            # ax.set_xlim(line_mask[0], line_mask[-1])
            # # ax.set_yscale('log')
            # plt.show()