__all__ = [
    'spatial_mask_generator',
    'load_fits',
    'load_cfg',
    'load_lines_log',
    'save_line_log',
    'save_param_maps']

import os
import configparser
import copy
import numpy as np
import pandas as pd

from sys import exit, stdout
from pathlib import Path
from distutils.util import strtobool

from astropy.io import fits
from astropy.table import Table

from matplotlib import pyplot as plt, colors, cm, patches
from .plots import table_fluxes


try:
    import openpyxl
    openpyxl_check = True

except ImportError:
    openpyxl_check = False


# Parameters configuration: 0) Normalized by flux, 1) Regions wavelengths, 2) Array variable
LOG_COLUMNS = {'wavelength': [False, False, True],
               'intg_flux': [True, False, False],
               'intg_err': [True, False, False],
               'gauss_flux': [True, False, True],
               'gauss_err': [True, False, True],
               'eqw': [False, False, True],
               'eqw_err': [False, False, True],
               'ion': [False, False, True],
               'latexLabel': [False, False, True],
               'blended_label': [False, False, False],
               'w1': [False, True, False],
               'w2': [False, True, False],
               'w3': [False, True, False],
               'w4': [False, True, False],
               'w5': [False, True, False],
               'w6': [False, True, False],
               'peak_wave': [False, False, False],
               'peak_flux': [True, False, False],
               'cont': [True, False, False],
               'std_cont': [True, False, False],
               'm_cont': [True, False, False],
               'n_cont': [True, False, False],
               'snr_line': [False, False, False],
               'snr_cont': [False, False, False],
               'z_line': [False, False, True],
               'amp': [True, False, True],
               'center': [False, False, True],
               'sigma': [False, False, True],
               'amp_err': [True, False, True],
               'center_err': [False, False, True],
               'sigma_err': [False, False, True],
               'v_r': [False, False, True],
               'v_r_err': [False, False, True],
               'sigma_vel': [False, False, True],
               'sigma_vel_err': [False, False, True],
               'FWHM_int': [False, False, False],
               'FWHM_g': [False, False, True],
               'v_med': [False, False, False],
               'v_50': [False, False, False],
               'v_5': [False, False, False],
               'v_10': [False, False, False],
               'v_90': [False, False, False],
               'v_95': [False, False, False],
               'chisqr': [False, False, False],
               'redchi': [False, False, False],
               'aic':  [False, False, False],
               'bic':  [False, False, False],
               'observations': [False, False, False],
               'comments': [False, False, False]}

LINELOG_TYPES = {'index': '<U50',
                 'wavelength': '<f8',
                 'intg_flux': '<f8',
                 'intg_err': '<f8',
                 'gauss_flux': '<f8',
                 'gauss_err': '<f8',
                 'eqw': '<f8',
                 'eqw_err': '<f8',
                 'ion': '<U50',
                 'pynebCode': '<f8',
                 'pynebLabel': '<f8',
                 'lineType': '<f8',
                 'latexLabel': '<U100',
                 'blended_label': '<U120',
                 'w1': '<f8',
                 'w2': '<f8',
                 'w3': '<f8',
                 'w4': '<f8',
                 'w5': '<f8',
                 'w6': '<f8',
                 'm_cont': '<f8',
                 'n_cont': '<f8',
                 'cont': '<f8',
                 'std_cont': '<f8',
                 'peak_flux': '<f8',
                 'peak_wave': '<f8',
                 'snr_line': '<f8',
                 'snr_cont': '<f8',
                 'amp': '<f8',
                 'mu': '<f8',
                 'sigma': '<f8',
                 'amp_err': '<f8',
                 'mu_err': '<f8',
                 'sigma_err': '<f8',
                 'v_r': '<f8',
                 'v_r_err': '<f8',
                 'sigma_vel': '<f8',
                 'sigma_err_vel': '<f8',
                 'FWHM_int': '<f8',
                 'FWHM_g': '<f8',
                 'v_med': '<f8',
                 'v_50': '<f8',
                 'v_5': '<f8',
                 'v_10': '<f8',
                 'v_90': '<f8',
                 'v_95': '<f8',
                 'chisqr': '<f8',
                 'redchi': '<f8',
                 'observations': '<U50',
                 'comments': '<U50',
                 'obsFlux': '<f8',
                 'obsFluxErr': '<f8',
                 'f_lambda': '<f8',
                 'obsInt': '<f8',
                 'obsIntErr': '<f8'}

_LOG_EXPORT = list(set(LOG_COLUMNS.keys()) - set(['ion', 'wavelength',
                                                 'latexLabel',
                                                 'w1', 'w2',
                                                 'w3', 'w4',
                                                 'w5', 'w6']))

_MASK_EXPORT = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'observation']

STANDARD_PLOT = {'figure.figsize': (14, 7),
                 'axes.titlesize': 14,
                 'axes.labelsize': 14,
                 'legend.fontsize': 12,
                 'xtick.labelsize': 12,
                 'ytick.labelsize': 12}

background_color = np.array((43, 43, 43))/255.0
foreground_color = np.array((179, 199, 216))/255.0
red_color = np.array((43, 43, 43))/255.0
yellow_color = np.array((191, 144, 0))/255.0

DARK_PLOT = {'figure.figsize': (14, 7),
             'axes.titlesize': 14,
             'axes.labelsize': 14,
             'legend.fontsize': 12,
             'xtick.labelsize': 12,
             'ytick.labelsize': 12,
             'text.color': foreground_color,
             'figure.facecolor': background_color,
             'axes.facecolor': background_color,
             'axes.edgecolor': foreground_color,
             'axes.labelcolor': foreground_color,
             'xtick.color': foreground_color,
             'ytick.color': foreground_color,
             'legend.edgecolor': 'inherit',
             'legend.facecolor': 'inherit'}

STANDARD_AXES = {'xlabel': r'Wavelength $(\AA)$', 'ylabel': r'Flux $(erg\,cm^{-2} s^{-1} \AA^{-1})$'}

STRINGCONFKEYS = ['sampler', 'reddenig_curve', 'norm_line_label', 'norm_line_pynebCode']
GLOBAL_LOCAL_GROUPS = ['line_fitting', 'chemical_model'] # TODO not implemented

FLUX_TEX_TABLE_HEADERS = [r'$Transition$', '$EW(\AA)$', '$F(\lambda)$', '$I(\lambda)$']
FLUX_TXT_TABLE_HEADERS = [r'$Transition$', 'EW', 'EW_error', 'F(lambda)', 'F(lambda)_error', 'I(lambda)', 'I(lambda)_error']

KIN_TEX_TABLE_HEADERS = [r'$Transition$', r'$Comp$', r'$v_{r}\left(\nicefrac{km}{s}\right)$', r'$\sigma_{int}\left(\nicefrac{km}{s}\right)$', r'Flux $(\nicefrac{erg}{cm^{-2} s^{-1} \AA^{-1})}$']
KIN_TXT_TABLE_HEADERS = [r'$Transition$', r'$Comp$', 'v_r', 'v_r_error', 'sigma_int', 'sigma_int_error', 'flux', 'flux_error']


# Function to check if variable can be converte to float else leave as string
def check_numeric_Value(s):
    try:
        output = float(s)
        return output
    except ValueError:
        return s


# Function to map a string to its variable-type
def format_option_value(entry_value, key_label, section_label='', float_format=None, nan_format='nan'):
    output_variable = None

    # None variable
    if (entry_value == 'None') or (entry_value is None):
        output_variable = None

    # Dictionary blended lines
    elif 'line_fitting' in section_label:
        output_variable = {}
        keys_and_values = entry_value.split(',')
        for pair in keys_and_values:

            # Conversion for parameter class atributes
            if ':' in pair:
                key, value = pair.split(':')
                if value == 'None':
                    output_variable[key] = None
                elif key in ['value', 'min', 'max']:
                    output_variable[key] = float(value)
                elif key == 'vary':
                    output_variable[key] = strtobool(value) == 1
                else:
                    output_variable[key] = value

            # Conversion for non-parameter class atributes (only str conversion possible)
            else:
                output_variable = check_numeric_Value(entry_value)

    # Arrays (The last boolean overrides the parameters
    # TODO keys with array are always converted to numpy array even if just one

    elif ',' in entry_value:

        # Specia cases conversion
        if key_label in ['input_lines']:
            if entry_value == 'all':
                output_variable = 'all'
            else:
                output_variable = np.array(entry_value.split(','))

        elif '_array' in key_label:
            output_variable = np.fromstring(entry_value, dtype=np.float, sep=',')

        elif '_prior' in key_label:
            entry_value = entry_value.split(',')
            output_variable = np.array([float(entry_value[i]) if i > 0 else
                                        entry_value[i] for i in range(len(entry_value))], dtype=object)

        # List of strings
        elif '_list' in key_label:
            output_variable = entry_value.split(',')

        # Objects arrays
        else:
            newArray = []
            textArrays = entry_value.split(',')
            for item in textArrays:
                convertValue = float(item) if item != 'None' else np.nan
                newArray.append(convertValue)
            output_variable = np.array(newArray)

    # Boolean
    elif '_check' in key_label:
        output_variable = strtobool(entry_value) == 1

    # Standard strings
    elif ('_folder' in key_label) or ('_file' in key_label):
        output_variable = entry_value

    # Check if numeric possible else string
    else:

        if '_list' in key_label:
            output_variable = [entry_value]

        elif '_array' in key_label:
            output_variable = np.array([entry_value], ndmin=1)

        else:
            output_variable = check_numeric_Value(entry_value)

    return output_variable


# Function to load SpecSyzer configuration file
def load_cfg(file_address, obj_section=None, mask_section=None, def_cfg_sec='line_fitting'):

    """
    This function reads a configuration file with the `standard ini format <https://en.wikipedia.org/wiki/INI_file>`_. Please
    check the ``.format_option_value`` function for the special keywords conversions done by LiMe.

    If the user provides a list of objects (via the ``obj_section`` parameter) this function will update each object fitting
    configuration to include the default configuration. If there are shared entries, the object configuration takes precedence.
    The object section must have have the "objectName_line_fitting" notation, where the "objectName" is obtained from
    the object list.

    If the user provides a list of masks (via the ``mask_section`` parameter) this function will update the spaxel line fitting
    configuration to include the mask configuration. If there are shared entries the spaxel configuration takes preference.
    The spaxel  section must have follow the "idxY-idxX_line_fitting" notation, where "idxY-idxX" are the y and x indices
    of the masked spaxel obtained from the mask.

    .. attention::
        For the right formatting of the line fitting configuration entries the user must include the "line_fitting" string
        in the file configuration section name. For example:

    .. code-block::

        [default_line_fitting]
        H1_6563A_b = H1_6563A-N1_6584A-N1_6548A

        [IZwicky18_line_fitting]
        O2_3726A_m = O2_3726A-O2_3729A

        [123-84_line_fitting]
        H1_6563A_b = H1_6563A-N1_6584A

    :param file_address: configuration file location
    :type file_address: str or ~pathlib.Path

    :param obj_section: the section:option location for the list of objects, e.g. {'sample data': 'obj_list'}
    :type obj_section: dict, optional

    :param mask_section: the section:option location of the spatial masks, e.g. {'sample data': 'obj_mask_list'}
    :type mask_section: dict, optional

    :param def_cfg_sec: the section(s) with the line fitting configuration, e.g. 'default_line_fitting'
    :type def_cfg_sec: str or list, optional

    :return: Parsed configuration data
    :rtype: dict

    """

    # Open the file
    if Path(file_address).is_file():
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        cfg.read(file_address)
    else:
        exit(f'-ERROR Configuration file not found at:\n{file_address}')

    # Convert the configuration entries from the string format if possible
    cfg_lime = {}
    for section in cfg.sections():
        cfg_lime[section] = {}
        for option_key in cfg.options(section):
            option_value = cfg[section][option_key]
            cfg_lime[section][option_key] = format_option_value(option_value, option_key, section)

    # Update the object line fitting sections if provided by user
    if obj_section is not None:

        for sec_objs, opt_objs in obj_section.items():

            # Get the list of objects
            assert sec_objs in cfg_lime, f'- ERROR: No {sec_objs} section in file {file_address}'
            assert opt_objs in cfg_lime[sec_objs], f'- ERROR: No {opt_objs} option in section {sec_objs} in file {file_address}'
            objList = cfg_lime[sec_objs][opt_objs]

            # Get the default configuration
            assert def_cfg_sec in cfg_lime, f'- ERROR: No {def_cfg_sec} section in file {file_address}'
            global_cfg = cfg_lime[def_cfg_sec]

            # Loop through the objects
            for obj in objList:
                global_dict = copy.deepcopy(global_cfg)

                local_label = f'{obj}_line_fitting'
                local_dict = cfg_lime[local_label] if local_label in cfg_lime else {}

                # Local configuration overwriting global
                global_dict.update(local_dict)
                cfg_lime[local_label] = global_dict

    return cfg_lime


# Function to save SpecSyzer configuration file
def save_cfg(output_file, param_dict, section_name=None, clear_section=False):

    """
    This function safes the input dictionary into a configuration file. If no section is provided the input dictionary
    overwrites the data

    """
    # TODO add mechanic for commented conf lines. Currently they are being erased in the load/safe process

    # Creating a new file (overwritting old if existing)
    if section_name == None:

        # Check all entries are dictionaries
        values_list = [*param_dict.values()]
        section_check = all(isinstance(x, dict) for x in values_list)
        assert section_check, f'ERROR: Dictionary for {output_file} cannot be converted to configuration file. Confirm all its values are dictionaries'

        output_cfg = configparser.ConfigParser()
        output_cfg.optionxform = str

        # Loop throught he sections and options to create the files
        for section_name, options_dict in param_dict.items():
            output_cfg.add_section(section_name)
            for option_name, option_value in options_dict.items():
                option_formatted = format_option_value(option_value, option_name, section_name)
                output_cfg.set(section_name, option_name, option_formatted)

        # Save to a text format
        with open(output_file, 'w') as f:
            output_cfg.write(f)

    # Updating old file
    else:

        # Confirm file exists
        file_check = os.path.isfile(output_file)

        # Load original cfg
        if file_check:
            output_cfg = configparser.ConfigParser()
            output_cfg.optionxform = str
            output_cfg.read(output_file)
        # Create empty cfg
        else:
            output_cfg = configparser.ConfigParser()
            output_cfg.optionxform = str

        # Clear section upon request
        if clear_section:
            if output_cfg.has_section(section_name):
                output_cfg.remove_section(section_name)

        # Add new section if it is not there
        if not output_cfg.has_section(section_name):
            output_cfg.add_section(section_name)

        # Map key values to the expected format and store them
        for option_name, option_value in param_dict.items():
            option_formatted = format_option_value(option_value, option_name, section_name)
            output_cfg.set(section_name, option_name, option_formatted)

        # Save to a text file
        with open(output_file, 'w') as f:
            output_cfg.write(f)

    return


def load_lines_log(log_address, ext='LINESLOG'):

    """
    This function reads a lines log table as a pandas dataframe. The accepted input file types are a whitespace separated
    text file a ``.fits`` file and an excel file (``.xls`` or ``.xlsx``). In the case of ``.fits`` or ``.xlsx`` files the user
    should specify the target page/sheet (the default value is ``LINESLOG``).

    :param log_address: Address of the configuration file. The function stops if the file is not found
    :type log_address: str

    :param ext: Name of the ``.fits`` file or ``.xlsx`` file extension with the extension name to read
    :type ext: str, optional

    :return: lines log table
    :rtype: pandas.DataFrame
    """

    # Check file is at path
    log_path = Path(log_address)
    assert log_path.is_file(), f'- Error: lines log not found at {log_address}'
    file_name, file_type = log_path.name, log_path.suffix

    try:

        # Fits file:
        if file_type == '.fits':
            log = Table.read(log_address, ext, character_as_bytes=False).to_pandas()
            log.set_index('index', inplace=True)

        # Excel table
        elif file_type in ['.xlsx' or '.xls']:
            log = pd.read_excel(log_address, sheet_name=ext, header=0, index_col=0)

        # Text file
        else:
            log = pd.read_csv(log_address, delim_whitespace=True, header=0, index_col=0)

    except ValueError as e:
        exit(f'\nERROR: LiMe could not open {file_type} file at {log_address}\n{e}')

    return log


def save_line_log(log, log_address, ext='LINESLOG', fits_header=None):

    """
    This function saves the input lines log at the location provided by the user.

    The function takes into consideration the extension of the output address for the log file format.

    The valid output formats are .txt, .pdf, .fits and .xlsx

    For .fits and excel files the user can provide an ``ext`` name for the HDU/sheet.

    For .fits files the user can provide a dictionary so that its keys-values are stored in the header.

    :param log: Lines log with the measurements
    :type log: pandas.DataFrame

    :param log_address: Address for the output lines log file.
    :type log_address: str

    :param ext: Name for the HDU/sheet in output .fits and excel files. If the target file already has this extension it
                will be overwritten. The default value is LINESLOG.
    :type ext: str, optional

    :param fits_header: Dictionary with key-values to be included in the output .fits file header.
    :type fits_header: dict, optional

    """

    # Confirm file path exits
    log_path = Path(log_address)
    assert log_path.parent.exists(), f'- ERROR: Output lines log folder not found ({log_path.parent})'
    file_name, file_type = log_path.name, log_path.suffix

    # Default txt log with the complete information
    if file_type == '.txt':
        with open(log_path, 'wb') as output_file:
            string_DF = log.to_string()
            output_file.write(string_DF.encode('UTF-8'))

    # Pdf fluxes table
    elif file_type == '.pdf':
        table_fluxes(log, log_path.parent / log_path.stem)

    # Lines log in a fits file
    elif file_type == '.fits':
        if isinstance(log, pd.DataFrame):
            lineLogHDU = lineslog_to_HDU(log, ext_name=ext, header_dict=fits_header)

            if log_path.is_file():
                try:
                    fits.update(log_path, data=lineLogHDU.data, header=lineLogHDU.header, extname=lineLogHDU.name, verify=True)
                except KeyError:
                    fits.append(log_path, data=lineLogHDU.data, header=lineLogHDU.header, extname=lineLogHDU.name)
            else:
                hdul = fits.HDUList([fits.PrimaryHDU(), lineLogHDU])
                hdul.writeto(log_path, overwrite=True, output_verify='fix')

    # Default log in excel format
    elif file_type == '.xlsx' or file_type == '.xls':

        # Check openpyxl is installed else leave
        if openpyxl:
            pass
        else:
            print(f'\n- WARNING: openpyxl is not installed. Lines log {log_address} could not be saved')
            return

        if not log_path.is_file():
            with pd.ExcelWriter(log_path) as writer:
                log.to_excel(writer, sheet_name=ext)
        else:
            # THIS WORKS IN WINDOWS BUT NOT IN LINUX
            # with pd.ExcelWriter(log_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
            #     log.to_excel(writer, sheet_name=ext)

            if file_type == '.xlsx':
                book = openpyxl.load_workbook(log_path)
                with pd.ExcelWriter(log_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                    writer.book = book
                    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                    log.to_excel(writer, sheet_name=ext)
            else:
                # TODO this does not write to a xlsx file
                with pd.ExcelWriter(log_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                    log.to_excel(writer, sheet_name=ext)

    else:
        print(f"--WARNING: output extension {file_type} was not recognised in file {log_path}")
        exit()

    return


def lineslog_to_HDU(log_DF, ext_name=None, column_types={}, header_dict={}):

    # For non empty logs
    if not log_DF.empty:

        if len(column_types) == 0:
            params_dtype = LINELOG_TYPES
        else:
            params_dtype = LINELOG_TYPES.copy()
            user_dtype = column_types.copy()
            params_dtype.update(user_dtype)

        linesSA = log_DF.to_records(index=True, column_dtypes=params_dtype, index_dtypes='<U50')
        linesCol = fits.ColDefs(linesSA)
        linesHDU = fits.BinTableHDU.from_columns(linesCol, name=ext_name)

        if header_dict is not None:
            if len(header_dict) != 0:
                for key, value in header_dict.items():
                    linesHDU.header[key] = value

    # Empty log
    else:
        linesHDU = None

    return linesHDU


def load_fits(file_address, instrument, frame_idx=None):

    if instrument == 'ISIS':

        # Open fits file
        with fits.open(file_address) as hdul:
            data, header = hdul[frame_idx].data, hdul[frame_idx].header

        assert 'ISIS' in header['INSTRUME'], 'Input spectrum instrument '

        # William Herschel Telescope ISIS instrument
        w_min = header['CRVAL1']
        dw = header['CD1_1']  # dw = 0.862936 INDEF (Wavelength interval per pixel)
        pixels = header['NAXIS1']  # nw = 3801 number of output pixels
        w_max = w_min + dw * pixels
        wave = np.linspace(w_min, w_max, pixels, endpoint=False)

        return wave, data, header

    elif instrument == 'fits-cube':

        # Open fits file
        with fits.open(file_address) as hdul:
            data, hdr = hdul[frame_idx].data, hdul[frame_idx].header


        dw = hdr['CD3_3']
        w_min = hdr['CRVAL3']
        nPixels = hdr['NAXIS3']
        w_max = w_min + dw * nPixels
        wave = np.linspace(w_min, w_max, nPixels, endpoint=False)

        print('CD3_3', dw)
        print('CRVAL3', w_min)
        print('NAXIS3', nPixels)
        print('np.diff', np.diff(wave).mean(), np.std(np.diff(wave)))

        return wave, data, hdr

    elif instrument == 'OSIRIS':

        # Default frame index
        if frame_idx is None:
            frame_idx = 0

        # Open fits file
        with fits.open(file_address) as hdul:
            data, header = hdul[frame_idx].data, hdul[frame_idx].header

        # assert 'OSIRIS' in header['INSTRUME']

        w_min = header['CRVAL1']
        dw = header['CD1_1']  # dw (Wavelength interval per pixel)
        pixels = header['NAXIS1']  # nw number of output pixels
        w_max = w_min + dw * pixels
        wave = np.linspace(w_min, w_max, pixels, endpoint=False)

        return wave, data, header

    elif instrument == 'SDSS':

        # Open fits file
        with fits.open(file_address) as hdul:
            data, header_0, header_2, header_3 = hdul[1].data, hdul[0].header, hdul[2].data, hdul[3].data

        assert 'SDSS 2.5-M' in header_0['TELESCOP']

        wave = 10.0 ** data['loglam']
        SDSS_z = float(header_2["z"][0] + 1)
        wave_rest = wave / SDSS_z

        flux_norm = data['flux']
        flux = flux_norm / 1e17

        headers = (header_0, header_2, header_3)

        # return wave_rest, flux, headers
        return wave, data, headers

    elif instrument == 'xshooter':

        # Default frame index
        if frame_idx is None:
            frame_idx = 1

        # Following the steps at: https://archive.eso.org/cms/eso-data/help/1dspectra.html
        with fits.open(file_address) as hdul:
            data, header = hdul[frame_idx].data, hdul[frame_idx].header

        w_min = header['CRVAL1']
        dw = header['CDELT1']  # dw (Wavelength interval per pixel)
        pixels = header['NAXIS1']  # nw number of output pixels
        w_max = w_min + dw * pixels
        wave = np.linspace(w_min, w_max, pixels, endpoint=False)

        # wave = data[0][0]

        return wave, data, header

    elif instrument == 'MEGARA':

        # Default frame index
        if frame_idx is None:
            frame_idx = 1

        # Following the steps at: https://archive.eso.org/cms/eso-data/help/1dspectra.html
        with fits.open(file_address) as hdul:
            data, header = hdul[frame_idx].data, hdul[frame_idx].header

        w_min = header['CRVAL1']
        dw = header['CDELT1']  # dw (Wavelength interval per pixel)
        pixels = header['NAXIS1']  # nw number of output pixels
        w_max = w_min + dw * pixels
        wave = np.linspace(w_min, w_max, pixels, endpoint=False)

        return wave, data, header

    else:

        print('-- WARNING: Instrument not recognize')

        # Open fits file
        with fits.open(file_address) as hdul:
            data, header = hdul[frame_idx].data, hdul[frame_idx].header

        return data, header




def spatial_mask_generator(image_flux, mask_param, contour_levels, mask_ref="", output_address=None,
                           min_level=None, show_plot=False):

    """
    This function computes a spatial mask for an input flux image given an array of limits for a certain intensity parameter.
    Currently the only one implented is the percentil intensity. If an output address is provided, the mask is saved as a fits file
    where each intensity level mask is stored in its corresponding page. The parameter calculation method, its intensity and mask
    index are saved in the corresponding HDU header as PARAM, PARAMIDX and PARAMVAL.

    :param image_flux: Matrix with the image flux to be spatially masked.
    :type image_flux: np.array()

    :param mask_param: Flux intensity model from which the masks are calculated. The options available are ['percentil'].
    :type mask_param: str, optional

    :param contour_levels: Vector in decreasing order with the parameter values for the mask_param chosen.
    :type contour_levels: np.array()

    :param mask_ref: String label for the mask. If none provided the masks will be named in cardinal order.
    :type mask_ref: str, optional

    :param output_address: Output address for the mask fits file.
    :type output_address: str, optional

    :param min_level: Minimum level for the masks calculation. If none is provided the minimum value from the contour_levels
                      vector will be used.
    :type min_level: float, optional

    :param show_plot: If true a plot will be displayed with the mask calculation. Additionally if an output_address is
                      provided the plot will be saved in the parent folder as image taking into consideration the
                      mask_ref value.

    :type show_plot: bool, optional
    :return:
    """

    # Check the contour vector is in decreasing order
    assert np.all(np.diff(contour_levels) < 0), '- ERROR contour_levels are not in decreasing order for spatial mask'

    # Check the logic for the mask calculation
    assert mask_param in ['percentil', 'SNR'], f'- ERROR {mask_param} is not recognise for the spatial mask calculation'

    # Compute the mask diagnostic parameter form the provided flux
    if mask_param == 'percentil':
        param_image = image_flux
        param_array = np.nanpercentile(image_flux, contour_levels)

    # If minimum level not provided by user use lowest contour_level
    min_level = param_array[-1] if min_level is None else min_level

    # Containers for the mask parameters
    mask_dict = {}
    param_level = {}
    boundary_dict = {}

    # Loop throught the counter levels and compute the
    for i, n_levels in enumerate(param_array):

        # # Operation every element
        if i == 0:
            maParamImage = np.ma.masked_where((param_image >= param_array[i]) &
                                              (param_image >= min_level),
                                               param_image)

        else:
            maParamImage = np.ma.masked_where((param_image >= param_array[i]) &
                                              (param_image < param_array[i - 1]) &
                                              (param_image >= min_level),
                                               param_image)

        if np.sum(maParamImage.mask) > 0:
            mask_dict[f'mask_{i}'] = maParamImage.mask
            boundary_dict[f'mask_{i}'] = contour_levels[i]
            param_level[f'mask_{i}'] = param_array[i]

    # Plot the combined masks
    if output_address is not None:
        fits_folder = Path(output_address).parent
    else:
        fits_folder = None

    if (fits_folder is not None) or show_plot:

        fig, ax = plt.subplots(figsize=(12, 12))

        cmap = cm.get_cmap('viridis_r', len(mask_dict))
        legend_list = [None] * len(mask_dict)
        # alpha_levels = np.linspace(0.1, 0.5, len(mask_dict))[::-1]

        for idx_region, region_items in enumerate(mask_dict.items()):

            region_label, region_mask = region_items

            # Inverse the mask array for the plot
            inv_mask_array = np.ma.masked_array(region_mask.data, ~region_mask)

            # Prepare the labels for each mask to add to imshow
            legend_text = f'mask_{idx_region}: {mask_param} = {boundary_dict[region_label]}'
            legend_list[idx_region] = patches.Patch(color=cmap(idx_region),label=legend_text)

            cm_i = colors.ListedColormap(['black', cmap(idx_region)])
            ax.imshow(inv_mask_array, cmap=cm_i, vmin=0, vmax=1)

        ax.legend(handles=legend_list, loc=2.)
        plt.tight_layout()

        if fits_folder is not None:

            if mask_ref is None:
                output_image = fits_folder/f'mask_contours.png'
            else:
                output_image = fits_folder/f'{mask_ref}_mask_contours.png'

            plt.savefig(output_image)

        if show_plot:
            plt.show()

        plt.close(fig)

    # Save to a fits file:
    if output_address is not None:

        fits_address = Path(output_address)

        for idx_region, region_items in enumerate(mask_dict.items()):
            region_label, region_mask = region_items

            # Metadata for the fits page
            header_dict = {'PARAM': mask_param,
                           'PARAMIDX': boundary_dict[region_label],
                           'PARAMVAL': param_level[region_label]}
            fits_hdr = fits.Header(header_dict)

            # Extension for the mask
            mask_ext = region_label if mask_ref is None else f'{mask_ref}_{region_label}'

            # Mask HDU
            mask_hdu = fits.ImageHDU(name=mask_ext, data=region_mask.astype(int), ver=1, header=fits_hdr)

            if fits_address.is_file():
                try:
                    fits.update(fits_address, data=mask_hdu.data, header=mask_hdu.header, extname=mask_ext, verify=True)
                except KeyError:
                    fits.append(fits_address, data=mask_hdu.data, header=mask_hdu.header, extname=mask_ext)
            else:
                hdul = fits.HDUList([fits.PrimaryHDU(), mask_hdu])
                hdul.writeto(fits_address, overwrite=True, output_verify='fix')

    return


def progress_bar(i, i_max, post_text, n_bar=10):

    # Size of progress bar
    j = i/i_max
    stdout.write('\r')
    message = f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {post_text}"
    stdout.write(message)
    stdout.flush()

    return


def save_param_maps(log_file_address, param_dict, output_folder, mask_file_address=None, ext_mask='all', image_shape=None,
                   ext_log='_LINESLOG', default_spaxel_value=np.nan, output_files_prefix=None, page_hdr={}):
    """

    This function loads a ``.fits`` file with the line log measurements and generates a set of spatial images from a dictionary
    of parameters and lines provided by the user. For every parameter, the function generates a .fits file with multiple
    pages (`HDUs <https://docs.astropy.org/en/stable/io/fits/api/hdus.html>`_), one per input line.

    The ``.fits`` log pages will be queried by voxel coordinates (the default format is ``{idx_j}-{idx_i}_LINESLOG``).
    The user can provide a spatial mask address with the spaxels for which to recover the line log measurements. If the mask
    ``.fits`` file contains several extensions, the user can provide a list of which ones to use. Otherwise, all will be used.

    .. attention::
        The user can provide an ``image_shape`` array to generate the output image size. However, in big images attempting this
        approach, instead of providing a spatial mask with the science data location, can require a long time to inspect
        the log measurements.

    The output ``.fits`` image maps include a header with the ``PARAM`` and ``LINE`` with the line and parameter labels
    respectively (see `measurements <documentation/measurements.html>`_).

    :param log_file_address: fits file address location with the line logs
    :type log_file_address: str

    :param param_dict: Dictionary with the lists of lines to map for every parameter, e.g. ``{'intg_flux': ['O3_5007A', 'H1_6563A']}``
    :type param_dict: dict

    :param output_folder: Output address for the fits maps
    :type output_folder: str

    :param mask_file_address: fits file address of the spatial mask images
    :type mask_file_address: str, optional

    :param ext_mask: Extension or list of extensions in the mask file to determine the list of spaxels to treat. 
                     By default uses all extensions (special keyword "all") 
    :type ext_mask: str or list, optional

    :param image_shape: Array with the image spatial size. The unis are the 2D array indices, e.g. (idx_j_max, idx_i_max)
    :type image_shape: list or array, optional

    :param ext_log: Suffix of the line logs extensions. The default value is "_LINESLOG". In this case the .fits file HDUs
                    will be queried as ``{idx_j}-{idx_i}_LINESLOG``, where ``idx_j`` and ``idx_i`` are the spaxel Y and X coordinates
                    respectively
    :type ext_log: str, optional

    :param default_spaxel_value: Default value for the output image spaxels, where no measurement was obtained from the logs.
                                 By default this value is numpy.nan
    :type default_spaxel_value: float, optional

    :param output_files_prefix: Prefix for the output image fits file. e.g. ``f'{output_files_prefix}{parameter}.fits'``. The
                                default value is None
    :type output_files_prefix: str, optional
    
    :param page_hdr: Dictionary with entries to include in the output parameter HDUs headers
    :type page_hdr: dict

    :return:
    """

    assert Path(log_file_address).is_file(), f'- ERROR: lines log at {log_file_address} not found'
    assert Path(output_folder).is_dir(), f'- ERROR: Output parameter maps folder {output_folder} not found'

    # Compile the list of voxels to recover the provided masks
    if mask_file_address is not None:

        assert Path(mask_file_address).is_file(), f'- ERROR: mask file at {mask_file_address} not found'

        with fits.open(mask_file_address) as maskHDUs:

            # Get the list of mask extensions
            if ext_mask == 'all':
                if ('PRIMARY' in maskHDUs) and (len(maskHDUs) > 1):
                    mask_list = []
                    for i, HDU in enumerate(maskHDUs):
                        mask_name = HDU.name
                        if mask_name != 'PRIMARY':
                            mask_list.append(mask_name)
                    mask_list = np.array(mask_list)
                else:
                    mask_list = np.array(['PRIMARY'])
            else:
                mask_list = np.array(ext_mask, ndmin=1)

            # Combine all the mask voxels into one
            for i, mask_name in enumerate(mask_list):
                if i == 0:
                    mask_array = maskHDUs[mask_name].data
                    image_shape = mask_array.shape
                else:
                    assert image_shape == maskHDUs[mask_name].data.shape, '- ERROR: Input masks do not have the same dimensions'
                    mask_array += maskHDUs[mask_name].data

            # Convert to boolean
            mask_array = mask_array.astype(bool)

            # List of spaxels in list [(idx_j, idx_i), ...] format
            spaxel_list = np.argwhere(mask_array)

    # No mask file is provided and the user just defines an image size tupple (nY, nX)
    else:
        mask_array = np.ones(image_shape).astype(bool)
        spaxel_list = np.argwhere(mask_array)

    # Generate containers for the data:
    images_dict = {}
    for param, line_list in param_dict.items():

        # Make sure is an array and loop throuh them
        line_list = np.array(line_list, ndmin=1)
        for line in line_list:
            images_dict[f'{param}-{line}'] = np.full(image_shape, default_spaxel_value)

    # Loop through the spaxels and fill the parameter images
    n_spaxels = spaxel_list.shape[0]
    spaxel_range = np.arange(n_spaxels)

    with fits.open(log_file_address) as logHDUs:

        for i_spaxel in spaxel_range:
            idx_j, idx_i = spaxel_list[i_spaxel]
            spaxel_ref = f'{idx_j}-{idx_i}{ext_log}'

            progress_bar(i_spaxel, n_spaxels, post_text=f'spaxels treated ({n_spaxels})')

            # Confirm log extension exists
            if spaxel_ref in logHDUs:

                # Recover extension data
                log_data = logHDUs[spaxel_ref].data
                log_lines = log_data['index']

                # Loop through the parameters and the lines:
                for param, user_lines in param_dict.items():
                    idcs_log = np.argwhere(np.in1d(log_lines, user_lines))
                    for i_line in idcs_log:
                        images_dict[f'{param}-{log_lines[i_line][0]}'][idx_j, idx_i] = log_data[param][i_line][0]

    # New line after the rustic progress bar
    print()

    # Save the parameter maps as individual fits files with one line per page
    output_files_prefix = '' if output_files_prefix is None else output_files_prefix
    for param, user_lines in param_dict.items():

        # Primary header
        paramHDUs = fits.HDUList()
        paramHDUs.append(fits.PrimaryHDU())

        # ImageHDU for the parameter maps
        for line in user_lines:
            hdr = fits.Header({'PARAM': param, 'LINE': param})
            hdr.update(page_hdr)
            data = images_dict[f'{param}-{line}']
            paramHDUs.append(fits.ImageHDU(name=line, data=data, header=hdr, ver=1))

        # Write to new file
        output_file = Path(output_folder)/f'{output_files_prefix}{param}.fits'
        paramHDUs.writeto(output_file, overwrite=True, output_verify='fix')

    return


