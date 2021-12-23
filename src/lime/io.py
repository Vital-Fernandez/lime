import os
import configparser
import copy
import numpy as np
import pandas as pd
import pylatex

from sys import exit
from pathlib import Path
from functools import partial
from collections import Sequence
from distutils.util import strtobool

from astropy.io import fits
from astropy.table import Table

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
               'observation': [False, False, False],
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
                 'observation': '<U50',
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
                                                 'w5', 'w6', 'observation']))

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
GLOBAL_LOCAL_GROUPS = ['_line_fitting', '_chemical_model']

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


# Function to import configparser
def importConfigFile(config_path):
    # Check if file exists
    if os.path.isfile(config_path):
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        cfg.read(config_path)
    else:
        exit(f"--WARNING: Configuration file {config_path} was not found. Exiting program")

    return cfg


# Function to map a string to its variable-type
def formatStringEntry(entry_value, key_label, section_label='', float_format=None, nan_format='nan'):
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

    # Arrays (The last boolean overrides the parameters # TODO unstable in case of one item lists
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


# Function to import SpecSyzer configuration file #TODO repeated
def load_cfg(filepath, objList_check=False):

    # Open the file
    if Path(filepath).is_file():
        cfg = importConfigFile(filepath)
        # TODO keys with array are always converted to numpy array even if just one
    else:
        exit(f'-ERROR Configuration file not found at:\n{filepath}')

    confDict = {}

    for section in cfg.sections():
        confDict[section] = {}
        for option_key in cfg.options(section):
            option_value = cfg[section][option_key]
            confDict[section][option_key] = formatStringEntry(option_value, option_key, section)

    if objList_check is True:

        assert 'file_information' in confDict, '- No file_information section in configuration file'
        assert 'object_list' in confDict['file_information'], '- No object_list option in configuration file'
        objList = confDict['file_information']['object_list']

        # Combine sample with obj properties if available
        if objList is not None:
            for key_group in GLOBAL_LOCAL_GROUPS:
                global_group = f'default{key_group}'
                if global_group in confDict:
                    for objname in objList:
                        local_group = f'{objname}{key_group}'
                        dict_global = copy.deepcopy(confDict[global_group])
                        if local_group in confDict:
                            dict_global.update(confDict[local_group])
                        confDict[local_group] = dict_global

    return confDict


def numberStringFormat(value, cifras = 4):
    if value > 0.001:
        newFormat = f'{value:.{cifras}f}'
    else:
        newFormat = f'{value:.{cifras}e}'

    return newFormat


def format_for_table(entry, rounddig=4, rounddig_er=2, scientific_notation=False, nan_format='-'):

    if rounddig_er == None: #TODO declare a universal tool
        rounddig_er = rounddig

    # Check None entry
    if entry != None:

        # Check string entry
        if isinstance(entry, (str, bytes)):
            formatted_entry = entry

        elif isinstance(entry, (pylatex.MultiColumn, pylatex.MultiRow, pylatex.utils.NoEscape)):
            formatted_entry = entry

        # Case of Numerical entry
        else:

            # Case of an array
            scalarVariable = True
            if isinstance(entry, (Sequence, np.ndarray)):

                # Confirm is not a single value array
                if len(entry) == 1:
                    entry = entry[0]
                # Case of an array
                else:
                    scalarVariable = False
                    formatted_entry = '_'.join(entry)  # we just put all together in a "_' joined string

            # Case single scalar
            if scalarVariable:

                # Case with error quantified # TODO add uncertainty protocol for table
                # if isinstance(entry, UFloat):
                #     formatted_entry = round_sig(nominal_values(entry), rounddig,
                #                                 scien_notation=scientific_notation) + r'$\pm$' + round_sig(
                #         std_devs(entry), rounddig_er, scien_notation=scientific_notation)

                # Case single float
                if np.isnan(entry):
                    formatted_entry = nan_format

                # Case single float
                else:
                    formatted_entry = numberStringFormat(entry, rounddig)
    else:
        # None entry is converted to None
        formatted_entry = 'None'

    return formatted_entry


def load_lines_log(lineslog_address, ext=None):
    """
    This function attemps several approaches to import a lines log from a sheet or text file lines as a pandas
    dataframe
    :param lineslog_address: String with the location of the input lines log file
    :return lineslogDF: Dataframe with line labels as index and default column headers (wavelength, w1 to w6)
    """

    # Fits file:
    if str(lineslog_address).endswith('.fits'):
        lineslogDF = Table.read(lineslog_address, ext, character_as_bytes=False).to_pandas()
        lineslogDF.set_index('index', inplace=True)

    else:
        # Text file # TODO add fits and excel formats
        try:
            lineslogDF = pd.read_csv(lineslog_address, delim_whitespace=True, header=0, index_col=0)
        except ValueError:

            # Excel file
            try:
                lineslogDF = pd.read_excel(lineslog_address, sheet_name=0, header=0, index_col=0)
            except ValueError:
                print(f'- ERROR: Could not open lines log at: {lineslog_address}')

    return lineslogDF


def save_line_log(linelog, file_address, file_type='txt', ext=None, fits_header=None):

    # Default txt log with the complete information
    if file_type == 'txt':
        with open(f'{file_address}.{file_type}', 'wb') as output_file:
            string_DF = linelog.to_string()
            output_file.write(string_DF.encode('UTF-8'))

    # Pdf fluxes table
    elif file_type == 'pdf':
        table_fluxes(linelog, file_address)

    # Linelog in a fits file
    elif file_type == 'fits':
        if isinstance(linelog, pd.DataFrame):
            lineLogHDU = lineslog_to_HDU(linelog, ext_name=ext, header_dict=fits_header)

            fits_address = Path(f'{file_address}.{file_type}')
            if fits_address.is_file():
                try:
                    fits.update(fits_address, data=lineLogHDU.data, header=lineLogHDU.header, extname=lineLogHDU.name, verify=True)
                except KeyError:
                    fits.append(fits_address, data=lineLogHDU.data, header=lineLogHDU.header, extname=lineLogHDU.name)
            else:
                hdul = fits.HDUList([fits.PrimaryHDU(), lineLogHDU])
                hdul.writeto(fits_address, overwrite=True, output_verify='fix')

    # Default log in excel format
    elif file_type == 'xlsx' or file_type == 'xls':
        sheet_name = ext if ext is not None else 'Sheet1'
        linelog.to_excel(f'{file_address}.{file_type}', sheet_name=sheet_name)

    else:
        print(f"--WARNING: output file extension {file_type} was not recognised. Exiting program")
        exit()

    return


def lineslog_to_HDU(log_DF, ext_name=None, column_types={}, header_dict={}):

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


def table_fluxes(lines_df, table_address, pyneb_rc=None, scaleTable=1000, table_type='pdf'):

    if table_type == 'pdf':
        output_address = f'{table_address}'
    if table_type == 'txt-ascii':
        output_address = f'{table_address}.txt'

    # Measure line fluxes
    pdf = PdfMaker()
    pdf.create_pdfDoc(pdf_type='table')
    pdf.pdf_insert_table(FLUX_TEX_TABLE_HEADERS)

    # Dataframe as container as a txt file
    tableDF = pd.DataFrame(columns=FLUX_TXT_TABLE_HEADERS[1:])

    # Normalization line
    if 'H1_4861A' in lines_df.index:
        flux_Hbeta = lines_df.loc['H1_4861A', 'intg_flux']
    else:
        flux_Hbeta = scaleTable

    obsLines = lines_df.index.values
    for lineLabel in obsLines:

        label_entry = lines_df.loc[lineLabel, 'latexLabel']
        wavelength = lines_df.loc[lineLabel, 'wavelength']
        eqw, eqwErr = lines_df.loc[lineLabel, 'eqw'], lines_df.loc[lineLabel, 'eqw_err']

        flux_intg = lines_df.loc[lineLabel, 'intg_flux'] / flux_Hbeta * scaleTable
        flux_intgErr = lines_df.loc[lineLabel, 'intg_err'] / flux_Hbeta * scaleTable
        flux_gauss = lines_df.loc[lineLabel, 'gauss_flux'] / flux_Hbeta * scaleTable
        flux_gaussErr = lines_df.loc[lineLabel, 'gauss_err'] / flux_Hbeta * scaleTable

        if (lines_df.loc[lineLabel, 'blended_label'] != 'None') and ('_m' not in lineLabel):
            flux, fluxErr = flux_gauss, flux_gaussErr
            label_entry = label_entry + '$_{gauss}$'
        else:
            flux, fluxErr = flux_intg, flux_intgErr

        # Correct the flux
        if pyneb_rc is not None:
            corr = pyneb_rc.getCorrHb(wavelength)
            intensity, intensityErr = flux * corr, fluxErr * corr
            intensity_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(intensity, intensityErr)
        else:
            intensity, intensityErr = '-', '-'
            intensity_entry = '-'

        eqw_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(eqw, eqwErr)
        flux_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(flux, fluxErr)

        # Add row of data
        tex_row_i = [label_entry, eqw_entry, flux_entry, intensity_entry]
        txt_row_i = [label_entry, eqw, eqwErr, flux, fluxErr, intensity, intensityErr]

        lastRow_check = True if lineLabel == obsLines[-1] else False
        pdf.addTableRow(tex_row_i, last_row=lastRow_check)
        tableDF.loc[lineLabel] = txt_row_i[1:]

    if pyneb_rc is not None:

        # Data last rows
        row_Hbetaflux = [r'$H\beta$ $(erg\,cm^{-2} s^{-1} \AA^{-1})$',
                         '',
                         flux_Hbeta,
                         flux_Hbeta * pyneb_rc.getCorr(4861)]

        row_cHbeta = [r'$c(H\beta)$',
                      '',
                      float(pyneb_rc.cHbeta),
                      '']
    else:
        # Data last rows
        row_Hbetaflux = [r'$H\beta$ $(erg\,cm^{-2} s^{-1} \AA^{-1})$',
                         '',
                         flux_Hbeta,
                         '-']

        row_cHbeta = [r'$c(H\beta)$',
                      '',
                      '-',
                      '']

    pdf.addTableRow(row_Hbetaflux, last_row=False)
    pdf.addTableRow(row_cHbeta, last_row=False)
    tableDF.loc[row_Hbetaflux[0]] = row_Hbetaflux[1:] + [''] * 3
    tableDF.loc[row_cHbeta[0]] = row_cHbeta[1:] + [''] * 3

    # Format last rows
    pdf.table.add_hline()
    pdf.table.add_hline()

    # Save the pdf table
    if table_type == 'pdf':
        try:
            pdf.generate_pdf(table_address, clean_tex=True)
        except:
            print('-- PDF compilation failure')

    # Save the txt table
    if table_type == 'txt-ascii':
        with open(output_address, 'wb') as output_file:
            string_DF = tableDF.to_string()
            string_DF = string_DF.replace('$', '')
            output_file.write(string_DF.encode('UTF-8'))

    return


class PdfMaker:

    def __init__(self):

        self.pdf_type = None
        self.pdf_geometry_options = {'right': '1cm',
                                     'left': '1cm',
                                     'top': '1cm',
                                     'bottom': '2cm'}
        self.table = None
        self.theme_table = None

        # TODO add dictionary with numeric formats for tables depending on the variable

    def create_pdfDoc(self, pdf_type=None, geometry_options=None, document_class=u'article', theme='white'):

        # TODO integrate this into the init
        # Case for a complete .pdf or .tex
        self.theme_table = theme

        if pdf_type is not None:

            self.pdf_type = pdf_type

            # Update the geometry if necessary (we coud define a dictionary distinction)
            if pdf_type == 'graphs':
                pdf_format = {'landscape': 'true'}
                self.pdf_geometry_options.update(pdf_format)

            elif pdf_type == 'table':
                pdf_format = {'landscape': 'true',
                              'paperwidth': '30in',
                              'paperheight': '30in'}
                self.pdf_geometry_options.update(pdf_format)

            if geometry_options is not None:
                self.pdf_geometry_options.update(geometry_options)

            # Generate the doc
            self.pdfDoc = pylatex.Document(documentclass=document_class, geometry_options=self.pdf_geometry_options)

            if theme == 'dark':
                self.pdfDoc.append(pylatex.NoEscape('\definecolor{background}{rgb}{0.169, 0.169, 0.169}'))
                self.pdfDoc.append(pylatex.NoEscape('\definecolor{foreground}{rgb}{0.702, 0.780, 0.847}'))
                self.pdfDoc.append(pylatex.NoEscape(r'\arrayrulecolor{foreground}'))

            if pdf_type == 'table':
                self.pdfDoc.packages.append(pylatex.Package('preview', options=['active', 'tightpage', ]))
                self.pdfDoc.packages.append(pylatex.Package('hyperref', options=['unicode=true', ]))
                self.pdfDoc.append(pylatex.NoEscape(r'\pagenumbering{gobble}'))
                self.pdfDoc.packages.append(pylatex.Package('nicefrac'))
                self.pdfDoc.packages.append(pylatex.Package('siunitx'))
                self.pdfDoc.packages.append(pylatex.Package('makecell'))
                # self.pdfDoc.packages.append(pylatex.Package('color', options=['usenames', 'dvipsnames', ]))  # Package to crop pdf to a figure
                self.pdfDoc.packages.append(pylatex.Package('colortbl', options=['usenames', 'dvipsnames', ]))  # Package to crop pdf to a figure
                self.pdfDoc.packages.append(pylatex.Package('xcolor', options=['table']))

            elif pdf_type == 'longtable':
                self.pdfDoc.append(pylatex.NoEscape(r'\pagenumbering{gobble}'))

        return

    def pdf_create_section(self, caption, add_page=False):

        with self.pdfDoc.create(pylatex.Section(caption)):
            if add_page:
                self.pdfDoc.append(pylatex.NewPage())

    def add_page(self):

        self.pdfDoc.append(pylatex.NewPage())

        return

    def pdf_insert_image(self, image_address, fig_loc='htbp', width=r'1\textwidth'):

        with self.pdfDoc.create(pylatex.Figure(position='h!')) as fig_pdf:
            fig_pdf.add_image(image_address, pylatex.NoEscape(width))

        return

    def pdf_insert_table(self, column_headers=None, table_format=None, addfinalLine=True, color_font=None,
                         color_background=None):

        # Set the table format
        if table_format is None:
            table_format = 'l' + 'c' * (len(column_headers) - 1)

        # Case we want to insert the table in a pdf
        if self.pdf_type != None:

            if self.pdf_type == 'table':
                self.pdfDoc.append(pylatex.NoEscape(r'\begin{preview}'))

                # Initiate the table
                with self.pdfDoc.create(pylatex.Tabular(table_format)) as self.table:
                    if column_headers != None:
                        self.table.add_hline()
                        # self.table.add_row(list(map(str, column_headers)), escape=False, strict=False)
                        output_row = list(map(partial(format_for_table), column_headers))

                        # if color_font is not None:
                        #     for i, item in enumerate(output_row):
                        #         output_row[i] = NoEscape(r'\color{{{}}}{}'.format(color_font, item))
                        #
                        # if color_background is not None:
                        #     for i, item in enumerate(output_row):
                        #         output_row[i] = NoEscape(r'\cellcolor{{{}}}{}'.format(color_background, item))

                        if (color_font is not None) or (self.theme_table != 'white'):
                            if self.theme_table == 'dark' and color_font is None:
                                color_font = 'foreground'

                            for i, item in enumerate(output_row):
                                output_row[i] = pylatex.NoEscape(r'\color{{{}}}{}'.format(color_font, item))

                        if (color_background is not None) or (self.theme_table != 'white'):
                            if self.theme_table == 'dark' and color_background is None:
                                color_background = 'background'

                            for i, item in enumerate(output_row):
                                output_row[i] = pylatex.NoEscape(r'\cellcolor{{{}}}{}'.format(color_background, item))

                        self.table.add_row(output_row, escape=False, strict=False)
                        if addfinalLine:
                            self.table.add_hline()

            elif self.pdf_type == 'longtable':

                # Initiate the table
                with self.pdfDoc.create(pylatex.LongTable(table_format)) as self.table:
                    if column_headers != None:
                        self.table.add_hline()
                        self.table.add_row(list(map(str, column_headers)), escape=False)
                        if addfinalLine:
                            self.table.add_hline()

        # Table .tex without preamble
        else:
            self.table = pylatex.Tabu(table_format)
            if column_headers != None:
                self.table.add_hline()
                # self.table.add_row(list(map(str, column_headers)), escape=False, strict=False)
                output_row = list(map(partial(format_for_table), column_headers))
                self.table.add_row(output_row, escape=False, strict=False)
                if addfinalLine:
                    self.table.add_hline()

        return

    def pdf_insert_longtable(self, column_headers=None, table_format=None):

        # Set the table format
        if table_format is None:
            table_format = 'l' + 'c' * (len(column_headers) - 1)

        # Case we want to insert the table in a pdf
        if self.pdf_type != None:

            if self.pdf_type == 'table':
                self.pdfDoc.append(pylatex.NoEscape(r'\begin{preview}'))

                # Initiate the table
            with self.pdfDoc.create(pylatex.Tabu(table_format)) as self.table:
                if column_headers != None:
                    self.table.add_hline()
                    self.table.add_row(map(str, column_headers), escape=False)
                    self.table.add_hline()

                    # Table .tex without preamble
        else:
            self.table = pylatex.LongTable(table_format)
            if column_headers != None:
                self.table.add_hline()
                self.table.add_row(list(map(str, column_headers)), escape=False)
                self.table.add_hline()

    def addTableRow(self, input_row, row_format='auto', rounddig=4, rounddig_er=None, last_row=False, color_font=None,
                    color_background=None):

        # Default formatting
        if row_format == 'auto':
            output_row = list(map(partial(format_for_table, rounddig=rounddig), input_row))

        # TODO clean this theming to default values
        if (color_font is not None) or (self.theme_table != 'white'):
            if self.theme_table == 'dark' and color_font is None:
                color_font = 'foreground'

            for i, item in enumerate(output_row):
                output_row[i] = pylatex.NoEscape(r'\color{{{}}}{}'.format(color_font, item))

        if (color_background is not None) or (self.theme_table != 'white'):
            if self.theme_table == 'dark' and color_background is None:
                color_background = 'background'

            for i, item in enumerate(output_row):
                output_row[i] = pylatex.NoEscape(r'\cellcolor{{{}}}{}'.format(color_background, item))

        # Append the row
        self.table.add_row(output_row, escape=False, strict=False)

        # Case of the final row just add one line
        if last_row:
            self.table.add_hline()

    def fig_to_pdf(self, label=None, fig_loc='htbp', width=r'1\textwidth', add_page=False, *args, **kwargs):

        with self.pdfDoc.create(pylatex.Figure(position=fig_loc)) as plot:
            plot.add_plot(width=pylatex.NoEscape(width), placement='h', *args, **kwargs)

            if label is not None:
                plot.add_caption(label)

        if add_page:
            self.pdfDoc.append(pylatex.NewPage())

    def generate_pdf(self, output_address, clean_tex=True):

        if self.pdf_type is None:
            self.table.generate_tex(str(output_address))

        else:
            if self.pdf_type == 'table':
                self.pdfDoc.append(pylatex.NoEscape(r'\end{preview}'))
            self.pdfDoc.generate_pdf(filepath=str(output_address), clean_tex=clean_tex, compiler='pdflatex')

        return
