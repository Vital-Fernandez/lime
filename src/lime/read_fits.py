from pathlib import Path
import requests

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import logging

from .io import LiMe_Error
from urllib.parse import urlparse

try:
    import requests
    requests_check = True
except ImportError:
    requests_check = False


_logger = logging.getLogger('LiMe')

DESI_SPECTRA_BANDS = ('B', 'R', 'Z')


SPECTRUM_FITS_PARAMS = {'nirspec': {'redshift': None, 'norm_flux': None, 'inst_FWHM': np.nan,
                                'units_wave': 'um', 'units_flux': 'mJy', 'pixel_mask': None, 'id_label': None},

                        'isis': {'redshift': None, 'norm_flux': None, 'inst_FWHM': np.nan,
                                 'units_wave': 'um', 'units_flux': 'mJy', 'pixel_mask': None, 'id_label': None},

                        'osiris': {'redshift': None, 'norm_flux': None, 'inst_FWHM': np.nan, 'units_wave': 'A',
                                   'units_flux': 'Flam', 'pixel_mask': None, 'id_label': None},

                        'sdss': {'redshift': None, 'norm_flux': None, 'inst_FWHM': np.nan, 'units_wave': 'A',
                                 'units_flux': 'Flam', 'pixel_mask': None, 'id_label': None},

                        'desi': {'redshift': None, 'norm_flux': None, 'inst_FWHM': np.nan, 'units_wave': 'A',
                             'units_flux': 'Flam', 'pixel_mask': None, 'id_label': None}

                        }

CUBE_FITS_PARAMS = {'manga': {'redshift': None, 'norm_flux': None, 'inst_FWHM': np.nan, 'units_wave': 'A',
                              'units_flux': 'Flam', 'pixel_mask': None, 'id_label': None},

                    'muse':  {'redshift': None, 'norm_flux': None, 'inst_FWHM': np.nan, 'units_wave': 'A',
                              'units_flux': 'Flam', 'pixel_mask': None, 'id_label': None},

                    'megara': {'redshift': None, 'norm_flux': None, 'inst_FWHM': np.nan, 'units_wave': 'A',
                               'units_flux': 'Jy', 'pixel_mask': None, 'id_label': None}
                    }


def check_url_status(url):

    url_check = False

    if requests_check:
        try:
            # Check if the response status code is 200 (OK)
            response = requests.get(url)
            if response.status_code != 200:
                _logger.warning(f'Not file found at {url}')
            else:
                url_check = True

        except requests.exceptions.RequestException as e:
            # Handle any exceptions (like connection errors)
            _logger.warning(f"Error checking {url}: {e}")
    else:
        _logger.warning("The requests package is not installed. LiMe won't be able to check url files")

    return url_check


def desi_bands_reconstruction(bands_dict, desi_bands=DESI_SPECTRA_BANDS):

    wave, flux, ivar = None, None, None

    for b in desi_bands:

        band = bands_dict[b]
        wave_b, flux_b, ivar_b = band['wave'], band['flux'], band['ivar']

        if wave is None:
            wave, flux, ivar = wave_b, flux_b, ivar_b
        else:
            idcs_match = (wave_b < wave[-1]) & (wave_b < wave_b[-1])
            median_idx = int(np.sum(idcs_match) / 2)
            wave = np.append(wave, wave_b[median_idx:])
            flux = np.append(flux, flux_b[median_idx:])
            ivar = np.append(ivar, ivar_b[median_idx:])

    # Reconstruct the bands
    err_flux = np.sqrt(1/ivar)

    return wave, flux, err_flux


def check_fits_source(fits_source, lime_object=None, load_function=None):

    spectrum_type = True

    # Lower case the source
    if fits_source is not None:

        fits_source = fits_source.lower()

        # Check if observation matches the LiMe type
        if lime_object is not None:
            valid_check = False

            if (lime_object == 'Spectrum') and (fits_source in SPECTRUM_FITS_PARAMS.keys()):
                valid_check = True

            elif (lime_object == 'Cube') and (fits_source in CUBE_FITS_PARAMS.keys()):
                valid_check = True
                spectrum_type = False

            elif (lime_object == 'Sample') and (fits_source in list(SPECTRUM_FITS_PARAMS.keys()) + list(CUBE_FITS_PARAMS.keys())):
                valid_check = True

                if fits_source in list(CUBE_FITS_PARAMS.keys()):
                    spectrum_type = False

            else:
                if lime_object not in ['Spectrum', 'Cube', 'Sample']:
                    raise LiMe_Error(f'Input {lime_object} is not recognized. Please use a LiMe spectrum or Cube')

            if valid_check is False:
                if fits_source not in list(SPECTRUM_FITS_PARAMS.keys()) + list(CUBE_FITS_PARAMS.keys()):
                    raise LiMe_Error(f'Input "{fits_source}" is not recognized. LiMe currently only recognizes: '
                                     f'{list(SPECTRUM_FITS_PARAMS.keys())} and {list(CUBE_FITS_PARAMS.keys())}')

    else:

        if load_function is None:
            raise LiMe_Error(f'Please introduce fits file instrument or a load function to import the fits file as a '
                             f'LiMe observation')

    return fits_source, spectrum_type


def check_fits_location(fits_address, lime_object=None):

    # Input address
    if fits_address is not None:

        # Special case for sample files reading
        if lime_object == 'Sample':
            if fits_address is not None:
                fits_folder = Path(fits_address)
                if not fits_folder.is_dir():
                    raise LiMe_Error(f'LiMe could not find root folder ({fits_address}) for the Sample creation')
                else:
                    output = fits_folder, False
            else:
                output = None, False

        # File address or url
        else:

            # Physical file:
            fits_path = Path(fits_address)

            if fits_path.is_file():
                output = fits_path, False

            # Online file
            else:

                # Check valid address
                fits_url = urlparse(fits_address)
                if all([fits_url.scheme, fits_url.netloc]):
                    url_validator(fits_address)
                    output = fits_address, True
                else:
                    raise LiMe_Error(f'LiMe could not find a file at "{fits_address}".\nIf you are specifying a physical '
                                     f'file please check the file location.\nIf you are introducing a url please include the'
                                     f' complete address')

    # Null address
    else:
        output = Path(""), False

    return output


def check_load_function():

    return

def url_validator(url):

    valid_output = False
    message = None
    try:
        response = requests.options(url)
        if response.ok:  # alternatively you can use response.status_code == 200
            valid_output = True
        else:
            message = f"Failure - API is accessible but sth is not right. Response code : {response.status_code}"

    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
        message = f"Failure - Unable to establish connection: {e}."

    except Exception as e:
        message = f"Failure - Unknown error occurred: {e}."

    if valid_output is False:
        raise LiMe_Error(f'LiMe could not access the url: {url}\n{message}')

    return


def check_fits_instructions(fits_source, online_provider=False):

    if fits_source is not None:

        fits_manager = OpenFits if online_provider is False else OpenFitsSurvey

        # Check LiMe can handle source type
        if hasattr(fits_manager, fits_source):
            fits_reader = getattr(fits_manager, fits_source)
        else:
            source_type = 'instrument' if online_provider is False else 'survey'
            raise LiMe_Error(f'Input {source_type} "{fits_source}" is not recognized. LiMe observation cannot be created.')

    else:
        fits_reader = None

    # # Check for url location for surveys function
    # if online_provider:
    #     if hasattr(UrlFitsSurvey, fits_source):
    #         url_locator = getattr(UrlFitsSurvey, fits_source)
    #     else:
    #         raise LiMe_Error(f'Input {fits_source} does not have a url manager for LiMe could not be created.')
    # else:
    #     url_locator = None

    return fits_reader


def load_fits(fits_address, data_ext_list=None, hdr_ext_list=None, url_check=False):

    """

    This method opens a fits file and returns the request extensions data and headers.

    :param fits_address: File location address for the observation .fits file.
    :type fits_address: str, Path

    :param data_ext_list: Data extension number or name to extract from the .fits file.
    :type fits_address: int, str or list of either, optional

    :param hdr_ext_list: header extension number or name to extract from the .fits file.
    :type hdr_ext_list: int, str or list of either, optional

    :return: list of extensions data, list of extensions headers

    """

    # Parse as numpy array
    data_ext_list, hdr_ext_list = np.atleast_1d(data_ext_list), np.atleast_1d(hdr_ext_list)

    # Open the fits file
    if url_check is False:

        data_list, header_list = [], []
        with fits.open(fits_address) as hdu_list:

            # Get data
            for ext in data_ext_list:
                if ext is not None:
                    try:
                        data_list.append(hdu_list[ext].data)
                    except KeyError:
                        _logger.warning(f'Extension "{ext}" data was not found in input fits file: {fits_address}')

            # Get header
            for ext in hdr_ext_list:
                if ext is not None:
                    try:
                        header_list.append(hdu_list[ext].header)
                    except KeyError:
                        _logger.warning(f'Extension "{ext}" header was not found in input fits file: {fits_address}')

    # Open the url file
    else:
        data_list, header_list = None, None

    return data_list, header_list


class OpenFits:

    def __init__(self, file_address, file_source=None, load_function=None, lime_object=None):

        self.source = None
        self.obs_args = None
        self.online_check = False
        self.fits_reader = None
        self.spectrum_check = None
        self.file_address = None

        # Check the fits source
        self.source, self.spectrum_check = check_fits_source(file_source, lime_object, load_function)

        # Check file or url
        self.file_address, self.online_check = check_fits_location(file_address, lime_object)

        # Recover the function to open the fits file
        self.fits_reader = check_fits_instructions(self.source, self.online_check)

        return

    def parse_data_from_file(self, file_address):

        # Read the fits data
        wave_array, flux_array, err_array, header_list, fits_params = self.fits_reader(file_address)

        # Construct attributes for LiMe object
        fits_args = {'input_wave': wave_array, 'input_flux': flux_array, 'input_err': err_array, **fits_params}

        return fits_args

    # def load_function(self, log_df, id_spec, root_address, **kwargs):
    #
    #     # Run the load function
    #     load_function_output = self.load_function(log_df, id_spec, root_address)
    #
    #     # Proceed to create LiMe object if necessary
    #     if isinstance(load_function_output, dict):
    #         obs_args = {**load_function_output, **kwargs}
    #         obs = Spectrum(**obs_args) if self.spectrum_check else Cube(**obs_args)
    #     else:
    #         obs = load_function_output
    #
    #     return obs

    def default_file_parser(self, log_df, id_spec, root_address, **kwargs):

        # Get address of observation
        file_spec = root_address/id_spec[log_df.index.names.index('file')]

        # Get observation data
        fits_args = self.fits_reader(file_spec)

        return fits_args

    @staticmethod
    def nirspec(fits_address, data_ext_list=1, hdr_ext_list=(0, 1)):

        """

        This method returns the spectrum array data and headers from a NIRSPEC observation.

        The function returns numpy arrays with the wavelength, flux and uncertainty flux (if available this is the
        standard deviation available), a list with the requested headers and a dictionary with the parameters to
        construct a LiMe Spectrum or Cube. These parameters include the observation wavelength/flux units, normalization
        and wcs from the input fits file.

        :param fits_address: File location address for the observation .fits file.
        :type fits_address: str, Path

        :param data_ext_list: Data extension number or name to extract from the .fits file.
        :type fits_address: int, str or list of either, optional

        :param hdr_ext_list: header extension number or name to extract from the .fits file.
        :type hdr_ext_list: int, str or list of either, optional

        :return: wavelength array, flux array, uncertainty array, header list, observation parameter dict

        """

        # Get data table and header dict lists
        data_list, header_list = load_fits(fits_address, data_ext_list, hdr_ext_list, url_check=False)

        # Re-construct spectrum arrays
        wave_array, flux_array, err_array = data_list[0]['WAVELENGTH'], data_list[0]['FLUX'], data_list[0]['FLUX_ERROR']

        # Spectrum properties
        params_dict = SPECTRUM_FITS_PARAMS['nirspec']

        return wave_array, flux_array, err_array, header_list, params_dict

    @staticmethod
    def isis(fits_address, data_ext_list=0, hdr_ext_list=0):

        """

        This method returns the spectrum array data and headers from a ISIS observation.

        The function returns numpy arrays with the wavelength, flux and uncertainty flux (if available this is the
        standard deviation available), a list with the requested headers and a dictionary with the parameters to
        construct a LiMe Spectrum. These parameters include the observation wavelength/flux units, normalization and wcs
        from the input fits file.

        :param fits_address: File location address for the observation .fits file.
        :type fits_address: str, Path

        :param data_ext_list: Data extension number or name to extract from the .fits file.
        :type fits_address: int, str or list of either, optional

        :param hdr_ext_list: header extension number or name to extract from the .fits file.
        :type hdr_ext_list: int, str or list of either, optional

        :return: wavelength array, flux array, uncertainty array, header list, observation parameter dict

        """

        # Get data table and header dict lists
        data_list, header_list = load_fits(fits_address, data_ext_list, hdr_ext_list, url_check=False)

        # Re-construct spectrum arrays
        w_min, dw, pixels = header_list[0]['CRVAL1'], header_list[0]['CD1_1'], header_list[0]['NAXIS1']
        w_max = w_min + dw * pixels

        wave_array = np.linspace(w_min, w_max, pixels, endpoint=False)

        # Iraf 2 array flux/err
        if len(data_list[0].shape) == 2:
            flux_array, err_array = data_list[0][0, :], data_list[0][1, :]
        else:
            flux_array, err_array = data_list[0], None

        # Spectrum properties
        params_dict = SPECTRUM_FITS_PARAMS['isis']

        return wave_array, flux_array, err_array, header_list, params_dict

    @staticmethod
    def osiris(fits_address, data_ext_list=0, hdr_ext_list=0):

        """

        This method returns the spectrum array data and headers from a OSIRIS observation.

        The function returns numpy arrays with the wavelength, flux and uncertainty flux (if available this is the
        standard deviation available), a list with the requested headers and a dictionary with the parameters to
        construct a LiMe Spectrum. These parameters include the observation wavelength/flux units, normalization and wcs
        from the input fits file.

        :param fits_address: File location address for the observation .fits file.
        :type fits_address: str, Path

        :param data_ext_list: Data extension number or name to extract from the .fits file.
        :type fits_address: int, str or list of either, optional

        :param hdr_ext_list: header extension number or name to extract from the .fits file.
        :type hdr_ext_list: int, str or list of either, optional

        :return: wavelength array, flux array, uncertainty array, header list, observation parameter dict

        """

        # Get data table and header dict lists
        data_list, header_list = load_fits(fits_address, data_ext_list, hdr_ext_list, url_check=False)

        # Re-construct spectrum arrays
        w_min, dw, pixels = header_list[0]['CRVAL1'], header_list[0]['CD1_1'], header_list[0]['NAXIS1']
        w_max = w_min + dw * pixels

        wave_array = np.linspace(w_min, w_max, pixels, endpoint=False)

        # Iraf 2 array flux/err
        if len(data_list[0].shape) == 2:
            flux_array, err_array = data_list[0][0, :], data_list[0][1, :]
        else:
            flux_array, err_array = data_list[0], None

        # Spectrum properties
        params_dict = SPECTRUM_FITS_PARAMS['osiris']

        return wave_array, flux_array, err_array, header_list, params_dict

    @staticmethod
    def sdss(fits_address, data_ext_list=(1, 2), hdr_ext_list=0):

        """

        This method returns the spectrum array data and headers from a SDSS observation.

        The function returns numpy arrays with the wavelength, flux and uncertainty flux (if available this is the
        standard deviation available), a list with the requested headers and a dictionary with the parameters to
        construct a LiMe Spectrum. These parameters include the observation wavelength/flux units, normalization and wcs
        from the input fits file.

        :param fits_address: File location address for the observation .fits file.
        :type fits_address: str, Path

        :param data_ext_list: Data extension number or name to extract from the .fits file.
        :type fits_address: int, str or list of either, optional

        :param hdr_ext_list: header extension number or name to extract from the .fits file.
        :type hdr_ext_list: int, str or list of either, optional

        :return: wavelength array, flux array, uncertainty array, header list, observation parameter dict

        """

        # Get data table and header dict lists
        data_list, header_list = load_fits(fits_address, data_ext_list, hdr_ext_list, url_check=False)

        # Re-construct spectrum arrays
        wave_array = 10.0 ** data_list[0]['loglam']
        flux_array = data_list[0]['flux']
        err_array = None

        # Spectrum properties
        params_dict = SPECTRUM_FITS_PARAMS['sdss']

        return wave_array, flux_array, err_array, header_list, params_dict

    @staticmethod
    def manga(fits_address, data_ext_list=('WAVE', 'FLUX', 'IVAR'), hdr_ext_list=('FLUX')):

        """

        This method returns the spectrum array data and headers from a MANGA observation.

        The function returns numpy arrays with the wavelength, flux and uncertainty flux (if available this is the
        standard deviation available), a list with the requested headers and a dictionary with the parameters to
        construct a LiMe Cube. These parameters include the observation wavelength/flux units, normalization and wcs
        from the input fits file.

        :param fits_address: File location address for the observation .fits file.
        :type fits_address: str, Path

        :param data_ext_list: Data extension number or name to extract from the .fits file.
        :type fits_address: int, str or list of either, optional

        :param hdr_ext_list: header extension number or name to extract from the .fits file.
        :type hdr_ext_list: int, str or list of either, optional

        :return: wavelength array, flux array, uncertainty array, header list, observation parameter dict

        """

        # Get data table and header dict lists
        data_list, header_list = load_fits(fits_address, data_ext_list, hdr_ext_list, url_check=False)

        # Re-construct spectrum arrays
        wave_array = data_list[0]
        flux_cube = data_list[1]
        ivar_cube = data_list[1]

        pixel_mask_cube = ivar_cube == 0
        pixel_mask_cube = pixel_mask_cube.reshape(ivar_cube.shape)
        err_cube = np.sqrt(1 / np.ma.masked_array(ivar_cube, pixel_mask_cube))

        wcs = WCS(header_list[0])

        # Fits properties
        fits_params = {**CUBE_FITS_PARAMS['manga'], 'pixel_mask': pixel_mask_cube, 'wcs': wcs}

        return wave_array, flux_cube, err_cube, header_list, fits_params

    @staticmethod
    def muse(fits_address, data_ext_list=(1, 2), hdr_ext_list=1):

        """

        This method returns the spectrum array data and headers from a MUSE observation.

        The function returns numpy arrays with the wavelength, flux and uncertainty flux (if available this is the
        standard deviation available), a list with the requested headers and a dictionary with the parameters to
        construct a LiMe Cube. These parameters include the observation wavelength/flux units, normalization and wcs
        from the input fits file.

        :param fits_address: File location address for the observation .fits file.
        :type fits_address: str, Path

        :param data_ext_list: Data extension number or name to extract from the .fits file.
        :type fits_address: int, str or list of either, optional

        :param hdr_ext_list: header extension number or name to extract from the .fits file.
        :type hdr_ext_list: int, str or list of either, optional

        :return: wavelength array, flux array, uncertainty array, header list, observation parameter dict

        """

        # Get data table and header dict lists
        data_list, header_list = load_fits(fits_address, data_ext_list, hdr_ext_list, url_check=False)

        # Re-construct spectrum arrays
        w_min, dw, pixels = header_list[0]['CRVAL3'], header_list[0]['CD3_3'], header_list[0]['NAXIS3']
        w_max = w_min + dw * pixels
        wave_array = np.linspace(w_min, w_max, pixels, endpoint=False)

        flux_cube = data_list[0]
        var_cube = data_list[1]
        err_cube = np.sqrt(var_cube)
        pixel_mask_cube = np.isnan(flux_cube)

        wcs = WCS(header_list[0])

        # Fits properties
        fits_params = {**CUBE_FITS_PARAMS['muse'], 'pixel_mask': pixel_mask_cube, 'wcs': wcs}

        return wave_array, flux_cube, err_cube, header_list, fits_params

    @staticmethod
    def megara(fits_address, data_ext_list=0, hdr_ext_list=(0, 1)):

        """

        This method returns the spectrum array data and headers from a MUSE observation.

        The function returns numpy arrays with the wavelength, flux and uncertainty flux (if available this is the
        standard deviation available), a list with the requested headers and a dictionary with the parameters to
        construct a LiMe Cube. These parameters include the observation wavelength/flux units, normalization and wcs
        from the input fits file.

        :param fits_address: File location address for the observation .fits file.
        :type fits_address: str, Path

        :param data_ext_list: Data extension number or name to extract from the .fits file.
        :type fits_address: int, str or list of either, optional

        :param hdr_ext_list: header extension number or name to extract from the .fits file.
        :type hdr_ext_list: int, str or list of either, optional

        :return: wavelength array, flux array, uncertainty array, header list, observation parameter dict

        """

        # Get data table and header dict lists
        data_list, header_list = load_fits(fits_address, data_ext_list, hdr_ext_list, url_check=False)

        # Re-construct spectrum arrays
        w_min, dw, pixels = header_list[0]['CRVAL3'], header_list[0]['CDELT3'], header_list[0]['NAXIS3']
        w_max = w_min + dw * pixels
        wave_array = np.linspace(w_min, w_max, pixels, endpoint=False)

        flux_cube = data_list[0]
        err_cube = None
        pixel_mask_cube = None

        wcs = WCS(header_list[1])

        # Fits properties
        fits_params = {**CUBE_FITS_PARAMS['megara'], 'pixel_mask': pixel_mask_cube, 'wcs': wcs}

        return wave_array, flux_cube, err_cube, header_list, fits_params



class OpenFitsSurvey:

    @staticmethod
    def url(fits_address, data_ext_list, hdr_ext_list=None):

        return

    @staticmethod
    def desi(fits_address, data_ext_list, hdr_ext_list=None):

        # Reshape into an array if necessary
        data_ext_list, hdr_ext_list = np.atleast_1d(data_ext_list), np.atleast_1d(hdr_ext_list)

        # Read the url data
        spectra_dict = {}

        with fits.open(fits_address, use_fsspec=True) as hdulist:

            file_idtargets = hdulist["FIBERMAP"].data['TARGETID']
            obj_idrows = np.where(np.isin(file_idtargets, data_ext_list))[0]

            for i, id_obj in enumerate(obj_idrows):
                spectra_dict = {}
                for band in ('B', 'R', 'Z'):
                    spectra_dict[band] = {'wave': hdulist[f'{band}_WAVELENGTH'].section[:],
                                          'flux': hdulist[f'{band}_FLUX'].section[obj_idrows[i], :],
                                          'ivar': hdulist[f'{band}_IVAR'].section[obj_idrows[i], :]}

        # Re-construct spectrum arrays
        wave, flux, err_flux = desi_bands_reconstruction(spectra_dict)

        # Spectrum properties
        params_dict = SPECTRUM_FITS_PARAMS['desi'].copy()

        return wave, flux, err_flux, None, params_dict


class UrlFitsSurvey:

    @staticmethod
    def desi(target_id, program, release='fuji', root_url='https://data.desi.lbl.gov/public/edr/spectro/redux',
             catalogue='healpix'):

        # Select the catalogue type
        if catalogue == 'healpix':
            ref_fits = f'zall-pix-{release}.fits'
        else:
            ref_fits = f'zall-tilecumulative-{release}.fits'
        ref_fits_url = f'{root_url}/{release}/zcatalog/{ref_fits}'

        with fits.open(ref_fits_url, use_fsspec=True) as hdul:

            # Index the object
            zCatalogBin = hdul['ZCATALOG']
            idx_target = np.where((zCatalogBin.data['TARGETID'] == target_id) & (zCatalogBin.data['PROGRAM'] == program))[0]

            # Get healpix, survey and redshift
            hpx = zCatalogBin.data['HEALPIX'][idx_target]
            survey = zCatalogBin.data['SURVEY'][idx_target]
            redshift = zCatalogBin.data['Z'][idx_target]

            # Compute the url address
            url_list = []
            for i, idx in enumerate(idx_target):
                hpx_number = hpx[i]
                hpx_ref = f'{hpx_number}'[:-2]
                target_dir = f"/healpix/{survey[i]}/{program}/{hpx_ref}/{hpx_number}"
                coadd_fname = f"coadd-{survey[i]}-{program}-{hpx_number}.fits"
                url_target = f'{root_url}/{release}{target_dir}/{coadd_fname}'

                # check_url_status(url_target)
                url_list.append(url_target)

        # Additional data from the url site
        params_dict = {'redshift': redshift}

        return url_list, params_dict