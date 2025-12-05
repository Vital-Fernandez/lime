import logging
import numpy as np
from pathlib import Path
from io import IOBase
from astropy.io import fits
from astropy.wcs import WCS

from lime.io import LiMe_Error, lime_cfg
from urllib.parse import urlparse

try:
    import requests
    requests_check = True
except ImportError:
    requests_check = False


_logger = logging.getLogger('LiMe')

DESI_SPECTRA_BANDS = ('B', 'R', 'Z')

SPECTRUM_FITS_PARAMS = lime_cfg['instrument_params']['long_slit']
CUBE_FITS_PARAMS = lime_cfg['instrument_params']['cube']

def show_instrument_cfg():

    """
    Display the available instrument configurations for LiMe FITS observations.

    The information is printed to the console for user inspection.

    Returns
    -------
    None

    Notes
    -----
    - Each entry includes:
        * ``units_wave`` — wavelength units
        * ``units_flux`` — flux units
        * ``pixel_mask`` — mask handling flag
        * ``res_power`` — instrumental resolving power

    Examples
    --------
    Display all supported instrument configurations:

    >>> show_instrument_cfg()

    Example output:
        Long-slit ".fits" observation instrument configuration:
        0 osiris)  units_wave: Angstrom, units_flux: erg/s/cm2/A, pixel_mask: False, res_power: 5000

        Cube ".fits" observation instrument configuration:
        0 megaracube)  units_wave: Angstrom, units_flux: erg/s/cm2/A, pixel_mask: True, res_power: 6000
    """

    # pick widths that fit your data
    SEC_W, UW_W, UF_W, PM_W, RP_W = 15, 10, 10, 8, 8
    row_fmt = (f'{{i:>2}}\t'  
               f'{{key:<{SEC_W}}}\t'  
               f'units_wave: {{uw:<{UW_W}}}\t'
               f'units_flux: {{uf:<{UF_W}}}\t'
               f'pixel_mask: {{pm:<{PM_W}}}\t'
               f'res_power: {{rp:<{RP_W}}}')

    print('\nLong-slit ".fits" observation instrument configuration:')
    for i, (key, value) in enumerate(SPECTRUM_FITS_PARAMS.items()):
        print(row_fmt.format(i=i, key=str(key),
                             uw=str(value["units_wave"]),
                             uf=str(value["units_flux"]),
                             pm=str(value["pixel_mask"]),
                             rp=str(value["res_power"]),))

    print('\nCube ".fits" observation instrument configuration:')
    for i, (key, value) in enumerate(CUBE_FITS_PARAMS.items()):
        print(row_fmt.format(i=i, key=str(key),
                             uw=str(value["units_wave"]),
                             uf=str(value["units_flux"]),
                             pm=str(value["pixel_mask"]),
                             rp=str(value["res_power"]),))

    return


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
            _logger.warning(f'Please introduce fits file instrument or a load function to import the fits file as a '
                             f'LiMe observation')

    return fits_source, spectrum_type


def check_fits_location(fits_address, lime_object=None, source=None):

    # Input address
    if fits_address is not None:

        # Case of surveys
        if source in ['desi']:
            output = fits_address, source

        # Special case for sample files reading
        elif lime_object == 'Sample':
            if fits_address is not None:
                fits_folder = Path(fits_address)
                if not fits_folder.is_dir():
                    raise LiMe_Error(f'LiMe could not find root folder ({fits_address}) for the Sample creation')
                else:
                    output = fits_folder, False
            else:
                output = None, False

        # Streamlit BytesIO input
        elif (type(fits_address).__name__ == 'UploadedFile') or isinstance(fits_address, IOBase):
            output = fits_address, False

        # File address or url
        else:

            # Physical file:
            fits_path = Path(fits_address)

            if fits_path.is_file():
                output = fits_path, False

            # Online file
            else:

                # Check valid address
                fits_url = urlparse(str(fits_address))
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

        fits_manager = OpenFits

        # Check LiMe can handle source type
        if hasattr(fits_manager, fits_source):
            fits_reader = getattr(fits_manager, fits_source)
        else:
            source_type = 'instrument' if online_provider is False else 'survey'
            raise LiMe_Error(f'Input {source_type} "{fits_source}" is not recognized. LiMe observation cannot be created.')

    else:
        fits_reader = None

    return fits_reader


def load_txt(text_address, **kwargs):

    # Columns
    out_array = np.loadtxt(text_address, **kwargs)

    # File address
    if not type(text_address).__name__ == "UploadedFile":
        with open(text_address, "r") as f:
            lines = f.readlines()

    # Uploaded file
    else:
        lines = text_address.getvalue().decode("utf-8").splitlines()

    # Reverse loop over the lines
    params_dict = {}
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("#") or line.startswith("# LiMe"):
            break
        key, value = line[1:].split(":", 1)
        params_dict[key.strip()] = value.strip()

    # # Transform foot comments as dictionary data
    # params_dict = {}
    # with open(text_address, "r") as f:
    #
    #     # Reverse loop while the lines start by a "#"
    #     for line in reversed(f.readlines()):
    #         line = line.strip()
    #         if not line.startswith("#") or (line.startswith("# LiMe")):
    #             break
    #
    #         # Extract key-value pairs
    #         key, value = line[1:].split(":", 1)  # Split at the first ':'
    #         params_dict[key.strip()] = value.strip()

    return out_array, params_dict


def load_fits(fits_address, data_ext_list=None, hdr_ext_list=None, url_check=False):

    """
    Open a FITS file and return the requested data and header extensions.

    This method reads the input FITS file and extracts the specified data and/or header
    extensions. The user can request extensions either by their numerical index or by name.
    Both single values and lists of extensions are supported.

    Parameters
    ----------
    fits_address : str or pathlib.Path
        Path to the input observation FITS file.
    data_ext_list : int, str, or list of (int or str), optional
        Data extension(s) to extract from the FITS file. Extensions can be specified by
        index (e.g., 0, 1, 2) or by name (e.g., "SCI", "FLUX").
    hdr_ext_list : int, str, or list of (int or str), optional
        Header extension(s) to extract from the FITS file. Similar syntax applies as for
        `data_ext_list`.

    Returns
    -------
    list
        List of extracted data extensions.
    list
        List of corresponding header objects for each extracted extension.

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
        self.file_address, self.online_check = check_fits_location(file_address, lime_object, self.source)

        # Recover the function to open the fits file
        self.fits_reader = check_fits_instructions(self.source, self.online_check)

        return

    def parse_data_from_file(self, file_address, pixel_mask, **kwargs):

        # Read the fits data
        wave_array, flux_array, err_array, header_list, fits_params = self.fits_reader(file_address, **kwargs)
        pixel_mask = pixel_mask if pixel_mask is not None else fits_params['pixel_mask']

        # Mask requested entries
        if pixel_mask is not None:
            pixel_mask = np.atleast_1d(pixel_mask)
            mask_bool_array = np.zeros(flux_array.shape).astype(bool)

            # String array: TODO Lime2.0 make this a method and add it to the spectrum and flux entries and add inf
            if pixel_mask.dtype.kind in ['U', 'S']:
                for entry in pixel_mask:
                    if entry == 'negative':
                        idcs = flux_array < 0
                    elif entry == 'nan':
                        idcs = np.isnan(flux_array) if err_array is None else np.isnan(flux_array) | np.isnan(err_array)
                    elif entry == 'zero':
                        idcs = (flux_array == entry) if err_array is None else (flux_array == entry) | (err_array == entry)
                    else:
                        raise LiMe_Error(f'Pixel entry "{entry}" is not recognized. Only boolean masks for the masked '
                                         f'data or these strings are supported: "nan", "negative", "zero"')
                    mask_bool_array[idcs] = True

            # Boolean mask
            else:
                assert flux_array.shape == pixel_mask.shape, LiMe_Error(f'- Input pixel mask shape {pixel_mask.shape}'
                                                                        f'is different from data array shape {flux_array.shape}')
                mask_bool_array = pixel_mask

        else:
            mask_bool_array = None

        # Construct attributes for LiMe object
        fits_args = {'input_wave': wave_array, 'input_flux': flux_array, 'input_err': err_array}
        fits_args.update(fits_params)

        # Add mask entry
        if mask_bool_array is not None:
            fits_args['pixel_mask'] = mask_bool_array

        return fits_args

    def parse_data_from_url(self, id_label, pixel_mask=None, **kwargs):

        # Read the fits data
        wave_array, flux_array, err_array, header_list, fits_params = self.fits_reader(id_label,  **kwargs)

        # Mask requested entries
        if pixel_mask is not None:
            pixel_mask = np.atleast_1d(pixel_mask)
            mask_array = np.zeros(flux_array.shape).astype(bool)
            for entry in pixel_mask:
                if entry == 'negative':
                    idcs = flux_array < 0
                else:
                    idcs = (flux_array == entry)
                mask_array[idcs] = True
        else:
            mask_array = None

        # Construct attributes for LiMe object
        fits_args = {'input_wave': wave_array, 'input_flux': flux_array, 'input_err': err_array, 'pixel_mask': mask_array,
                     **fits_params}

        return fits_args

    def default_file_parser(self, log_df, id_spec, root_address, **kwargs):

        # Get address of observation
        file_spec = root_address/id_spec[log_df.index.names.index('file')]

        # Get observation data
        fits_args = self.fits_reader(file_spec)

        return fits_args

    @staticmethod
    def text(file_address, **kwargs):

        """
        Load a spectrum from a plain text file.

        This function reads a spectral table from a text file, unpacks the wavelength and flux
        columns, and optionally extracts uncertainty and pixel mask data if present.

        Commented lines (for example #units_flux: FLAM) at the end of the document are parsed as arguments for the
        lime.Spectrum.Spectrum function.

        Parameters
        ----------
        file_address : str or pathlib.Path
            Path to the text file containing the spectrum.
        **kwargs
            Additional keyword arguments passed directly to
            [`numpy.loadtxt`](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html),
            such as ``delimiter``, ``comments``, or ``skiprows``.

        Returns
        -------
        wave_array : ndarray of shape (n,)
            Wavelength array.
        flux_array : ndarray of shape (n,)
            Flux values.
        err_array : ndarray of shape (n,), optional
            Flux uncertainties if available (third column). ``None`` if not provided.
        header_list : None
            Placeholder for compatibility with other readers; always ``None`` for text input.
        params_dict : dict
            Dictionary containing spectrum metadata such as:
              - ``redshift`` : float, optional
              - ``norm_flux`` : float, optional
              - ``id_label`` : str, optional
              - ``pixel_mask`` : ndarray, optional

            These parameters are extracted from the comment section or inferred from the
            file content.

        Notes
        -----
        - If the text file follows the format produced by
          :meth:`lime.Spectrum.retrieve.spectrum`, the function automatically recovers
          the wavelength and flux units, flux normalization, and redshift values stored
          in the file header.
        - The function supports files with up to four columns:
            1. Wavelength
            2. Flux
            3. Uncertainty (optional)
            4. Pixel mask (optional)

        Examples
        --------
        Load a simple two-column spectrum file:

        >>> wave, flux, err, hdr, params = lime.OpenFits.text("spectrum.txt")

        Load a spectrum with custom delimiters using numpy.loadtxt options:

        >>> wave, flux, err, hdr, params = lime.OpenFits.text("spectrum.dat", delimiter=",", skiprows=2)
        """

        # Read text file dividing the columns into the spectrum axis and the comments as its parameters
        data_arr, params_dict = load_txt(file_address, **kwargs)

        # Unpack the columns into the spectrum axes
        wave_array, flux_array = data_arr[:, 0], data_arr[:, 1]

        # Check if pixel error and masks are included
        err_array = data_arr[:, 2] if data_arr.shape[1] > 2 else None
        mask_array = data_arr[:, 3] if data_arr.shape[1] > 3 else None

        # Convert strings to expected format
        params_dict['redshift'] = float(params_dict['redshift']) if 'redshift' in params_dict else None
        params_dict['norm_flux'] = float(params_dict['norm_flux']) if 'norm_flux' in params_dict else None
        params_dict['id_label'] = params_dict['id_label'] if 'id_label' in params_dict else None
        params_dict['pixel_mask'] = mask_array

        return wave_array, flux_array, err_array, None, params_dict

    @staticmethod
    def nirspec(fits_address, data_ext_list=1, hdr_ext_list=(0, 1), **kwargs):

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
        pixel_mask = np.isnan(flux_array) | np.isnan(err_array)

        # Spectrum properties
        params_dict = SPECTRUM_FITS_PARAMS['nirspec']
        params_dict['pixel_mask'] = pixel_mask

        return wave_array, flux_array, err_array, header_list, params_dict

    @staticmethod
    def nirspec_grizli(fits_address, data_ext_list=1, hdr_ext_list=(0, 1), **kwargs):

        """

        This method returns the spectrum array data and headers from a GRIZLI (Brammer (2023a) and Valentino et al.
        (2023)) reduction of a NIRSPEC observation.

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
        wave_array, flux_array, err_array = data_list[0]['wave'], data_list[0]['flux'], data_list[0]['err']
        pixel_mask = np.isnan(flux_array) | np.isnan(err_array)

        # Spectrum properties
        params_dict = SPECTRUM_FITS_PARAMS['nirspec_grizli']
        params_dict['pixel_mask'] = pixel_mask

        return wave_array, flux_array, err_array, header_list, params_dict

    @staticmethod
    def isis(fits_address, data_ext_list=0, hdr_ext_list=0, **kwargs):

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
    def osiris(fits_address, data_ext_list=0, hdr_ext_list=0, **kwargs):

        """
        Load spectral data and metadata from an OSIRIS FITS observation.

        The function extracts the wavelength, flux, and (if available) uncertainty arrays from
        the specified FITS file. It also returns the selected header(s) and a dictionary of
        LiMe Spectrum parameters describing the observation units, normalization, and WCS
        configuration.

        Parameters
        ----------
        fits_address : str or pathlib.Path
            Path to the OSIRIS observation FITS file.
        data_ext_list : int, str, or list of {int, str}, optional
            Extension(s) containing the spectral data. Each element may be an integer index
            or an extension name. Default is ``0``.
        hdr_ext_list : int, str, or list of {int, str}, optional
            Extension(s) containing the FITS headers corresponding to the data extensions.
            Each element may be an integer index or an extension name. Default is ``0``.
        **kwargs
            Additional keyword arguments passed to :func:`load_fits`.

        Returns
        -------
        wave_array : ndarray of shape (n,)
            Wavelength array reconstructed from FITS header keywords ``CRVAL1``, ``CD1_1``,
            and ``NAXIS1``.
        flux_array : ndarray of shape (n,)
            Flux values from the FITS data extension.
        err_array : ndarray of shape (n,), optional
            Flux uncertainty array, if available. Returned as ``None`` if no uncertainty
            extension is found.
        header_list : list of astropy.io.fits.Header
            Headers corresponding to the selected data extensions.
        params_dict : dict
            Observation parameters required to initialize a LiMe Spectrum, including
            wavelength/flux units, normalization, and WCS information.

        Notes
        -----
        - The function assumes a linear wavelength solution defined by FITS header
          keywords ``CRVAL1``, ``CD1_1``, and ``NAXIS1``.
        - For two-dimensional FITS data arrays (``shape == (2, n)``), the first row is
          interpreted as flux and the second as uncertainty.
        - For one-dimensional arrays, only the flux is returned and ``err_array`` is ``None``.

        Examples
        --------
        Load an OSIRIS FITS file and retrieve its spectral arrays:

        >>> wave, flux, err, hdrs, params = osiris("osiris_obs.fits")

        Load a specific data and header extension by name:

        >>> wave, flux, err, hdrs, params = osiris("osiris_obs.fits",
        ...                                        data_ext_list="SCI",
        ...                                        hdr_ext_list="SCI_HDR")
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
    def cos(fits_address, data_ext_list=(1), hdr_ext_list=(0), **kwargs):

        """

        This method returns the spectrum array data and headers from the COS instrument at Hubble.

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

        # Check dimensions of array
        data_list, header_list = load_fits(fits_address, data_ext_list, hdr_ext_list, url_check=False)
        if data_list[0]['WAVELENGTH'].squeeze().ndim == 1:
            wave_arr = data_list[0]['WAVELENGTH'].squeeze()
            flux_arr = data_list[0]['FLUX'].squeeze()
            err_arr = data_list[0]['ERROR'].squeeze()

        else:
            # Get common middle index for joining the spectra
            # wave_matrix = data_list[0]['WAVELENGTH'][::-1]
            idcs_common = np.nonzero(data_list[0]['WAVELENGTH'][1, :] > data_list[0]['WAVELENGTH'][0, 0])[0]
            center_idx = idcs_common.shape[0] // 2

            # Create empty containers
            wave_arr = np.empty(data_list[0]['WAVELENGTH'].size - center_idx, data_list[0]['WAVELENGTH'].dtype)  # TODO check for additional extension to join the spectra
            flux_arr = np.empty(data_list[0]['FLUX'].size - center_idx, data_list[0]['FLUX'].dtype)  # dtype=data_list[0]['FLUX'].dtype)
            err_arr = np.empty(data_list[0]['ERROR'].size - center_idx, data_list[0]['ERROR'].dtype)  # dtype=data_list[0]['ERROR'].dtype)

            # Fill with the array data
            arr_size = data_list[0]['WAVELENGTH'].shape[1]
            for key_arr, cont_arr in zip(['WAVELENGTH', 'FLUX', 'ERROR'], [wave_arr, flux_arr, err_arr]):
                cont_arr[0:arr_size - center_idx] = data_list[0][key_arr][1][0:arr_size - center_idx]
                cont_arr[arr_size - center_idx:] = data_list[0][key_arr][0]
                # print(key_arr, np.any(np.isnan(cont_arr)))

        # Spectrum properties
        params_dict = SPECTRUM_FITS_PARAMS['cos']

        return wave_arr, flux_arr, err_arr, header_list, params_dict

    @staticmethod
    def sdss(fits_address, data_ext_list=(1, 2), hdr_ext_list=(0), **kwargs):

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
        ivar_array = data_list[0]['ivar']

        # Recover the redshift
        try:
            redshift = data_list[1]['Z'][0]
        except:
            redshift = None
            _logger.warning(f'The SDSS redshift could not be read. ZWARNING = {data_list[1]["ZWARNING"]}')

        # Convert ivar = 0 to nan
        ivar_array[ivar_array == 0] = np.nan

        # Get standard deviation cube
        err_array = np.sqrt(1 / ivar_array)

        # Spectrum properties
        params_dict = SPECTRUM_FITS_PARAMS['sdss']
        params_dict['redshift'] = redshift

        return wave_array, flux_array, err_array, header_list, params_dict

    @staticmethod
    def manga(fits_address, data_ext_list=('WAVE', 'FLUX', 'IVAR'), hdr_ext_list=('FLUX'), **kwargs):

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
        ivar_cube = data_list[2]

        # Convert ivar = 0 to nan
        ivar_cube[ivar_cube == 0] = np.nan

        # Get standard deviation cube
        err_cube = np.sqrt(1 / ivar_cube)

        # WCS from hearder
        wcs = WCS(header_list[0])

        # Fits properties
        fits_params = {**CUBE_FITS_PARAMS['manga'], 'wcs': wcs}

        return wave_array, flux_cube, err_cube, header_list, fits_params

    @staticmethod
    def muse(fits_address, data_ext_list=(1, 2), hdr_ext_list=1, **kwargs):

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
    def kcwi(fits_address, data_ext_list=(0, 1, 2), hdr_ext_list=(1,2), **kwargs):

        """

        This method returns the spectrum array data and headers from a kcwi observation.

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
        err_cube = np.sqrt(data_list[1])
        mask_cube = (data_list[2] == 1) | np.isnan(flux_cube)
        # mask_cube = np.isnan(flux_cube)

        wcs = WCS(header_list[0])

        # Fits properties
        fits_params = {**CUBE_FITS_PARAMS['kcwi'], 'pixel_mask': mask_cube, 'wcs': wcs}

        return wave_array, flux_cube, err_cube, header_list, fits_params

    @staticmethod
    def megara(fits_address, data_ext_list=0, hdr_ext_list=(0, 1), **kwargs):

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

    @staticmethod
    def miri(fits_address, data_ext_list=(1,2), hdr_ext_list=(1), **kwargs):

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
        err_cube = data_list[1]
        pixel_mask_cube = None

        wcs = WCS(header_list[0])

        # Fits properties
        fits_params = {**CUBE_FITS_PARAMS['miri'], 'pixel_mask': pixel_mask_cube, 'wcs': wcs}

        return wave_array, flux_cube, err_cube, header_list, fits_params

    @staticmethod
    def desi(target_id, root_url='https://data.desi.lbl.gov/public/edr/spectro/redux', **kwargs):

        # Get the reference catalogue file
        release = kwargs.get('release')
        program = kwargs.get('program')
        catalogue = kwargs.get('catalogue')
        ref_fits = kwargs.get('ref_fits')

        # Check the user specified the
        for (param, param_value) in zip(['release', 'program', 'catalogue'], [release, program, catalogue]):
            if param_value is None:
                raise LiMe_Error(f'To create Spectrum from DESI observation you need to specify the "{param}" argument')

        # Get the file or url location
        if ref_fits is not None:
            if Path(ref_fits).is_file():
                conf_fits = {'name': ref_fits, 'use_fsspec': False}
            else:
                raise LiMe_Error(f'File {ref_fits} not found')

        else:
            # Select the catalogue type
            if catalogue == 'healpix':
                ref_fits = f'zall-pix-{release}.fits'
            else:
                ref_fits = f'zall-tilecumulative-{release}.fits'
            ref_fits_url = f'{root_url}/{release}/zcatalog/{ref_fits}'
            conf_fits = {'name': ref_fits_url, 'use_fsspec': True}

        # Open the reference file with the redshifts
        with fits.open(**conf_fits) as hdul:

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

        # Check the objects found
        if len(url_list) == 0:
            raise LiMe_Error(f'No observations for Object ID {target_id} were found for the input {program} (program), '
                             f'{catalogue} (catalogue), {release} (release)')
        elif len(url_list) > 1:
            url = url_list[0]
            redshift = redshift[0]
            _logger.warning(f' Multiple observations for Object ID {target_id} found for the input {program} (program),'
                             f' {catalogue} (catalogue), {release} (release)\nUsing the first observation.')
        else:
            url = url_list[0]
            redshift = redshift[0]

        # Read the url data
        spectra_dict = {}

        with fits.open(url, use_fsspec=True) as hdulist:

            file_idtargets = hdulist["FIBERMAP"].data['TARGETID']
            obj_idrows = np.where(np.isin(file_idtargets, target_id))[0]

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
        params_dict['redshift'] = redshift

        return wave, flux, err_flux, None, params_dict

# class OpenFitsSurvey:
#
#     @staticmethod
#     def url(fits_address, data_ext_list, hdr_ext_list=None):
#
#         return
#
#     @staticmethod
#     def desi(fits_address, data_ext_list, hdr_ext_list=None):
#
#         # Reshape into an array if necessary
#         data_ext_list, hdr_ext_list = np.atleast_1d(data_ext_list), np.atleast_1d(hdr_ext_list)
#
#         # Read the url data
#         spectra_dict = {}
#
#         with fits.open(fits_address, use_fsspec=True) as hdulist:
#
#             file_idtargets = hdulist["FIBERMAP"].data['TARGETID']
#             obj_idrows = np.where(np.isin(file_idtargets, data_ext_list))[0]
#
#             for i, id_obj in enumerate(obj_idrows):
#                 spectra_dict = {}
#                 for band in ('B', 'R', 'Z'):
#                     spectra_dict[band] = {'wave': hdulist[f'{band}_WAVELENGTH'].section[:],
#                                           'flux': hdulist[f'{band}_FLUX'].section[obj_idrows[i], :],
#                                           'ivar': hdulist[f'{band}_IVAR'].section[obj_idrows[i], :]}
#
#         # Re-construct spectrum arrays
#         wave, flux, err_flux = desi_bands_reconstruction(spectra_dict)
#
#         # Spectrum properties
#         params_dict = SPECTRUM_FITS_PARAMS['desi'].copy()
#
#         return wave, flux, err_flux, None, params_dict



