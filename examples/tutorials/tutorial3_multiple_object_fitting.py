from pathlib import Path

import numpy as np

import lime
from lime.transitions import au
lime.theme.set_style('dark')

# Data folder
data_folder = Path('../sample_data')
ref_bands = '/home/vital/PycharmProjects/lime/src/lime/resources/lines_database_v2.0.0.txt'

# Configuration file
cfg = lime.load_cfg(data_folder/'long_slit.toml')

# Spectra list
object_dict = {'sdss':'SHOC579', 'nirspec':'ceers1027', 'osiris':'gp121903',  'isis':'Izw18', }

# File list
files_dict = {'sdss':'SHOC579_SDSS_dr18.fits',
              'osiris': 'gp121903_osiris.fits',
              'nirspec':'hlsp_ceers_jwst_nirspec_nirspec10-001027_comb-mgrat_v0.7_x1d-masked.fits',
              'isis': 'IZW18_isis.fits'}

# Loop through the files and measure the lines
for i, items in enumerate(object_dict.items()):

    inst, obj = items
    file_path = data_folder/'spectra'/files_dict[inst]
    redshift = cfg[inst][obj]['z']

    # Create the observation object
    fname = f'./{obj}_{inst}_file.txt'
    spec = lime.Spectrum.from_file(file_path, inst, redshift=redshift)

    if spec.units_wave != 'AA':
        spec.unit_conversion('AA', 'FLAM')

    # Bands the results
    bands = spec.retrieve.line_bands(band_vsigma=100, fit_conf=cfg, obj_conf_prefix=f'{obj}_{inst}')
    # spec.plot.spectrum(bands=bands, rest_frame=True)
    # lime.save_frame(data_folder/'bands'/f'{obj}_{inst}_bands.txt', bands)

    # spec.plot.spectrum(bands=bands)
    # spec.check.bands(data_folder/'bands'/f'{obj}_{inst}_bands.txt', exclude_continua=False, band_vsigma=100, fit_conf=cfg,
    #                  obj_conf_prefix=f'{obj}_{inst}')

    spec.fit.frame(bands, cfg, obj_conf_prefix=f'{obj}_{inst}')
    spec.plot.spectrum(rest_frame=True)
    # spec.save_frame(data_folder / f'{obj}_{inst}_lines.txt')

    # spec.retrieve.spectrum(output_address=fname)
    #
    # spec2 = lime.Spectrum.from_file(fname, 'text')
    # spec2.plot.spectrum(rest_frame=True)


    # out_array = np.loadtxt(fname)
    #
    #
    # metadata = {}
    # with open(fname, "r") as f:
    #     for i, line in enumerate(reversed(f.readlines())):  # Read in reverse
    #         print(i)
    #         line = line.strip()
    #         if not line.startswith("#") or (line.startswith("# LiMe")):
    #             break  # EXIT the loop entirely when a non-comment line is found
    #
    #         # Extract key-value pairs
    #         key, value = line[1:].split(":", 1)  # Split at the first ':'
    #         metadata[key.strip()] = value.strip()
    #
    #
    # metadata['redshift'] = float(metadata['redshift']) if 'redshift' in metadata else None
    # metadata['units_wave'] = au.Unit(metadata['units_wave']) if 'units_wave' in metadata else au.Unit('AA')
    # metadata['units_flux'] = au.Unit(metadata['units_flux']) if 'units_flux' in metadata else au.Unit('FLAM')
    # metadata['norm_flux'] = float(metadata['norm_flux']) if 'norm_flux' in metadata else None
    # metadata['id_label'] = metadata['id_label'] if 'norm_flux' in metadata else None
    #
    # # Example usage:
    # print(metadata)
    # spec2 = lime.Spectrum(out_array[:, 0], out_array[:, 1], **metadata)
    # spec2.plot.spectrum(rest_frame=True)

    # spec2 = lime.Spectrum(wave=, flux=)
        # if spec.units_wave != 'AA':
        #     spec.unit_conversion('AA', 'FLAM')
        #
        # # Bands the results
        # bands = spec.retrieve.line_bands(band_vsigma=100)
        # # spec.check.bands(data_folder/'bands'/f'{obj}_{inst}_bands.txt', ref_bands=bands, exclude_continua=False)
        #
        # spec.fit.frame(data_folder/'bands'/f'{obj}_{inst}_bands.txt', cfg, id_conf_prefix=f'{obj}_{inst}',
        #                line_list=['H1_4861A_b'])
        # spec.fit.report()
        # spec.plot.bands()
        #
        # # spec.plot.spectrum(rest_frame=False)


