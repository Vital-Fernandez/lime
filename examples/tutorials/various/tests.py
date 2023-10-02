from pathlib import Path
from astropy.io import fits


def join_fits_logs(log_path_list, output_address, keep_individual_files=False):

    # Create new HDU for the combined file with a new PrimaryHDU
    hdulist = fits.HDUList([fits.PrimaryHDU()])

    # Iterate through the file paths, open each FITS file, and append the non-primary HDUs to hdulist
    missing_files = []
    for log_path in log_path_list:
        if log_path.is_file():
            with fits.open(log_path) as hdulist_i:

                # Remove primary
                if isinstance(hdulist_i[0], fits.PrimaryHDU):
                    hdulist_i.pop(0)

                # Combine list
                hdulist += hdulist_i.copy()

        else:
            missing_files.append(log_path)

    # Save to a combined file
    hdulist.writeto(output_address, overwrite=True, output_verify='ignore')
    hdulist.close()

    return


mask_log_list = [Path("..\..\sample_data\SHOC579_log_MASK-MASK_0.fits"),
                 Path("..\..\sample_data\SHOC579_log_MASK-MASK_1.fits"),
                 Path("..\..\sample_data\SHOC579_log_MASK-MASK_2.fits")]

join_fits_logs(mask_log_list, Path("..\..\sample_data\SHOC579_log_COMB.fits"))
