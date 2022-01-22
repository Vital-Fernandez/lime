import lime
import numpy as np
from astropy.io import fits

# Input the data files
obsFitsFile = './sample_data/manga-8626-12704-LOGCUBE.fits'
linesMaskFile = './sample_data/gp121903_BR_mask.txt'
z_obj = 0.047232

# Open the cube fits file
with fits.open(obsFitsFile) as hdul:
    hdr = hdul['FLUX'].header
    wave = hdul['WAVE'].data
    flux = hdul['FLUX'].data

# Lines from which generate the spatialmasks
maskDF = lime.load_lines_log(linesMaskFile)
lineList = ['H1_6563A_b', 'O3_4363A', 'S2_6716A_b']

# Using the most intensive percentils to define the masks
contours_param = 'percentil'
percentil_array = np.array([99.99, 99.90, 99.50, 97.50])

# Loop throught he lines
for lineLabel in lineList:

    if lineLabel in maskDF.index:
        lineWaves = maskDF.loc[lineLabel, 'w1':'w6'] * (1 + z_obj)
    else:
        exit(f'- ERROR: {lineLabel} not found in input mask log')

    # Sum the fluxes in the wavelength interval to generate the line flux images
    idcs_wave = np.searchsorted(wave, lineWaves)
    flux_image = np.nansum(flux[idcs_wave[0]:idcs_wave[1], :, :], axis=0)

    # Compute the masks and save them as a fits file and an image
    fits_name = f'./sample_data/manga-8626-12704_{lineLabel}.fits'
    lime.spatial_mask_generator(flux_image, 'percentil', percentil_array, mask_ref=lineLabel, output_address=fits_name,
                                 show_plot=True)

    # Open the fits file and check the masked spaxels
    with fits.open(fits_name) as hdul:
        ext = f'{lineLabel}_mask_1'
        mask_frame = hdul[ext].data.astype('bool')
        idcs_spaxels = np.argwhere(mask_frame)
        for idx_voxel in np.arange(idcs_spaxels.shape[0]):
            idx_j, idx_i = idcs_spaxels[idx_voxel]
            print(idx_j, idx_i)

        print(f'True spaxels in {ext}: {np.sum(mask_frame.data)}')

