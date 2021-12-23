import lime
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt, rcParams, colors, cm, gridspec

# Input the data files
obsFitsFile = './sample_data/manga-8626-12704-LOGCUBE.fits'
linesMaskFile = './sample_data/gp121903_BR_mask.txt'
z_obj = 0.047232

# Open the cube fits file
with fits.open(obsFitsFile) as hdul:
    hdr = hdul['FLUX'].header
    wave = hdul['WAVE'].data
    flux = hdul['FLUX'].data

# Object mask
maskDF = lime.load_lines_log(linesMaskFile)


lineList = ['H1_6563A_b', 'O3_4363A', 'S2_6716A_b']

for lineLabel in lineList:

    if lineLabel in maskDF.index:
        lineWaves = maskDF.loc[lineLabel, 'w1':'w6'] * (1 + z_obj)
    else:
        exit(f'- ERROR: {lineLabel} not found in input mask log')

    # Use Halpha for the background
    wave_regions = maskDF.loc[lineLabel, 'w1':'w6']
    wave_regions = np.array(wave_regions, ndmin=2)
    idcs_wave = np.searchsorted(wave, wave_regions)

    # Adjacent continua flux
    idcsCont = (((wave[idcs_wave[:, 0]] <= wave[:, None]) &
                (wave[:, None] <= wave[idcs_wave[:, 1]])) |
                ((wave[idcs_wave[:, 4]] <= wave[:, None]) & (
                  wave[:, None] <= wave[idcs_wave[:, 5]]))).squeeze()

    contFluxImage = flux[idcsCont]
    lineFluxImage = flux[idcs_wave[0][0]:idcs_wave[0][1], :, :]

    # Line region integrated flux
    interval_line = wave[idcs_wave[0][0]:idcs_wave[0][1]]
    pixelWidth = np.diff(interval_line).mean()
    # lineIntgrFlux = np.nansum(flux[idcs_wave[0]:idcs_wave[1], :, :], axis=0) * pixelWidth
    SN_image = np.nansum(lineFluxImage - np.nanmean(contFluxImage), axis=0) / np.nanstd(contFluxImage, axis=0)

    SN_image = np.ma.masked_array(SN_image, mask=~np.isfinite(SN_image))

    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_subplot()

    norm_color_bg = colors.SymLogNorm(linthresh=0,
                                      vmin=0)
    ax.imshow(np.nansum(lineFluxImage, axis=0), cmap=cm.gray)

    plt.show()