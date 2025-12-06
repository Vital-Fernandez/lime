import lime
import numpy as np
from scipy import stats
import spectres
import numpy as np

def bin_data(wave, flux, err, opt_elem):

    '''

    This script bins data
    by 1 resolution element for COS (6 pixes).

    # get errors on binned flux #
    1) loop through the bin numbers,
    2) find the index in the binnumber array,
    3) square the error (for the corresponding index)
    4) sum the squared errors
    5) get the square root of the summed errors
    6) append to binn_err array

    '''

    # This is in Angstroms per resel
    if opt_elem == 'G130M':
        bin_width =  0.05982
    elif opt_elem == 'G160M':
        bin_width = 0.07338

    # Bin data
    bin_edges = np.arange(wave[0], wave[-1], bin_width)
    binned_flux, edges, binnumber = stats.binned_statistic(wave, flux, statistic='mean', bins=bin_edges)
    binned_wave = bin_edges[:-1]+(bin_width/2.)

    # get unique bin numbers #
    uni_bin = np.unique(binnumber)
    err_binned = []

    for binnum in uni_bin[:-1]:
        index_bin = np.where(binnumber==binnum)
        errors_bin = err[index_bin]
        err_bin = np.sqrt(np.sum(errors_bin**2))/(len(errors_bin))
        err_binned.append(err_bin)

    return binned_wave, binned_flux, err_binned

file_path = '/home/vital/Astrodata/STScI/LyC_leakers_COS/Direct_downloads/LF9G01010/hst_17515_cos_mrk-209_g130m_lf9g01_cspec.fits'
spec = lime.Spectrum.from_file(file_path, instrument='cos', redshift=0.000932, norm_flux=1e-17)

spec.infer.components()
spec.plot.spectrum(show_err=True, in_fig=None)

wave_svea, flux_svea, err_svea = bin_data(spec.wave, spec.flux, spec.err_flux, opt_elem='G130M')
flux_carnall, err_carnall = spectres.spectres(wave_svea, spec.wave, spec.flux, spec.err_flux)

rebin_wave = np.arange(spec.wave.data[0], spec.wave.data[-1], 0.05982)
wave_vit, flux_vit, err_vit = spec.retrieve.rebinned(disp_intvl=rebin_wave)
wave_vit, flux_vit, err_vit = spec.retrieve.rebinned(pixel_width=0.05982)
wave_vit, flux_vit, err_vit = spec.retrieve.rebinned(pixel_number=6, constant_pixel_width=True)

spec.plot.ax.step(wave_svea, flux_svea, label='Svea rebin', where='mid')
spec.plot.ax.fill_between(x=wave_svea, y1=(flux_svea - err_svea), y2=(flux_svea + err_svea), step='mid', alpha=0.5, color='blue', ec=None)

# spec.plot.ax.step(wave_svea, flux_carnall, label='Carnall rebin', where='mid')
# spec.plot.ax.fill_between(x=wave_svea, y1=(flux_carnall - err_carnall), y2=(flux_carnall + err_carnall), step='mid', alpha=0.5, color='orange', ec=None)

spec.plot.ax.step(wave_vit, flux_vit, label='Vital rebin', where='mid', linestyle='--', color='red')
spec.plot.ax.fill_between(x=wave_vit, y1=(flux_vit - err_vit), y2=(flux_vit + err_vit), step='mid', alpha=0.2, color='red', ec=None)

spec.plot.ax.legend()
spec.plot.show()
