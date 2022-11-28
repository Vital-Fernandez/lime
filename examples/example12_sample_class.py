import numpy as np
import lime
from pathlib import Path

smacs_spectra = lime.Sample()



# for order in [1, 2, 3, 4, 5]:
#
#     name = f'spec_{order}'
#     wave = np.array([10, 12, 13, 14, 15, 16])
#     flux = np.ones(wave.size) * order
#     units_wave = 'um' if order == 3 else 'A'
#
#     smacs_spectra.add_object(name, obs_type='spectrum', input_wave=wave, input_flux=flux, units_wave=units_wave)
#
# for label, spec in smacs_spectra.items():
#     print(label, spec.flux)
#
# smacs_spectra.plot.spectra()

spectrumA1 = Path(f'D:/AstroData/IZW18_A1/IZW18_A1_Blue_fglobal.fits')
spectrumA2 = Path(f'D:/AstroData/IZW18_A2/IZW18_A2_Blue_fglobal.fits')

for i, spec_address in enumerate([spectrumA1, spectrumA2]):
    wave, flux, hdr = lime.load_fits(spec_address, instrument='ISIS')
    spec = lime.Spectrum(wave, flux, redshift=0.00256539, norm_flux=1e-18)
    spec.fit.band(6716)
    spec.plot.line()
    spec.plot.spectrum()
