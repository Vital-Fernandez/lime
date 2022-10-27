import numpy as np
import lime

smacs_spectra = lime.Sample()

for order in [1, 2, 3, 4, 5]:

    name = f'spec_{order}'
    wave = np.array([10, 12, 13, 14, 15, 16])
    flux = np.ones(wave.size) * order
    units_wave = 'um' if order == 3 else 'A'

    smacs_spectra.add_object(name, obs_type='spectrum', input_wave=wave, input_flux=flux, units_wave=units_wave)

for label, spec in smacs_spectra.items():
    print(label, spec.flux)

smacs_spectra.plot.spectra()
