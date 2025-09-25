from pathlib import Path
import lime

# Specify the data location the observations
data_folder = Path('../0_resources')
spec_path = data_folder/'spectra'/'MRK209_cos_x1dsum.fits'

spec = lime.Spectrum.from_file(spec_path, instrument='COS', z=0)
spec.plot.spectrum()


# # Specify the data location the observations
# data_folder = Path('../0_resources')
# sloan_SHOC579 = data_folder/'spectra'/'sdss_dr18_0358-51818-0504.fits'
# bands_df_file = data_folder/'bands'/'SHOC579_bands.txt'
#
# # Create the observation object
# spec = lime.Spectrum.from_file(sloan_SHOC579, instrument='sdss', redshift=0.0475)
# # spec.plot.spectrum()
#
# # Fit a line from the default label list
#
# spec.fit.bands('H1_4861A_p-l')
# spec.plot.bands()
#
# spec.fit.bands('H1_4861A', profile='l')
# spec.plot.bands()
