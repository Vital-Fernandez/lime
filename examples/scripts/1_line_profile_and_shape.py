from pathlib import Path
import lime


# Specify the data location the observations
data_folder = Path('../doc_notebooks/0_resources/')
sloan_SHOC579 = data_folder/'spectra'/'sdss_dr18_0358-51818-0504.fits'
bands_df_file = data_folder/'bands'/'SHOC579_bands.txt'

# Create the observation object
spec = lime.Spectrum.from_file(sloan_SHOC579, instrument='sdss', redshift=0.0475)

# Fit a line from the default label list

spec.fit.bands('H1_4861A_p-l')
spec.plot.bands()

spec.clear_data()
spec.fit.bands('H1_4861A', profile='l', cont_source='adjacent')
spec.plot.bands()
