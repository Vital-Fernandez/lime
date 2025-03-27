from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
import lime


# Plot function
def plot_mixture(gmm, X, show_legend=True, ax=None, limits=(-6, 6), n_axis=1000):
    if ax is None:
        ax = plt.gca()

    # Compute PDF of whole mixture
    x = np.linspace(limits[0], limits[1], 1000)
    logprob = gmm.score_samples(x.reshape(-1, 1))
    pdf = np.exp(logprob)

    # Compute PDF for each component
    responsibilities = gmm.predict_proba(x.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    # Plot data histogram
    ax.hist(X, 30, density=True, histtype='stepfilled', alpha=0.4, label='Data')

    # Plot PDF of whole model
    ax.plot(x, pdf, '-k', label='Mixture PDF')

    # Plot PDF of each component
    ax.plot(x, pdf_individual, '--', label='Component PDF')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$p(x)$')

    if show_legend:
        ax.legend()

    return




# # Show all models for n_components 1 to 9
# _, axes = plt.subplots(3, 3, figsize=np.array([3,3])*3, dpi=100)
# for gmm, ax in zip(models, axes.ravel()):
#     plot_mixture(gmm, X, show_legend=False, ax=ax)
#     ax.set_title(f'k={gmm.n_components}')
#     plt.tight_layout()
# plt.show()
#
#
# # Compute metrics to determine best hyperparameter
# AIC = [m.aic(X) for m in models]
# BIC = [m.bic(X) for m in models]
#
# # Plot these metrics
# plt.plot(k_arr, AIC, label='AIC')
# plt.plot(k_arr, BIC, label='BIC')
# plt.xlabel('Number of Components ($k$)')
# plt.legend()
# plt.show()
#
# # Plot these results
# gmm_best = models[np.argmin(AIC)]
# plot_mixture(gmm_best, X)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import lime
from lime.fitting.lines import c_KMpS

# Data folder
data_folder = Path('../sample_data')
ref_bands = '/home/vital/PycharmProjects/lime/src/lime/resources/lines_database_v2.0.0.txt'

# Configuration file
cfg = lime.load_cfg(data_folder/'long_slit.toml')

# Spectra list
object_dict = {'osiris':'gp121903', 'nirspec':'ceers1027', 'isis':'Izw18', 'sdss':'SHOC579'}

# File list
files_dict = {'osiris': 'gp121903_osiris.fits',
              'nirspec':'hlsp_ceers_jwst_nirspec_nirspec10-001027_comb-mgrat_v0.7_x1d-masked.fits',
              'isis': 'IZW18_isis.fits',
              'sdss':'SHOC579_SDSS_dr18.fits'}


inst, obj = 'sdss', 'SHOC579'
file_path = data_folder/'spectra'/files_dict[inst]
redshift = cfg[inst][obj]['z']


# Create the observation object
spec = lime.Spectrum.from_file(file_path, inst, redshift=redshift)

spec.fit.bands('H1_6563A_b', data_folder/'bands'/f'{obj}_{inst}_bands.txt', cfg, id_conf_prefix=f'{obj}_{inst}')

line = spec.fit.line
idcs = np.searchsorted(spec.wave.data, line.mask[2:4] * (1 + spec.redshift))

ingt_area = line.intg_flux/spec.norm_flux
ref_wave = line.wavelength[0] * (1 + spec.redshift)
ref_vel = 0

x_arr = spec.wave.data[idcs[0]:idcs[-1]]
y_arr = spec.flux.data[idcs[0]:idcs[-1]]
err_arr = spec.err_flux.data[idcs[0]:idcs[-1]]
cont_arr = line.m_cont * x_arr + line.n_cont
vel_arr = c_KMpS * (x_arr - ref_wave)/ref_wave

# Log scale
x_norm = vel_arr
y_norm = np.log10(y_arr)
cont_norm = np.log10(cont_arr)
err_norm = err_arr/y_arr
peak_norm = np.log10(y_arr.max())

# Normalizing conversion
log_intg = np.log10(ingt_area)
x_norm = vel_arr
y_norm = y_norm/log_intg
cont_norm = cont_norm/log_intg
err_norm = err_norm/log_intg
peak_norm = peak_norm/log_intg

x_in = vel_arr
y_in = y_norm - cont_norm
err_in = err_norm
peak_in = peak_norm - cont_norm[np.argmax(y_norm)]

fig, ax = plt.subplots()
ax.step(x_in, y_in, where='mid')
ax.scatter(0, peak_in)
ax.fill_between(x_norm, y_in-err_in, y_in+err_in, color='r', alpha=0.7)
ax.update({'xlabel': 'Velocity (km/s)', 'ylabel':'Normalized log(Flux)'})
plt.show()


# import scipy.integrate as integrate
# from scipy.interpolate import interp1d
#
# # Normalize the curve (convert y into a probability density function)
# y_pdf = y_in / integrate.trapezoid(y_in, x_in)  # Normalize so integral is 1
#
# # Compute the cumulative distribution function (CDF)
# cdf = integrate.cumulative_trapezoid(y_pdf, x_in, initial=0)
#
# # Normalize CDF to range [0,1]
# cdf /= cdf[-1]
#
# # Invert CDF using interpolation
# inverse_cdf = interp1d(cdf, x_in, kind='linear', fill_value="extrapolate")
#
# # Sample m points from uniform [0,1] and map through inverse CDF
# random_samples = np.random.uniform(0, 1, size=100000)
# sampled_x = inverse_cdf(random_samples)
#
# # Plot results
# plt.figure(figsize=(8, 5))
# plt.plot(x_in, y_pdf, label="Normalized Curve (PDF)", color="black")
# plt.hist(sampled_x, bins=x_in.size, density=True, alpha=0.6, color="red", label="Sampled Distribution")
# plt.xlabel("X")
# plt.ylabel("Probability Density")
# plt.legend()
# plt.title("Sampling Using Inverse Transform Method")
# plt.show()


from scipy.interpolate import interp1d

# Define number of points to sample
m = 100000

# Set up rejection sampling parameters
x_min, x_max = np.min(x_in), np.max(x_in)  # X range
y_max = np.max(y_in)  # Maximum Y value for rejection sampling

# Preallocate arrays
samples_x = np.empty(m)
samples_y = np.empty(m)

# Create the interpolator once (instead of inside the loop)
interpolator = interp1d(x_in, y_in, kind='linear', fill_value="extrapolate")

count = 0  # Counter for accepted points
while count < m:
    # Generate batch of random (x, y) values
    x_rand = np.random.uniform(x_min, x_max, size=m - count)
    y_rand = np.random.uniform(0, y_max, size=m - count)

    # Get interpolated y values at random x points
    y_curve = interpolator(x_rand)

    # Select valid points under the curve
    valid_mask = y_rand <= y_curve
    num_valid = np.sum(valid_mask)

    # Fill available slots in preallocated arrays
    if count + num_valid > m:
        num_valid = m - count  # Ensure we do not exceed m

    samples_x[count:count + num_valid] = x_rand[valid_mask][:num_valid]
    samples_y[count:count + num_valid] = y_rand[valid_mask][:num_valid]

    count += num_valid  # Update counter


# Plot the curve and sampled points
plt.figure(figsize=(8, 5))
# plt.plot(x_in, y_in, label="Curve", color="black")
# plt.scatter(samples_x, samples_y, s=5, color="red", alpha=0.5, label="Sampled Points")
plt.hist(samples_x, bins=x_in.size, density=True, alpha=0.6, color="red", label="Sampled Distribution")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Sampling Points Under the Curve (Optimized Rejection Sampling)")
plt.show()

# Reshape the format
sampled_x = samples_x.reshape(-1, 1)

# Fit models with 1-10 components
k_arr = np.arange(10) + 1
models = [GaussianMixture(n_components=k).fit(sampled_x) for k in k_arr]

# Show all models for n_components 1 to 9
a_, axes = plt.subplots(4, 4, figsize=np.array([3, 2])*3, dpi=100)
for gmm, ax in zip(models, axes.ravel()):
    plot_mixture(gmm, sampled_x, show_legend=False, ax=ax, limits=(-2000, 2000))
    ax.set_title(f'k={gmm.n_components}')
    plt.tight_layout()
plt.show()

# Compute metrics to determine best hyperparameter
AIC = [m.aic(sampled_x) for m in models]
BIC = [m.bic(sampled_x) for m in models]

# Plot these metrics
plt.plot(k_arr, AIC, label='AIC')
plt.plot(k_arr, BIC, label='BIC')
plt.xlabel('Number of Components ($k$)')
plt.legend()
plt.show()

gmm_best = models[np.argmin(AIC)]
plot_mixture(gmm_best, sampled_x, limits=(-2000, 2000) )
plt.show()


from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import norm


# Fit Bayesian Gaussian Mixture Model (DPMM)
dpmm = BayesianGaussianMixture(n_components=4,  # Upper bound for components
                               covariance_type="full",
                               weight_concentration_prior_type="dirichlet_process",
                               random_state=42)
dpmm.fit(sampled_x)

# Extract Gaussian parameters
means = dpmm.means_.flatten()  # Gaussian means
std_devs = np.sqrt(dpmm.covariances_.flatten())  # Standard deviations
weights = dpmm.weights_.flatten()  # Mixture weights

# Define x-axis range
x_vals = np.linspace(min(sampled_x) - 1, max(sampled_x) + 1, 1000)

# Compute the total Gaussian Mixture Model (sum of all Gaussians)
gmm_sum = np.zeros_like(x_vals)
plt.figure(figsize=(8, 5))

# Plot individual Gaussian components
for mean, std, weight in zip(means, std_devs, weights):
    if weight > 0.01:  # Ignore very small components
        gaussian_curve = weight * norm.pdf(x_vals, mean, std)
        gmm_sum += gaussian_curve  # Add to total mixture
        plt.plot(x_vals, gaussian_curve, linestyle="dotted", alpha=0.7, label=f"Gaussian (μ={mean:.2f}, σ={std:.2f})")

# Plot histogram of data
plt.hist(sampled_x, bins=30, density=True, alpha=0.6, color="gray", label="Data Histogram")

# Plot summed mixture model
plt.plot(x_vals, gmm_sum, color="red", linewidth=2, label="Summed Mixture Model")

plt.xlabel("X values")
plt.ylabel("Density")
plt.title("Bayesian Gaussian Mixture Model (DPMM) with Summed Profile")
plt.legend()
plt.show()