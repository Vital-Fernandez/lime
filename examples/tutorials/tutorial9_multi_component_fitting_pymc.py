import pymc as pm
import arviz as az
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import pytensor.tensor as pt
import pandas as pd
import xarray as xr

# print(f"Running on PyMC3 v{pm.__version__}")
#
# N = 1000
# W = np.array([0.35, 0.4, 0.25])
# MU = np.array([0.0, 2.0, 5.0])
# SIGMA = np.array([0.5, 0.5, 1.0])
#
RANDOM_SEED = 8927
# rng = np.random.default_rng(RANDOM_SEED)

# component = rng.choice(MU.size, size=N, p=W)
# x = rng.normal(MU[component], SIGMA[component], size=N)
#
# # fig, ax = plt.subplots(figsize=(8, 6))
# # ax.hist(x, bins=30, density=False, lw=0)
# # plt.show()
#
# coords = {"cluster": np.arange(len(W)), "obs_id": np.arange(N)}
#
# with pm.Model(coords=coords) as model:
#
#     w = pm.Dirichlet("w", np.ones_like(W))
#
#     mu = pm.Normal("mu", np.zeros_like(W), 1.0, dims="cluster", transform=pm.distributions.transforms.Ordered)
#
#     tau = pm.Gamma("tau", 1.0, 1.0, dims="cluster")
#
#     x_obs = pm.NormalMixture("x_obs", w, mu, tau=tau, observed=x, dims="obs_id")
#

# N = 20
# K = 30
#
# alpha = 2.0
# P0 = sp.stats.norm
#
# # beta = sp.stats.beta.rvs(1, alpha, size=(N, K))
# # w = np.empty_like(beta)
# # w[:, 0] = beta[:, 0]
# # w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)
# #
# # omega = P0.rvs(size=(N, K))
# # x_plot = xr.DataArray(np.linspace(-3, 3, 200), dims=["plot"])
# # sample_cdfs = (w[..., np.newaxis] * np.less.outer(omega, x_plot.values)).sum(axis=1)
# #
# # fig, ax = plt.subplots(figsize=(8, 6))
# #
# # ax.plot(x_plot, sample_cdfs[0], c="gray", alpha=0.75, label="DP sample CDFs")
# # ax.plot(x_plot, sample_cdfs[1:].T, c="gray", alpha=0.75)
# # ax.plot(x_plot, P0.cdf(x_plot), c="k", label="Base CDF")
# #
# # ax.set_title(rf"$\alpha = {alpha}$")
# # ax.legend(loc=2)
# # plt.show()
#
# fig, (l_ax, r_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 6))
#
# K = 50
# alpha = 10.0
#
# beta = sp.stats.beta.rvs(1, alpha, size=(N, K))
# w = np.empty_like(beta)
# w[:, 0] = beta[:, 0]
# w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)
#
# omega = P0.rvs(size=(N, K))
#
# x_plot = xr.DataArray(np.linspace(-3, 3, 200), dims=["plot"])
#
#
# sample_cdfs = (w[..., np.newaxis] * np.less.outer(omega, x_plot.values)).sum(axis=1)
#
# l_ax.plot(x_plot, sample_cdfs[0], c="gray", alpha=0.75, label="DP sample CDFs")
# l_ax.plot(x_plot, sample_cdfs[1:].T, c="gray", alpha=0.75)
# l_ax.plot(x_plot, P0.cdf(x_plot), c="k", label="Base CDF")
#
# l_ax.set_title(rf"$\alpha = {alpha}$")
# l_ax.legend(loc=2)
#
# K = 200
# alpha = 50.0
#
# beta = sp.stats.beta.rvs(1, alpha, size=(N, K))
# w = np.empty_like(beta)
# w[:, 0] = beta[:, 0]
# w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)
# x_plot = xr.DataArray(np.linspace(-3, 3, 200), dims=["plot"])
#
# omega = P0.rvs(size=(N, K))
#
# sample_cdfs = (w[..., np.newaxis] * np.less.outer(omega, x_plot.values)).sum(axis=1)
#
# r_ax.plot(x_plot, sample_cdfs[0], c="gray", alpha=0.75, label="DP sample CDFs")
# r_ax.plot(x_plot, sample_cdfs[1:].T, c="gray", alpha=0.75)
# r_ax.plot(x_plot, P0.cdf(x_plot), c="k", label="Base CDF")
#
# r_ax.set_title(rf"$\alpha = {alpha}$")
# r_ax.legend(loc=2)
# plt.show()
#
# N = 5
# K = 30
#
# alpha = 2
# P0 = sp.stats.norm
# f = lambda x, theta: sp.stats.norm.pdf(x, theta, 0.3)
#
# beta = sp.stats.beta.rvs(1, alpha, size=(N, K))
# w = np.empty_like(beta)
# w[:, 0] = beta[:, 0]
# w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)
#
# theta = P0.rvs(size=(N, K))
#
# dpm_pdf_components = f(x_plot, theta[..., np.newaxis])
# dpm_pdfs = (w[..., np.newaxis] * dpm_pdf_components).sum(axis=1)
#
# fig, ax = plt.subplots(figsize=(8, 6))
#
# ax.plot(x_plot, dpm_pdfs.T, c="gray")
#
# ax.set_yticklabels([])
# plt.show()
#
# fig, ax = plt.subplots(figsize=(8, 6))
#
# ix = 1
#
# ax.plot(x_plot, dpm_pdfs[ix], c="k", label="Density")
# ax.plot(
#     x_plot,
#     (w[..., np.newaxis] * dpm_pdf_components)[ix, 0],
#     "--",
#     c="k",
#     label="Mixture components (weighted)",
# )
# ax.plot(x_plot, (w[..., np.newaxis] * dpm_pdf_components)[ix].T, "--", c="k")
#
# ax.set_yticklabels([])
# ax.legend(loc=1)
# plt.show()


# ---------------------------- PYMC DIRILICH
# old_faithful_df = pd.read_csv(pm.get_data("old_faithful.csv"))
#
# old_faithful_df["std_waiting"] = (
#     old_faithful_df.waiting - old_faithful_df.waiting.mean()
# ) / old_faithful_df.waiting.std()
#
#
# fig, ax = plt.subplots(figsize=(8, 6))
#
# n_bins = 20
# ax.hist(old_faithful_df.std_waiting, bins=n_bins, color="C0", lw=0, alpha=0.5)
#
# ax.set_xlabel("Standardized waiting time between eruptions")
# ax.set_ylabel("Number of eruptions")
# plt.show()
#
# N = old_faithful_df.shape[0]
# K = 30
#
# def stick_breaking(beta):
#     portion_remaining = pt.concatenate([[1], pt.extra_ops.cumprod(1 - beta)[:-1]])
#     return beta * portion_remaining
#
# with pm.Model(coords={"component": np.arange(K), "obs_id": np.arange(N)}) as model:
#     alpha = pm.Gamma("alpha", 1.0, 1.0)
#     beta = pm.Beta("beta", 1.0, alpha, dims="component")
#     w = pm.Deterministic("w", stick_breaking(beta), dims="component")
#
#     tau = pm.Gamma("tau", 1.0, 1.0, dims="component")
#     lambda_ = pm.Gamma("lambda_", 10.0, 1.0, dims="component")
#     mu = pm.Normal("mu", 0, tau=lambda_ * tau, dims="component")
#     obs = pm.NormalMixture(
#         "obs", w, mu, tau=lambda_ * tau, observed=old_faithful_df.std_waiting.values, dims="obs_id"
#     )
#
# with model:
#     trace = pm.sample(
#         tune=2500,
#         init="advi",
#         target_accept=0.975,
#         random_seed=RANDOM_SEED,
#     )
#
# az.plot_trace(trace, var_names=["alpha"])
# plt.show()


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
bands_obj = spec.retrieve.line_bands(fit_conf=cfg, obj_conf_prefix=f'{obj}_{inst}')
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

import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# # Load the provided x, y data
# x_data = np.array([-1.141333047004756054e+03, -1.072450594768959945e+03, -1.003738485537147085e+03,
#                    -9.348347404258528286e+02, -8.660800454430441278e+02, -7.971550074562520649e+02,
#                    -7.283790195979454438e+02, -6.595604459886430959e+02, -5.905928222508550789e+02,
#                    -5.217529557660546971e+02, -4.527427462772709532e+02, -3.838815869169726511e+02,
#                    -3.148500845526909302e+02, -2.459463394413967308e+02, -1.770425943301025882e+02,
#                    -1.079472133393270639e+02, -3.900088247703705235e+01, 3.011579138923638510e+01,
#                    9.908341512702432397e+01, 1.682426747442936232e+02, 2.372528842330774239e+02,
#                    3.062630937218611962e+02, 3.754649390901263928e+02, 4.445177343299059771e+02,
#                    5.137621654491670142e+02, 5.828362535644446325e+02, 6.521019775592035330e+02,
#                    7.212186514254771055e+02, 7.903566181672483708e+02, 8.596649279130031118e+02,
#                    9.288454804057703313e+02, 9.981750830270231063e+02, 1.067376928395288360e+03,
#                    1.136770409643034782e+03, 1.205993547886797842e+03, 1.275237979006058822e+03,
#                    1.344652753129303164e+03, 1.413939769999559985e+03])
#
# y_data = np.array([1.269252861401837240e-02, 1.866478042324198894e-02, 7.995373714930142928e-03,
#                    1.633627605675391781e-02, 4.307642384045989870e-02, 9.979968681993017787e-02,
#                    1.544735576809872502e-01, 1.672059002351845014e-01, 1.388687731886845689e-01,
#                    9.114591908858654667e-02, 7.533039999119972574e-02, 1.064619668621594739e-01,
#                    1.610458663467823071e-01, 2.413047090265221795e-01, 3.685718793026175333e-01,
#                    4.821645310993077360e-01, 5.464141474577505209e-01, 5.508944582576453808e-01,
#                    4.884113278067476194e-01, 3.702127911398576110e-01, 2.580495564388293195e-01,
#                    1.768080381554892333e-01, 1.243829171186386362e-01, 8.723683946318022686e-02,
#                    6.139261728793710127e-02, 5.185393746679262117e-02, 5.077958061808596213e-02,
#                    7.380528797057056556e-02, 1.375286145051325959e-01, 2.268864274233291511e-01,
#                    2.685388232028499411e-01, 2.490230533808084346e-01, 1.728924436996950309e-01,
#                    8.218626100554077274e-02, 3.630517785595716029e-02, 2.293511051693597480e-02,
#                    1.135296943726366026e-02, 4.731005665310150654e-03])

# # Define the PyMC model for three Gaussians
# with pm.Model() as gaussian_mixture_model:
#
#     # Priors for Gaussian parameters (mean, sigma, amplitude)
#     mu1 = pm.Normal("mu1", mu=np.mean(0), sigma=200)
#     mu2 = pm.Normal("mu2", mu=np.mean(0), sigma=200)
#     mu3 = pm.Normal("mu3", mu=np.mean(0), sigma=200)
#
#     sigma1 = pm.HalfNormal("sigma1", sigma=100)
#     sigma2 = pm.HalfNormal("sigma2", sigma=100)
#     sigma3 = pm.HalfNormal("sigma3", sigma=100)
#
#     A1 = pm.HalfNormal("A1", sigma=1)
#     A2 = pm.HalfNormal("A2", sigma=1)
#     A3 = pm.HalfNormal("A3", sigma=1)
#
#     # Gaussian mixture model
#     def gaussian(x, A, mu, sigma):
#         return A * pm.math.exp(-0.5 * ((x - mu) / sigma) ** 2)
#
#     y_model = (
#         gaussian(x_data, A1, mu1, sigma1) +
#         gaussian(x_data, A2, mu2, sigma2) +
#         gaussian(x_data, A3, mu3, sigma3)
#     )
#
#     # Likelihood (observed data)
#     y_obs = pm.Normal("y_obs", mu=y_model, sigma=0.05, observed=y_data)
#
#     # Run inference using NUTS sampler
#     trace = pm.sample(2000, return_inferencedata=True, random_seed=42, cores=2)
# Define the PyMC model for three Gaussians

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

n_comps = 3
with pm.Model() as gaussian_mixture_model:

    # Priors for Gaussian parameters (3D vectors)
    mu = pm.Normal("mu", mu=0, sigma=200, shape=n_comps)
    sigma = pm.HalfNormal("sigma", sigma=100, shape=n_comps)
    A = pm.HalfNormal("A", sigma=1, shape=n_comps)

    # Gaussian mixture model (vectorized computation)
    y_model = pm.math.sum(A[:, None] * pm.math.exp(-0.5 * ((x_in - mu[:, None]) / sigma[:, None]) ** 2), axis=0)

    # Likelihood (observed data)
    y_obs = pm.Normal("y_obs", mu=y_model, sigma=0.05, observed=y_in)

    # Run inference using NUTS sampler
    trace = pm.sample(2000, return_inferencedata=True, random_seed=42, cores=2)

# Plot posterior distributions of the parameters
az.plot_trace(trace, var_names=["mu", "sigma", "A"])
plt.show()

# Generate x values for model fitting
mu_mean = trace.posterior["mu"].mean(dim=("chain", "draw")).values
sigma_mean = trace.posterior["sigma"].mean(dim=("chain", "draw")).values
A_mean = trace.posterior["A"].mean(dim=("chain", "draw")).values
x_fit = np.linspace(min(x_in), max(x_in), 500)
gaussian_components = np.array([gaussian(x_fit, A_mean[i], mu_mean[i], sigma_mean[i]) for i in range(n_comps)])
y_fit_total = np.sum(gaussian_components, axis=0)

# Plot the individual Gaussians and the total fit
plt.figure(figsize=(8, 5))
plt.step(x_in, y_in, label="Observed Data", color="black", where='mid')

for i in range(n_comps):
    plt.plot(x_fit, gaussian_components[i], label=f"Gaussian {i+1}", linestyle="dashed")

plt.plot(x_fit, y_fit_total, label="Total Fit", color="red", lw=2)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Individual Gaussian Components & Blended Fit")
plt.show()