"""
Forward-model fit of a mock C IV 1548,1550 absorption doublet observed with a
wavelength-dependent, non-Gaussian (COS-like) line-spread function, following
the same steps as VoigtFit with an LSF file:

  1. tau(lambda) on a sub-pixel grid, summed over transitions -> I = exp(-tau)
  2. I convolved with one LSF kernel per sub-pixel (read from an HST-style table)
  3. back to data pixels, multiplied by a Chebyshev continuum: M = C x (LSF * I)
  4. residual = (data - M) / err, minimised with lmfit Levenberg-Marquardt

Then, to reproduce the profiles later WITHOUT the LSF file:

  5. line_kernels()       : for each line take ONE kernel = the LSF row at the sub-pixel
                            nearest the observed line centre, and compress it to a mixture of a
                            few Gaussians (2 Gaussians = 3 free numbers instead of 151 samples)
  6. save_record()        : write fit results + those few numbers to a small JSON file
  7. profile_from_record(): rebuild the kernel from the mixture and the observed profile from the
                            JSON alone (no LSF file, no fit objects), e.g. on a finer grid for plots

The mock data are made independently of the fit model ("nature"): a finer grid,
the analytic LSF instead of the interpolated table, the LSF acting on C*I, and
integration over each pixel.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.special import wofz
import lmfit

C_KMS = 299792.458
C_CGS, M_E, E_CGS = 2.99792458e10, 9.1094e-28, 4.8032e-10

# C IV doublet, atomic data from VoigtFit's static/linelist.dat: (name, l0 [A], f, Gamma [s^-1])
LINES = [("CIV 1548", 1548.1870, 0.191, 2.63e8),
         ("CIV 1550", 1550.7720, 0.0953, 2.63e8)]

TRUE = dict(z=0.003, b=8.0, logN=13.8, c0=1.0, c1=0.05)  # b in km/s, logN in cm^-2
DISP = 0.01223       # A per data pixel (COS G160M-like, ~2.4 km/s)
NSUB = 3             # sub-pixels per data pixel for the LSF table and model (VoigtFit's nsub)
LSF_HALF = 25        # kernel half-width in data pixels
SNR = 25.0
LSF_FILE = "mock_lsf.txt"
RECORD_FILE = "civ_fit_record.json"
rng = np.random.default_rng(42)


# ---------------------------------------------------------------- physics ---
def optical_depth(wl, z, b, logN, lines=LINES):
    """Step 1: tau = K N H(a, x) summed over transitions (as voigt.Voigt in VoigtFit).
    H(a, x) = Re[w(x + ia)] is the exact Voigt-Hjerting function."""
    b_cgs = b * 1e5
    tau = np.zeros_like(wl)
    for _, l0, f, gam in lines:
        x = (C_CGS / b_cgs) * (1.0 - l0 * (1 + z) / wl)
        a = l0 * 1e-8 * gam / (4 * np.pi * b_cgs)
        K = np.sqrt(np.pi) * E_CGS**2 * f * l0 * 1e-8 / (M_E * C_CGS * b_cgs)
        tau += K * 10**logN * wofz(x + 1j * a).real
    return tau


def continuum(wl, c0, c1, domain):
    """First-order Chebyshev continuum (VoigtFit's evaluate_continuum)."""
    return np.polynomial.Chebyshev([c0, c1], domain=domain)(wl)


# -------------------------------------------------------------------- LSF ---
LSF_KIND = "moffat"   # "moffat": power-law wings (not a Gaussian mixture: a fair test of the compression)
                      # "gauss2": Gaussian core + Gaussian wings (a 2-Gaussian summary is then exact)


def lsf_shape(dpix, wl):
    """Mock non-Gaussian LSF. dpix: offset in data pixels; wl: wavelength in A.
    The FWHM of the core grows 6.5 -> 7.5 px over 1545-1565 A."""
    fwhm = 6.5 + (wl - 1545.0) / 20.0
    if LSF_KIND == "moffat":
        alpha = fwhm / 1.1305                       # Moffat beta = 2.5: FWHM = 2 alpha sqrt(2^(1/beta) - 1)
        return (1 + (dpix / alpha)**2) ** -2.5 / alpha
    s1 = fwhm / 2.35482
    s2 = 3.0 * s1
    return 0.85 * np.exp(-0.5 * (dpix / s1)**2) / s1 + 0.15 * np.exp(-0.5 * (dpix / s2)**2) / s2


def write_lsf_table(fname, wl_nodes):
    """HST/COS format: first row = wavelengths, each column = kernel on the sub-pixel grid."""
    dpix = np.arange(-LSF_HALF * NSUB, LSF_HALF * NSUB + 1) / NSUB          # odd length
    np.savetxt(fname, np.vstack([wl_nodes, lsf_shape(dpix[:, None], wl_nodes[None, :])]), fmt="%.6e")


def load_lsf(fname, wl_grid):
    """As VoigtFit regions.load_lsf: normalise each column, then bilinear interpolation
    in (pixel, wavelength) -> one kernel per grid point, shape (N, M)."""
    tab = np.loadtxt(fname)
    lsf_wl, lsf = tab[0], tab[1:] / tab[1:].sum(axis=0)
    pix = np.arange(lsf.shape[0])
    return RectBivariateSpline(pix, lsf_wl, lsf, kx=1, ky=1)(pix, wl_grid).T


def convolve_lsf(P, kernel):
    """Step 2: wavelength-dependent convolution, kernel[i] is the LSF at output pixel i.
    The kernel is flipped so asymmetric LSFs act as a true convolution
    (VoigtFit's voigt.convolve applies it unflipped; identical for symmetric LSFs).
    Edge padding instead of ones, so it also works on un-normalised flux."""
    n = kernel.shape[1] // 2
    windows = sliding_window_view(np.pad(P, n, mode="edge"), kernel.shape[1])
    return np.einsum("ij,ij->i", windows, kernel[:, ::-1])


def subpixel_grid(x, nsub):
    """nsub sub-pixel centres inside each data pixel, flattened."""
    return (x[:, None] + ((np.arange(nsub) + 0.5) / nsub - 0.5) * DISP).ravel()


def to_pixels(P_sub, nsub):
    """Average sub-pixels -> data pixels (pixel integration; VoigtFit uses np.interp here)."""
    return P_sub.reshape(-1, nsub).mean(axis=1)


# -------------------------------------------------------------- mock data ---
def make_mock(x, nsub_true=15):
    """'Nature': fine grid, analytic LSF, LSF acting on C*I, pixel integration, noise."""
    wl_f = subpixel_grid(x, nsub_true)
    C_f = continuum(wl_f, TRUE["c0"], TRUE["c1"], (x.min(), x.max()))
    I_f = np.exp(-optical_depth(wl_f, TRUE["z"], TRUE["b"], TRUE["logN"]))
    dpix = np.arange(-LSF_HALF * nsub_true, LSF_HALF * nsub_true + 1) / nsub_true
    kern = lsf_shape(dpix[None, :], wl_f[:, None])
    kern /= kern.sum(axis=1, keepdims=True)
    F = to_pixels(convolve_lsf(C_f * I_f, kern), nsub_true)
    err = np.sqrt(F * TRUE["c0"]) / SNR                                     # photon-noise-like
    return F + rng.normal(0.0, err), err


# -------------------------------------------------------------------- fit ---
def build_model(x, kernel):
    # The kernel has one row per model sub-pixel, so nsub follows from its shape (no global needed)
    nsub = kernel.shape[0] // x.size
    assert kernel.shape[0] == nsub * x.size, \
        f"kernel has {kernel.shape[0]} rows, expected nsub x {x.size} data pixels"
    wl_sub = subpixel_grid(x, nsub)
    domain = (x.min(), x.max())

    def model(p, sub=False):
        I = np.exp(-optical_depth(wl_sub, p["z"], p["b"], p["logN"]))      # step 1
        F_sub = convolve_lsf(I, kernel)                                     # step 2
        if sub:
            return wl_sub, F_sub
        return continuum(x, p["c0"], p["c1"], domain) * to_pixels(F_sub, nsub)  # step 3

    return model


def residual(pars, model, data, err):
    return (data - model(pars.valuesdict())) / err                          # step 4


def fit(x, flux, err, kernel):
    model = build_model(x, kernel)
    pars = lmfit.Parameters()
    pars.add("z",       value=TRUE["z"] + 15.0 / C_KMS * (1 + TRUE["z"]))        # start 15 km/s off
    pars.add("b",       value=15.0, min=1.0, max=100.0)
    pars.add("logN",    value=13.3, min=11.0, max=17.0)
    pars.add("c0",      value=np.median(flux))
    pars.add("c1",      value=0.0)

    # VoigtFit uses factor=1 and ftol=0.01; here ftol stays at the tight MINPACK default.
    result = lmfit.Minimizer(residual, pars, fcn_args=(model, flux, err),
                             nan_policy="omit").minimize(method="leastsq", factor=1.0)
    return result, model


# ------------------------------------------- reproduce profiles without the LSF file ---
def kernel_fwhm(offs, k):
    """FWHM of a single-peaked kernel k sampled at offsets offs (same units as offs)."""
    h = k.max() / 2
    above = np.where(k >= h)[0]
    a, b = above[0], above[-1]
    left = np.interp(h, [k[a - 1], k[a]], [offs[a - 1], offs[a]])
    right = np.interp(h, [k[b + 1], k[b]], [offs[b + 1], offs[b]])
    return right - left


def mixture(v, w, s):
    """Normalised kernel made of Gaussians: weights w, widths s [km/s], offsets v [km/s]."""
    g = np.exp(-0.5 * (v[:, None] / s[None, :])**2) / s[None, :]
    k = g @ w
    return k / k.sum()


def fit_mixture(v, k, n):
    """Least-squares fit of an n-Gaussian mixture (weights sum to 1) to the kernel k(v)."""
    sig0 = kernel_fwhm(v, k) / 2.35482
    s0 = sig0 * (np.geomspace(0.8, 3.0, n) if n > 1 else 1.0)
    unpack = lambda q: (np.exp(q[:n]) / np.exp(q[:n]).sum(), np.exp(q[n:]))
    res = lambda q: (mixture(v, *unpack(q)) - k) / k.max()
    q = least_squares(res, np.r_[np.zeros(n), np.log(s0)]).x
    return unpack(q)


def line_kernels(kernel, wl_sub, z, n_gauss=2):
    """Step 5. For each line: take the row of the (N, M) kernel array at the sub-pixel nearest the
    observed line centre l0*(1+z) (sum = 1), then compress it to an n_gauss Gaussian mixture.
    Stored per line: weights, widths [km/s], the half-width of the original table [km/s] (so the
    rebuilt kernel is cut where the table was) and the worst kernel error relative to the peak.
    Atomic data are stored too, so the record is self-contained.
    Returns (records, tables); tables hold the full (v, k) only for plotting/validation."""
    records, tables = [], []
    for name, l0, f, gam in LINES:
        l_obs = l0 * (1 + z)
        i0 = np.argmin(np.abs(wl_sub - l_obs))
        k = kernel[i0] / kernel[i0].sum()
        step_kms = (wl_sub[1] - wl_sub[0]) / l_obs * C_KMS  # sampling step of the table [km/s]
        v = (np.arange(k.size) - k.size // 2) * step_kms
        w, s = fit_mixture(v, k, n_gauss)
        records.append(dict(name=name, l0=l0, f=f, gam=gam, l_obs=l_obs,
                            lsf_weights=w.tolist(), lsf_sigma_kms=s.tolist(),
                            lsf_halfwidth_kms=float(v[-1]),
                            kernel_fwhm_kms=kernel_fwhm(v, k),
                            kernel_max_err=float(np.abs(mixture(v, w, s) - k).max() / k.max())))
        tables.append((v, k))
    return records, tables


def save_record(fname, params, x, kernels):
    """Step 6. Everything needed to redraw the profile: best-fit parameters (+ errors),
    continuum domain, and the per-line kernels. No LSF file, no lmfit objects."""
    rec = dict(params={n: dict(value=p.value, stderr=p.stderr) for n, p in params.items()},
               continuum_domain=[float(x.min()), float(x.max())],
               lines=kernels)
    with open(fname, "w") as fh:
        json.dump(rec, fh, indent=1)


def profile_from_record(rec, i, wl):
    """Step 7. Observed normalised profile of line i, rebuilt from the saved record only.
    wl must be a uniform wavelength grid (any sampling): the kernel is rebuilt from the stored
    Gaussian mixture on the step of wl (cut at the stored half-width, renormalised), then
    exp(-tau) of all lines is convolved with it."""
    p = {n: v["value"] for n, v in rec["params"].items()}
    atomic = [(L["name"], L["l0"], L["f"], L["gam"]) for L in rec["lines"]]
    I = np.exp(-optical_depth(wl, p["z"], p["b"], p["logN"], atomic))
    L = rec["lines"][i]
    dv = (wl[1] - wl[0]) / L["l_obs"] * C_KMS                          # grid step in km/s
    n = int(L["lsf_halfwidth_kms"] // dv)
    kn = mixture(np.arange(-n, n + 1) * dv, np.array(L["lsf_weights"]), np.array(L["lsf_sigma_kms"]))
    return fftconvolve(np.pad(I, n, mode="edge"), kn, mode="valid")   # constant kernel -> FFT


# ------------------------------------------------------------------- plot ---
DATA, FIT_OBS, FIT_INT, TRUTH, GRID = "#52514e", "#2a78d6", "#eb6834", "#0b0b0b", "#d6d5d0"
LINE2, GAUSS = "#6250d6", "#1baf7a"


def plot(x, flux, err, result, model, rec, tables):
    p = result.params.valuesdict()
    cont = continuum(x, p["c0"], p["c1"], (x.min(), x.max()))
    resid = (flux - model(p)) / err

    fig, axd = plt.subplot_mosaic([["r0", "r1", "lsf"], ["p0", "p1", "lsf"]],
                                  height_ratios=[1, 3], figsize=(13, 5.2), layout="constrained")
    for k, (name, l0, _, _) in enumerate(LINES):
        l_ref = l0 * (1 + TRUE["z"])
        vel = lambda wl: (wl - l_ref) / l_ref * C_KMS
        ax, rax = axd[f"p{k}"], axd[f"r{k}"]
        rax.sharex(ax)
        wl_line = np.arange(l_ref - 1.5, l_ref + 1.5, 0.002)               # fine plotting grid
        I_true = np.exp(-optical_depth(wl_line, TRUE["z"], TRUE["b"], TRUE["logN"]))
        I_fit = np.exp(-optical_depth(wl_line, p["z"], p["b"], p["logN"]))
        F_rec = profile_from_record(rec, k, wl_line)                       # from the record only
        ax.step(vel(x), flux / cont, where="mid", color=DATA, lw=0.9, label="Mock data")
        ax.plot(vel(wl_line), I_true, color=TRUTH, ls="--", lw=1.0, label="True intrinsic exp(-τ)")
        ax.plot(vel(wl_line), I_fit, color=FIT_INT, lw=1.5, label="Best-fit intrinsic exp(-τ)")
        ax.plot(vel(wl_line), F_rec, color=FIT_OBS, lw=2.0, label="Rebuilt from saved record (no LSF file)")
        for y in (0, 1):
            ax.axhline(y, color=GRID, lw=0.8, zorder=0)
        ax.set(xlim=(-200, 200), ylim=(-0.08, 1.25), xlabel="Velocity (km/s)")
        ax.text(0.03, 0.05, name, transform=ax.transAxes, fontsize=11)
        rax.step(vel(x), resid, where="mid", color=DATA, lw=0.9)
        for y in (-3, 3):
            rax.axhline(y, color=GRID, ls=":", lw=1.0)
        rax.axhline(0, color=GRID, lw=0.8)
        rax.set(ylim=(-5, 5))
        rax.tick_params(labelbottom=False)
    axd["p0"].set_ylabel("Normalised flux")
    axd["r0"].set_ylabel("(data−model)/σ")
    handles, labels = axd["p0"].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower left", ncol=4, fontsize=8, frameon=False)

    # CIV 1548: full table kernel vs the few numbers kept in the record vs a single Gaussian
    ax = axd["lsf"]
    L, (v, k) = rec["lines"][0], tables[0]
    w, s = np.array(L["lsf_weights"]), np.array(L["lsf_sigma_kms"])
    km = mixture(v, w, s)
    ax.semilogy(v, k / k.max(), color=GRID, lw=5, label=f"Table kernel ({k.size} numbers)")
    ax.semilogy(v, km / km.max(), color=TRUTH, lw=1.3,
                label=f"{len(w)} Gaussians ({2 * len(w) - 1} numbers): max err {L['kernel_max_err']:.0e} of peak")
    comp = np.exp(-0.5 * (v[:, None] / s[None, :])**2) / s[None, :] * w[None, :]   # each Gaussian, weighted
    lines = ax.semilogy(v, comp / comp.sum(axis=1).max(), color=FIT_OBS, lw=1.0, ls=":")
    lines[0].set_label("its Gaussian components")                  # one legend entry for both
    sg = L["kernel_fwhm_kms"] / 2.35482
    ax.semilogy(v, np.exp(-0.5 * (v / sg)**2), color=GAUSS, ls="--", lw=1.5, label="1 Gaussian, same FWHM")
    ax.set(ylim=(1e-4, 80), xlim=(v.min(), v.max()), xlabel="Offset (km/s)", ylabel="Kernel (peak = 1)",
           title=f"{L['name']} kernel: {k.size} samples -> {2 * len(w) - 1} numbers")
    ax.title.set_fontsize(10)
    ax.legend(loc="upper center", fontsize=7.5, frameon=False)
    fig.savefig("voigt_lsf_forward_fit.png", dpi=150)
    plt.show()


# ------------------------------------------------------------------- main ---
if __name__ == "__main__":
    l_lo = LINES[0][1] * (1 + TRUE["z"]) - 2.5
    l_hi = LINES[1][1] * (1 + TRUE["z"]) + 2.5
    x = np.arange(l_lo, l_hi, DISP)

    write_lsf_table(LSF_FILE, np.linspace(1545.0, 1565.0, 5))
    flux, err = make_mock(x)
    kernel = load_lsf(LSF_FILE, subpixel_grid(x, NSUB))     # fixed during the fit
    result, model = fit(x, flux, err, kernel)

    z_ref = TRUE["z"]
    print(f"{'param':6s} {'true':>10s} {'fit':>10s} {'error':>9s}  (z shown as velocity, km/s)")
    for name, par in result.params.items():
        val, sig, tru = par.value, par.stderr or np.nan, TRUE[name]
        if name == "z":
            val, sig, tru = (val - z_ref) / (1 + z_ref) * C_KMS, sig / (1 + z_ref) * C_KMS, 0.0
        print(f"{name:6s} {tru:10.4f} {val:10.4f} {sig:9.4f}")
    print(f"reduced chi2 = {result.redchi:.3f}  (N_pix = {x.size}, nfev = {result.nfev})")

    # --- save the fit + a 2-Gaussian summary of ONE kernel per line, then work from the JSON alone ---
    wl_sub, _ = model(result.params.valuesdict(), sub=True)
    records, tables = line_kernels(kernel, wl_sub, result.params["z"].value, n_gauss=2)
    save_record(RECORD_FILE, result.params, x, records)
    with open(RECORD_FILE) as fh:
        rec = json.load(fh)                                  # from here on: no LSF file, no lmfit objects

    n_rec = sum(len(L["lsf_weights"]) + len(L["lsf_sigma_kms"]) for L in rec["lines"])
    print(f"\nLSF table ({LSF_KIND}): {kernel.size} floats  ->  record: {n_rec} floats for the LSF of both lines")
    print(f"{'line':9s} {'FWHM [km/s]':>12s} {'weights':>14s} {'sigmas [km/s]':>16s} {'max kernel err':>15s}")
    for L in rec["lines"]:
        print(f"{L['name']:9s} {L['kernel_fwhm_kms']:12.2f} "
              f"{np.array2string(np.array(L['lsf_weights']), precision=3):>14s} "
              f"{np.array2string(np.array(L['lsf_sigma_kms']), precision=2):>16s} {L['kernel_max_err']:15.1e}")

    # --- check: profile rebuilt from the record vs the per-pixel-kernel fit model --------
    _, F_fit = model(result.params.valuesdict(), sub=True)
    for i, L in enumerate(rec["lines"]):
        v = (wl_sub - L["l_obs"]) / L["l_obs"] * C_KMS
        w = np.abs(v) < 250
        diff = np.abs(profile_from_record(rec, i, wl_sub[w]) - F_fit[w])[np.abs(v[w]) < 200].max()
        print(f"{L['name']}: max |rebuilt - fit model| = {diff:.1e}  (noise per pixel ~ {1 / SNR:.0e})")

    plot(x, flux, err, result, model, rec, tables)