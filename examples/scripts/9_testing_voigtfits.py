"""
Test of Spectrum.fit.ionic_absorptions() on a mock normalized COS-like spectrum (G130M binned by 3 pixels) with
host (z = 0.0025) and Milky Way (z = 0) absorbers, a damped Lya and a wavelength-dependent non-Gaussian LSF table.

The mock data are made independently of the LiMe model ("nature"): a 15x finer grid, the analytic LSF instead of the
interpolated table and the integration over each pixel. The fitting configuration follows the VoigtFit-style "kinem"
entries split between a default and an object section to test the nested (deep) configuration update.
"""
import tomllib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.special import wofz
import lime

C_KMS = 299792.458
C_CGS, M_E, E_CGS = 2.99792458e10, 9.1094e-28, 4.8032e-10

Z_SPEC = 0.0025
DISP = 0.00997 * 3   # data pixel (A): COS G130M native pixels binned by 3
NSUB = 3             # the LSF table is sampled at the native pixels (VoigtFit nsub)
LSF_HALF = 75        # kernel half-width in native pixels
SNR = 30.0
LSF_FILE = 'mock_cos_lsf.txt'
rng = np.random.default_rng(42)

# True components: the metals share z and b with C II in each origin (as the configuration ties)
v_host, v_mw = 15.0, -6.0
z_host, z_mw = Z_SPEC + v_host / C_KMS * (1 + Z_SPEC), v_mw / C_KMS
TRUE = {('HI', 'none'): (Z_SPEC, 10.0, 20.30), ('HI', 'MW'): (0.0, 10.0, 20.00),
        ('CII', 'none'): (z_host, 18.0, 14.60), ('CII', 'MW'): (z_mw, 10.0, 14.90),
        ('CIIa', 'none'): (z_host, 18.0, 13.30), ('CIIa', 'MW'): (z_mw, 10.0, 13.40),
        ('SiII', 'none'): (z_host, 18.0, 14.20), ('SiII', 'MW'): (z_mw, 10.0, 14.60),
        ('SII', 'none'): (z_host, 18.0, 14.90), ('SII', 'MW'): (z_mw, 10.0, 15.20),
        ('SIII', 'none'): (z_host, 18.0, 14.00), ('SIII', 'MW'): (z_mw, 10.0, 13.60),
        ('FeII', 'none'): (z_host, 18.0, 14.10), ('FeII', 'MW'): (z_mw, 10.0, 14.50)}

# Transitions in the mock (atomic data from the LiMe database = VoigtFit linelist.dat): ion, lambda_0, f, Gamma
TRANSITIONS = [('FeII', 1142.36565, 0.00401, 5.012e8), ('FeII', 1143.22573, 0.0192, 5.623e8),
               ('FeII', 1144.9392, 0.112, 6.026e8), ('SIII', 1190.203, 0.022, 6.166e7),
               ('SiII', 1190.4158, 0.284, 4.266e9), ('SiII', 1193.2897, 0.594, 4.266e9),
               ('HI', 1215.67, 0.416, 6.265e8), ('SII', 1250.578, 0.0052, 5.129e7),
               ('SII', 1253.805, 0.0103, 4.266e7), ('SII', 1259.518, 0.016, 3.388e7),
               ('SiII', 1260.4221, 1.45, 3.631e9), ('CII', 1334.5323, 0.129, 3.981e8),
               ('CIIa', 1335.6627, 0.0127, 3.981e8), ('CIIa', 1335.7077, 0.116, 3.89e8)]

CFG_TOML = """
[default_line_fitting]
Fe2_1142A_b = 'Fe2_1142A+Fe2_1143A+Fe2_1145A+Fe2_1142A_o-MW+Fe2_1143A_o-MW+Fe2_1145A_o-MW'
C2_1335A_b = 'C2_1335A+C2_1336A+C2_1335.7A+C2_1335A_o-MW+C2_1336A_o-MW+C2_1335.7A_o-MW'
S2_1251A_b = 'S2_1251A'
S2_1254A_b = 'S2_1254A+S2_1254A_o-MW'
S2_1260A_b = 'S2_1260A+Si2_1260A+S2_1260A_o-MW+Si2_1260A_o-MW'
S3_1190A_b = 'S3_1190A+Si2_1190A+Si2_1193A+S3_1190A_o-MW+Si2_1190A_o-MW+Si2_1193A_o-MW'
grouped_lines = ['Fe2_1142A_b', 'C2_1335A_b', 'S2_1260A_b', 'S3_1190A_b', 'S2_1251A_b', 'S2_1254A_b']

kinem.none.HI.logN = 21
kinem.none.HI.b = 10
kinem.none.HI.var_b = false
kinem.none.HI.var_z = false

kinem.MW.HI.logN = 20.476
kinem.MW.HI.b = 10
kinem.MW.HI.var_b = false
kinem.MW.HI.var_N = false
kinem.MW.HI.var_z = false

kinem.none.SII.logN = 14.5
kinem.none.SiII.logN = 13.5

kinem.none.k-0_tie = "C2_1335A"
kinem.MW.k-0_tie = "C2_1335A_o-MW"

[obj_line_fitting]
kinem.MW.HI.logN = 20.0
kinem.MW.SiII.logN = 14.0
"""

LINES = ['Fe2_1142A', 'Fe2_1143A', 'Fe2_1145A', 'S3_1190A', 'Si2_1190A', 'Si2_1193A', 'H1_1216A', 'S2_1251A',
         'S2_1254A', 'S2_1260A', 'Si2_1260A', 'C2_1335A', 'C2_1336A', 'C2_1335.7A']


def optical_depth(wl, l0, f, gam, z, b, logN):
    """Exact Voigt optical depth (independent of the LiMe implementation)."""
    b_cgs = b * 1e5
    x = (C_CGS / b_cgs) * (1.0 - l0 * (1 + z) / wl)
    a = l0 * 1e-8 * gam / (4 * np.pi * b_cgs)
    K = np.sqrt(np.pi) * E_CGS ** 2 * f * l0 * 1e-8 / (M_E * C_CGS * b_cgs)
    return K * 10 ** logN * wofz(x + 1j * a).real


def lsf_shape(dnat, wl):
    """Mock COS-like LSF (Moffat, beta = 2.5): dnat offset in native pixels, FWHM 6.5 -> 7.5 px over 1150-1450 A."""
    fwhm = 6.5 + (wl - 1150.0) / 300.0
    alpha = fwhm / 1.1305
    return (1 + (dnat / alpha) ** 2) ** -2.5 / alpha


def write_lsf_table(fname, wl_nodes):
    """HST/COS format: first row = wavelengths, each column = kernel at the native pixels."""
    dnat = np.arange(-LSF_HALF, LSF_HALF + 1)
    np.savetxt(fname, np.vstack([wl_nodes, lsf_shape(dnat[:, None], wl_nodes[None, :])]), fmt='%.6e')


def make_mock(x, nsub_true=15, chunk=4.0):
    """'Nature': fine grid, analytic LSF at each chunk centre, pixel integration and noise."""
    step = DISP / nsub_true
    wl_f = (x[:, None] + ((np.arange(nsub_true) + 0.5) / nsub_true - 0.5) * DISP).ravel()
    tau = np.zeros_like(wl_f)
    for (ion, origin), (z, b, logN) in TRUE.items():
        for t_ion, l0, f, gam in TRANSITIONS:
            if t_ion == ion:
                tau += optical_depth(wl_f, l0, f, gam, z, b, logN)
    I_f = np.exp(-tau)

    # Convolve in chunks with the analytic kernel (the LSF changes by < 0.1% within a chunk)
    n_half = int(LSF_HALF * (DISP / NSUB) / step)
    F_f = np.empty_like(I_f)
    edges = np.arange(wl_f[0], wl_f[-1] + chunk, chunk)
    for lo, hi in zip(edges[:-1], edges[1:]):
        idcs = (wl_f >= lo) & (wl_f < hi)
        kernel = lsf_shape(np.arange(-n_half, n_half + 1) * step / (DISP / NSUB), 0.5 * (lo + hi))
        conv = fftconvolve(np.pad(I_f, n_half, mode='edge'), kernel / kernel.sum(), mode='valid')
        F_f[idcs] = conv[idcs]
    F = F_f.reshape(-1, nsub_true).mean(axis=1)
    err = np.full(x.size, 1.0 / SNR)
    return F + rng.normal(0.0, err), err


if __name__ == '__main__':

    # Mock observation and LSF table
    x = np.arange(1132.0, 1345.0, DISP)
    write_lsf_table(LSF_FILE, np.linspace(1100.0, 1450.0, 8))
    flux, err = make_mock(x)

    # LiMe spectrum with the LSF (nsub = 3 kernel samples per pixel)
    spec = lime.Spectrum(x, flux, err, redshift=Z_SPEC, norm_flux=1,
                         lsf=LSF_FILE, lsf_nsub=NSUB)
    print(spec)

    # Bands with the host and Milky Way origins, wider bands for the damped Lya
    cfg = lime.io.parse_lime_cfg(tomllib.loads(CFG_TOML))
    bands = spec.retrieve.lines_frame(fit_cfg=cfg, default_cfg_prefix='default', obj_cfg_prefix='obj',
                                      map_origin=['none', 'MW'], line_list=LINES,
                                      map_band_vsigma={'H1_1216A': 900, 'H1_1216A_o-MW': 900})
    print(bands[[c for c in ['wavelength', 'w3', 'w4', 'origin', 'z_line'] if c in bands]].to_string())

    # Fit the ionic absorptions
    spec.fit.ionic_absorptions(bands, cfg, default_cfg_prefix='default', obj_cfg_prefix='obj')
    spec.fit.ionic_report()

    # Compare with the true values
    print(f'\n{"component":16s} {"v true":>7s} {"v fit":>14s} {"b true":>7s} {"b fit":>13s} {"logN true":>9s} {"logN fit":>15s}')
    pulls = []
    for (ion, origin), (z, b, logN) in TRUE.items():
        comp = spec.fit.ionic.components[(ion, origin, 0)]
        p = spec.fit.ionic.output.params
        z_f, b_f, N_f = (p[comp.names[k]] for k in ('z', 'b', 'logN'))
        v_t, v_f = (z - Z_SPEC) / (1 + Z_SPEC) * C_KMS, (z_f.value - Z_SPEC) / (1 + Z_SPEC) * C_KMS
        v_e = np.nan if z_f.stderr is None else z_f.stderr / (1 + Z_SPEC) * C_KMS
        b_e = np.nan if b_f.stderr is None else b_f.stderr
        N_e = np.nan if N_f.stderr is None else N_f.stderr
        print(f'{ion + " " + origin:16s} {v_t:7.1f} {v_f:7.1f} ± {v_e:4.1f} {b:7.1f} {b_f.value:6.1f} ± {b_e:4.1f} '
              f'{logN:9.2f} {N_f.value:7.3f} ± {N_e:5.3f}')
        if N_f.vary and np.isfinite(N_e):
            pulls.append((N_f.value - logN) / N_e)
    print(f'logN pulls: mean = {np.mean(pulls):+.2f}, std = {np.std(pulls):.2f} (n = {len(pulls)}), '
          f'reduced chi2 = {spec.fit.ionic.output.redchi:.3f}')

    # Measurements frame
    cols = ['particle', 'z_line', 'z_line_err', 'v_r', 'v_r_err', 'b', 'b_err', 'logN', 'logN_err', 'observations']
    print(spec.frame[cols].to_string(float_format=lambda v: f'{v:.5g}'))

    # Quick check plot of the regions with the best fit model
    regions = spec.fit.ionic.regions
    fig, axes = plt.subplots(1, len(regions), figsize=(3 * len(regions), 3.2), sharey=True, layout='constrained')
    for ax, reg in zip(np.atleast_1d(axes), regions):
        ax.step(reg.wave, reg.flux, where='mid', color='0.4', lw=0.8)
        ax.plot(reg.wave, reg.model, color='tab:orange', lw=1.2)
        ax.set_xlabel('Wavelength (Å)')
    np.atleast_1d(axes)[0].set_ylabel('Normalized flux')
    fig.savefig('lime_ionic_absorptions_test.png', dpi=120)
    plt.show()