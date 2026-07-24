"""Report-style plots: mock (a SuperMock) vs bundled observations.

Cheap panels use the flat photometry / luminosities (no SEDs):
    redshift_distribution, gsmf, smhm, luminosity_function, number_counts,
    optical_colors, wise_w1w2
SED panels need SuperMock(..., seds=True):
    example_seds, gama_stacks, wise_colors  (W2-W3 needs W3/W4 from SEDs)

Every function takes a SuperMock `sm`, draws on an optional `ax`, and returns
the Axes.  Cosmology and the literature reference curves are local so the
package has no pipeline dependency.
"""
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from . import observations as obs

COS = FlatLambdaCDM(H0=67.77, Om0=0.307115)      # SMDPL, as in the pipeline
AB_ZERO_JY = 3631.0
WISE_VEGA_TO_AB = {'W1': 2.699, 'W2': 3.339, 'W3': 5.174, 'W4': 6.620}
# Baldry+12 GSMF (double Schechter) and Blanton+03 ^0.1r LF
_BALDRY = (10**10.66, 3.96e-3, -0.35, 0.79e-3, -1.47)
_BLANTON = dict(Mstar=-20.44, phistar=1.49e-2, alpha=-1.05, h=0.6777)


def _ax(ax, **kw):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(**kw)
    return ax


# --------------------------------------------------------------- 01 redshift
def redshift_distribution(sm, ax=None, bins=80, zmax=3.0):
    ax = _ax(ax)
    z = sm.redshift
    ax.hist(z[z < zmax], bins=bins, color='#2f6fb2', alpha=.8)
    ax.set(xlabel='redshift', ylabel='galaxies', yscale='log',
           title=f'redshift distribution (patches {sm.patches})')
    return ax


# ------------------------------------------------------------------ 02 GSMF
def _baldry(mid):
    Ms, p1, a1, p2, a2 = _BALDRY
    mm = 10**mid
    return np.log(10)*np.exp(-mm/Ms)*(p1*(mm/Ms)**(a1+1) + p2*(mm/Ms)**(a2+1))


def gsmf(sm, ax=None, zlo=0.05, zhi=0.30, epoch_mass=True):
    """Stellar mass function vs Baldry+12 (GAMA).  Uses observed-epoch masses
    (stellar_mass_zobs) by default."""
    ax = _ax(ax)
    logM = sm.logM_zobs if epoch_mass else sm.logM
    z = sm.redshift
    mb = np.arange(8.0, 12.6, 0.1)
    mid = 0.5*(mb[1:]+mb[:-1])
    Vc = (COS.comoving_volume(zhi)-COS.comoving_volume(zlo)).value * sm.area_deg2/41253.
    sel = (z >= zlo) & (z < zhi)
    h = np.histogram(logM[sel], bins=mb)[0]/Vc/0.1
    ax.plot(mid, np.where(h > 0, h, np.nan), color='#1d4e79', lw=1.9,
            label=f'mock {zlo}<z<{zhi}')
    ax.plot(mid, _baldry(mid), 'k--', lw=1.3, label='Baldry+12 (GAMA)')
    ax.set(yscale='log', ylim=(1e-6, 3e-1), xlim=(8.5, 12.2),
           xlabel=r'$\log_{10} M_*\,[M_\odot]$',
           ylabel=r'$\phi\,[\mathrm{Mpc^{-3}\,dex^{-1}}]$', title='GSMF')
    ax.legend(frameon=False, fontsize=9)
    return ax


# ------------------------------------------------------------------ 03 SMHM
def smhm(sm, ax=None, zmax=0.3):
    """Stellar-to-halo mass, centrals (core's own peak mass)."""
    ax = _ax(ax)
    cen = (sm.field('central') == 1) & (sm.redshift < zmax)
    lh = np.log10(np.clip(sm.field('peak_mass')[cen] / _BLANTON['h'], 1, None))
    lm = sm.logM_zobs[cen]
    xb = np.linspace(11, 14.5, 30)
    idx = np.digitize(lh, xb)
    med = [np.median(lm[idx == i]) if np.count_nonzero(idx == i) > 20 else np.nan
           for i in range(1, xb.size)]
    ax.hexbin(lh, lm, gridsize=50, bins='log', cmap='Blues', mincnt=1)
    ax.plot(0.5*(xb[1:]+xb[:-1]), med, color='#e07b39', lw=2, label='median')
    ax.set(xlabel=r'$\log_{10} M_{\rm peak}\,[M_\odot]$',
           ylabel=r'$\log_{10} M_*\,[M_\odot]$',
           title=f'SMHM: centrals, z<{zmax}')
    ax.legend(frameon=False, fontsize=9)
    return ax


# --------------------------------------------------------- 14 luminosity func
def _blanton_lf(Mr):
    b = _BLANTON
    Mstar = b['Mstar'] + 5*np.log10(b['h'])
    x = 10**(0.4*(Mstar - Mr))
    return 0.4*np.log(10)*b['phistar']*b['h']**3 * x**(b['alpha']+1)*np.exp(-x)


def luminosity_function(sm, ax=None, zmax=0.3):
    """Rest-frame r-band LF vs Blanton+03 (needs luminosities file)."""
    ax = _ax(ax)
    Mr = sm.abs_mag('SDSS', 'r')
    z = sm.luminosity('redshift')
    sel = np.isfinite(Mr) & (z < zmax)
    mb = np.arange(-24, -16, 0.4)
    mid = 0.5*(mb[1:]+mb[:-1])
    Vc = COS.comoving_volume(zmax).value * sm.area_deg2/41253.
    h = np.histogram(Mr[sel], bins=mb)[0]/Vc/0.4
    ax.plot(mid, np.where(h > 0, h, np.nan), color='#1d4e79', lw=1.9,
            marker='o', ms=3, label=f'mock z<{zmax}')
    ax.plot(mid, _blanton_lf(mid), 'k--', lw=1.3, label='Blanton+03 $^{0.1}r$')
    ax.set(yscale='log', ylim=(1e-6, 1e-1), xlabel=r'$M_r$ (rest AB)',
           ylabel=r'$\phi\,[\mathrm{Mpc^{-3}\,mag^{-1}}]$',
           title='luminosity function')
    ax.legend(frameon=False, fontsize=9)
    return ax


# --------------------------------------------------------- number counts N(m)
def number_counts(sm, ax=None, survey='LSST', band='i',
                  bins=np.arange(14, 25, 0.5)):
    ax = _ax(ax)
    m = sm.mag(survey, band)
    m = m[np.isfinite(m)]
    mid = 0.5*(bins[1:]+bins[:-1])
    h = np.histogram(m, bins=bins)[0]/sm.area_deg2/np.diff(bins)
    ax.semilogy(mid, h, color='#1d4e79', lw=1.9, marker='o', ms=3)
    ax.set(xlabel=f'{survey} {band} (AB)',
           ylabel=r'N [deg$^{-2}$ mag$^{-1}$]',
           title=f'differential number counts ({survey} {band})')
    return ax


# ---------------------------------------------------- 07 optical colour-colour
def optical_colors(sm, ax=None, survey='SDSS', cx='g-i', cy='u-g', zmax=None):
    """Mock LSST ugriz colours vs SDSS/DEEP2 spectroscopic colours.

    NOTE: mock colours are LSST ugriz (a close proxy for SDSS filters -- exact
    SDSS-filter colours require projecting SEDs; see the notebook).  The mock
    is restricted to the survey's z-range for a like-for-like comparison."""
    ax = _ax(ax)
    o = obs.load_survey(survey)
    ci = {n: i for i, n in enumerate(obs.SURVEY_COLOR_NAMES)}
    # mock LSST colours
    u, g, r, i, zb = (sm.mag('LSST', b) for b in 'ugriz')
    mcol = {'u-g': u-g, 'g-i': g-i, 'r-i': r-i, 'i-z': i-zb}
    if zmax is None:
        zmax = {'SDSS': 0.42, 'DEEP2': 1.05}.get(survey, sm.redshift.max())
    msel = np.isfinite(mcol[cx]) & np.isfinite(mcol[cy]) & (sm.redshift <= zmax)
    ax.scatter(mcol[cx][msel][::20], mcol[cy][msel][::20], s=2, alpha=.15,
               color='0.4', label=f'mock LSST (z<{zmax})', rasterized=True)
    os_ = o['zspec'] <= zmax
    ax.scatter(o['colors'][os_, ci[cx]], o['colors'][os_, ci[cy]], s=4,
               alpha=.4, color='#c23b3b', label=f'{survey} (data)')
    ax.set(xlabel=cx, ylabel=cy, title=f'mock vs {survey} colours')
    ax.legend(frameon=False, fontsize=8, markerscale=3)
    return ax


# ------------------------------------------------------- 08 WISE W1-W2 (cheap)
def wise_w1w2(sm, ax=None, zmax=0.4):
    """W1-W2 from the photometry file (W1,W2 are stored) vs XSCOS.  Fast; for
    W2-W3 use wise_colors() with SEDs."""
    ax = _ax(ax)
    w1 = sm.mag('WISE', 'W1'); w2 = sm.mag('WISE', 'W2')       # AB
    col = (w1-w2) - (WISE_VEGA_TO_AB['W1']-WISE_VEGA_TO_AB['W2'])   # -> Vega
    sel = np.isfinite(col) & (sm.redshift < zmax)
    x = obs.load_xscos()
    ax.hist(col[sel], bins=np.linspace(-0.6, 1.2, 60), density=True,
            histtype='step', color='0.35', lw=1.8, label=f'mock (z<{zmax})')
    ax.hist(x['colors'][:, 0], bins=np.linspace(-0.6, 1.2, 60), density=True,
            histtype='step', color='#e07b39', lw=1.8, label='WISExSCOS')
    ax.set(xlabel='W1-W2 [Vega]', ylabel='density', title='WISE W1-W2')
    ax.legend(frameon=False, fontsize=9)
    return ax


# =======================================================  SED-based panels ===
def _load_wise_rsr():
    import os
    fdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data', 'filters')
    curves = []
    for b in (1, 2, 3, 4):
        d = np.loadtxt(os.path.join(fdir, f'RSR-W{b}.txt'))
        keep = d[:, 1] > 0
        curves.append((d[keep, 0]*1e4, d[keep, 1]))          # micron->A, trans
    return curves


def project_wise_vega(wave_rest, sed_fnu_jy, z):
    """Project rest-frame f_nu SEDs (Jy) through the WISE W1-W4 RSRs -> Vega
    mags.  wave_rest [A], sed_fnu_jy [nrows, nwave], z [nrows]."""
    curves = _load_wise_rsr()
    wobs = wave_rest[None, :] * (1 + np.asarray(z)[:, None])
    voff = np.array([WISE_VEGA_TO_AB[f'W{b}'] for b in (1, 2, 3, 4)])
    out = np.full((sed_fnu_jy.shape[0], 4), np.nan)
    for b, (fw, ft) in enumerate(curves):
        denom = np.trapz(ft/fw, fw)
        for i in range(sed_fnu_jy.shape[0]):
            if fw[0] < wobs[i, 0] or fw[-1] > wobs[i, -1]:
                continue
            fi = np.interp(fw, wobs[i], sed_fnu_jy[i])
            out[i, b] = np.trapz(fi*ft/fw, fw)/denom
    with np.errstate(divide='ignore', invalid='ignore'):
        return -2.5*np.log10(out/AB_ZERO_JY) - voff[None, :]


def wise_colors(sm, ax=None, n=4000, zmax=0.4, seed=0):
    """Full WISE W2-W3 vs XSCOS -- projects a mock SED subsample through the
    WISE RSRs (W3/W4 are not in the photometry file).  Needs seds=True.

    XSCOS is W3+W4 detected (star-forming biased); a W1-window mock keeps
    quiescent galaxies, so the mock median sits bluer -- a selection effect,
    not a model error (see the report notes)."""
    ax = _ax(ax)
    w1_ab = sm.mag('WISE', 'W1')
    win = (np.isfinite(w1_ab) & (w1_ab > 16.1) & (w1_ab < 19.7)
           & (sm.redshift < zmax))
    rows = sm.sample_seds(n, seed=seed, mask=win)
    wave, seds = sm.seds(rows=rows)
    mag = project_wise_vega(wave, seds, sm.redshift[rows])
    ok = np.all(np.isfinite(mag), axis=1)
    w2w3 = mag[ok, 1] - mag[ok, 2]
    x = obs.load_xscos()
    ax.hist(w2w3, bins=np.linspace(0, 6, 50), density=True, histtype='step',
            color='0.35', lw=1.9, label=f'mock (median {np.median(w2w3):.2f})')
    ax.hist(x['colors'][:, 1], bins=np.linspace(0, 6, 50), density=True,
            histtype='step', color='#e07b39', lw=1.9,
            label=f'XSCOS (median {np.median(x["colors"][:,1]):.2f})')
    ax.set(xlabel='W2-W3 [Vega]', ylabel='density',
           title='WISE W2-W3 (SED-projected, z-matched)')
    ax.legend(frameon=False, fontsize=8)
    return ax


def example_seds(sm, ax=None, n=6, seed=1, zmax=0.5):
    """A handful of observed-frame mock SEDs (f_nu Jy)."""
    ax = _ax(ax)
    rows = sm.sample_seds(n, seed=seed, mask=sm.redshift < zmax)
    wave, seds = sm.seds(rows=rows)
    for k, r in enumerate(rows):
        wobs = wave*(1+sm.redshift[r])/1e4                    # micron
        good = seds[k] > 0
        ax.loglog(wobs[good], seds[k][good], lw=1,
                  label=f'z={sm.redshift[r]:.2f}, logM*={sm.logM[r]:.1f}')
    ax.set(xlabel=r'$\lambda_{\rm obs}\,[\mu m]$', ylabel=r'$f_\nu$ [Jy]',
           title='example painted SEDs')
    ax.legend(frameon=False, fontsize=7)
    return ax


def gama_stacks(sm, ax=None, z0=0.1, logm=10.5, dz=0.02, dm=0.25, seed=2,
                nmax=3000):
    """Median mock SED stack vs a GAMA median stack in a (z, logM*) bin."""
    ax = _ax(ax)
    g = obs.load_gama_stacks()
    key = min(g['stacks'], key=lambda k: abs(k[0]-z0)+abs(k[1]-logm))
    z0, logm = key
    mask = (np.abs(sm.redshift - z0) < dz) & (np.abs(sm.logM_zobs - logm) < dm)
    rows = sm.sample_seds(nmax, seed=seed, mask=mask)
    wave, seds = sm.seds(rows=rows)
    wobs = wave*(1+z0)/1e4                                     # micron
    med = np.median(seds, axis=0)*1e3                          # Jy -> mJy
    ax.loglog(wobs, med, color='#1d4e79', lw=1.8,
              label=f'mock (N={rows.size})')
    ax.loglog(g['wave_um'], g['stacks'][key], 'o', color='#e07b39', ms=4,
              label='GAMA median')
    ax.set(xlim=(0.3, 8), xlabel=r'$\lambda_{\rm obs}\,[\mu m]$',
           ylabel=r'$f_\nu$ [mJy]',
           title=f'GAMA stack: z~{z0}, logM*~{logm}')
    ax.legend(frameon=False, fontsize=8)
    return ax


# ----------------------------------------------------------- convenience grid
def report(sm, seds=False, save=None):
    """A compact multi-panel summary (the cheap panels, + SED panels if the
    SuperMock was opened with seds=True)."""
    import matplotlib.pyplot as plt
    have_sed = seds and bool(sm._sed_paths)
    panels = [redshift_distribution, gsmf, smhm, luminosity_function,
              number_counts, wise_w1w2]
    if have_sed:
        panels += [wise_colors, gama_stacks, example_seds]
    ncol = 3
    nrow = int(np.ceil(len(panels)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6*ncol, 4.2*nrow))
    for p, ax in zip(panels, np.ravel(axes)):
        try:
            p(sm, ax=ax)
        except Exception as e:                      # keep the grid drawing
            ax.text(.5, .5, f'{p.__name__}:\n{e}', ha='center', fontsize=7)
    for ax in np.ravel(axes)[len(panels):]:
        ax.axis('off')
    fig.suptitle(f'SuperMock report -- patches {sm.patches}, '
                 f'{sm.area_deg2:.0f} deg2, {sm.n:,} galaxies', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    if save:
        fig.savefig(save, dpi=130)
    return fig
