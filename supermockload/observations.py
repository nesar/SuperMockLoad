"""Bundled observational comparison data (ships with the package, no synthetic
data needed).  All small, downsampled to what the report comparisons use.

    load_survey('SDSS'|'DEEP2') -> dict(colors[N,5], err_colors[N,5], zspec[N])
        colors = [u-g, g-i, r-i, i-z, mag_i]  (AB)
    load_xscos()  -> dict(colors[N,4], err_colors[N,4], zphot[N])
        colors = [W1-W2, W2-W3, W3-W4, W1]  (Vega); W3+W4 detected => SF-biased
    load_allwise_locus() -> dict(colors[N,4], err_colors)
        colors = [W1-W2, W2-W3, W3-W4, W1] (Vega); AllWISE galaxy locus, NO z
    load_gama_stacks() -> dict(wave_um, stacks{(z,logM): f_nu_mJy}, zbins, mbins)
        Richard's GAMA median SED stacks (observed mJy) in z x M* bins
"""
import os

import numpy as np

_OBS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'data', 'observations')

SURVEY_COLOR_NAMES = ['u-g', 'g-i', 'r-i', 'i-z', 'mag_i']
WISE_COLOR_NAMES = ['W1-W2', 'W2-W3', 'W3-W4', 'W1']


def load_survey(name):
    d = np.load(os.path.join(_OBS, f'{name.lower()}.npz'))
    return {k: d[k] for k in d.files}


def load_xscos():
    d = np.load(os.path.join(_OBS, 'xscos_wise.npz'))
    return {k: d[k] for k in d.files}


def load_allwise_locus():
    d = np.load(os.path.join(_OBS, 'allwise_locus.npz'))
    return {k: d[k] for k in d.files}


def load_gama_stacks():
    """Parse the GAMA median-stack CSV into (z, logM*) keyed spectra."""
    path = os.path.join(_OBS, 'gama_median_stacks.csv')
    rows = np.genfromtxt(path, delimiter=',', names=True)
    cols = rows.dtype.names
    wave = rows[cols[0]]                                   # lambda_micron
    stacks, zset, mset = {}, [], []
    for c in cols[1:]:                                     # f_nu_mJy_z<z>_m<M>
        parts = c.split('_')
        z = float(parts[-2][1:]); m = float(parts[-1][1:])
        stacks[(z, m)] = rows[c]
        zset.append(z); mset.append(m)
    return dict(wave_um=wave, stacks=stacks,
                zbins=sorted(set(zset)), mbins=sorted(set(mset)))
