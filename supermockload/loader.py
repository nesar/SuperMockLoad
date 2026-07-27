"""Load SuperMock lightcone patches (catalog / photometry / luminosities / SEDs).

One simulation today (HACC Last Journey); the API is patch-oriented so several
patches -- eventually the full sky -- combine transparently.

A patch on disk is four row-aligned HDF5 files under a data root:

    <root>/lightcone_catalogs/lightcone_galaxies_skypatch_<P>.h5   per-core groups
    <root>/photometry/photometry_skypatch_<P>.h5                   flat, per survey
    <root>/luminosities/luminosities_skypatch_<P>.h5              flat
    <root>/seds/seds_skypatch_<P>.h5                               flat, LARGE

The catalog is per-core groups; the flat files are concatenated in the SAME
(sorted group) order, so row i is the same galaxy in every file.  Verified via
`core_tag` when SEDs are opened.

Loading is lazy and downsample-aware:
  * catalog scalar fields load eagerly (cheap);
  * photometry (esp. the 102 SPHEREx bands, ~15 GB/patch) and luminosities load
    per survey/column on first access and are cached;
  * SEDs (~1.6 TB/patch) are only pulled for explicit rows via .seds();
  * `downsample=` keeps a random subset per patch -- essential for many-patch /
    full-sky exploration and for holding SEDs/SPHEREx in memory.

    from supermockload import SuperMock
    sm = SuperMock(3)                              # one patch, all rows
    sm = SuperMock([3, 5], downsample=500_000)     # combined, 500k/patch
    z, mi = sm.redshift, sm.mag('LSST', 'i')
    wave, sed = sm.seds(rows=sm.sample(1000))      # SEDs only when asked
"""
from __future__ import annotations

import glob
import json
import os

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))     # .../supermockload
_DATA = os.path.join(_HERE, 'data')                    # bundled package data
with open(os.path.join(_DATA, 'band_names.json')) as _f:
    BAND_NAMES = json.load(_f)

# default SYNTHETIC-data root: the directory the repo sits in (…/Mocks_v3_data),
# overridable via $SUPERMOCK_DATA or root=.
_SCALAR_FIELDS = ('redshift', 'ra', 'dec', 'stellar_mass', 'stellar_mass_zobs',
                  'peak_mass', 'central', 'merged', 'core_tag', 'fof_halo_tag')


def _resolve_root(root):
    """Where the synthetic patch files live.  Resolved at CALL time (not at
    import) so a `$SUPERMOCK_DATA` set in the session/notebook is picked up:
        explicit root=  >  $SUPERMOCK_DATA  >  repo-checked-out-inside-data  >  cwd
    The last two are conveniences; a missing patch raises a FileNotFoundError
    that names the path tried, so a wrong root is obvious."""
    if root:
        return root
    env = os.environ.get('SUPERMOCK_DATA')
    if env:
        return env
    guess = os.path.dirname(os.path.dirname(_HERE))     # repo inside the data dir
    if os.path.isdir(os.path.join(guess, 'lightcone_catalogs')):
        return guess
    return os.getcwd()


def available_patches(root=None):
    """Skypatch ids that have a catalog under `root` (or $SUPERMOCK_DATA)."""
    root = _resolve_root(root)
    pat = os.path.join(root, 'lightcone_catalogs',
                       'lightcone_galaxies_skypatch_*.h5')
    ids = []
    for f in glob.glob(pat):
        base = os.path.basename(f)
        if '.shard' in base:
            continue
        try:
            ids.append(int(base.split('skypatch_')[1].split('.h5')[0]))
        except (IndexError, ValueError):
            pass
    return sorted(ids)


class SuperMock:
    """Row-aligned view of one or more patches.

    Parameters
    ----------
    patches : int or sequence of int
    root : str, optional          data root ($SUPERMOCK_DATA or package parent).
    fields : sequence of str       catalog scalar fields (default: standard set).
    extra_fields : sequence of str additional catalog fields, e.g. 'sfh','mah'.
    photometry, luminosities : bool  enable those files (default True).
    seds : bool                    open SED handles (default False).
    downsample : int or float      keep this many rows per patch (int) or this
                                   fraction (0<f<=1) -- random, seeded.  None =
                                   all rows.
    seed : int                     downsample RNG seed.
    """

    def __init__(self, patches, root=None, fields=None, extra_fields=(),
                 photometry=True, luminosities=True, seds=False,
                 downsample=None, seed=0, verbose=True):
        self.root = _resolve_root(root)
        self.patches = [int(patches)] if np.isscalar(patches) else [int(p) for p in patches]
        self._verbose = verbose
        self._has_phot = photometry
        self._has_lum = luminosities
        self._downsample = downsample
        self._seed = seed
        fields = tuple(fields) if fields is not None else _SCALAR_FIELDS
        self._fields = tuple(dict.fromkeys(fields + tuple(extra_fields)))

        self.cat = {}                  # field -> concatenated array
        self.phot = {}                 # survey -> matrix (lazy, cached)
        self.lum = {}                  # column -> array  (lazy, cached)
        self._sed_paths = {}           # patch -> sed path
        self._file_row = []            # per kept row: original per-patch file row
        self._patch_id = []            # per kept row: patch id
        self._keep = {}                # patch -> file rows kept (sorted)
        self._n_full = 0               # total rows on disk (before downsample)

        for P in self.patches:
            self._load_catalog(P)
            if seds:
                self._register_seds(P)
        for k in self._fields:
            if k in self.cat:
                self.cat[k] = np.concatenate(self.cat[k], axis=0)
        self._file_row = np.concatenate(self._file_row)
        self._patch_id = np.concatenate(self._patch_id)
        self.n = self._patch_id.size
        if seds:
            self._verify_sed_alignment()
        ds = '' if downsample is None else f', downsampled to {self.n:,}'
        self._log(f'ready: {self.n:,} galaxies, patches {self.patches}{ds} '
                  f'(photometry/luminosities/SEDs load on demand)')

    # ------------------------------------------------------------------ IO
    def _log(self, msg):
        if self._verbose:
            print(f'[SuperMock] {msg}', flush=True)

    def _path(self, P, kind, stem):
        return os.path.join(self.root, kind, f'{stem}_skypatch_{P}.h5')

    def _select(self, n_full, P):
        """File-row indices to keep for a patch of n_full rows."""
        if self._downsample is None:
            return None                          # sentinel: keep all
        if isinstance(self._downsample, float) and self._downsample <= 1.0:
            k = int(round(n_full * self._downsample))
        else:
            k = int(self._downsample)
        if k >= n_full:
            return np.arange(n_full)
        rng = np.random.default_rng(self._seed + P)
        return np.sort(rng.choice(n_full, k, replace=False))

    def _load_catalog(self, P):
        path = self._path(P, 'lightcone_catalogs', 'lightcone_galaxies')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'catalog not found:\n  {path}\n'
                f'root is {self.root!r}. Point it at the data dir with '
                f"SuperMock({P}, root='/path/to/data') or "
                "os.environ['SUPERMOCK_DATA']='/path/to/data'.")
        with h5py.File(path, 'r') as f:
            groups = sorted(f.keys())            # SAME order as the flat files
            sizes = [f[g]['redshift'].shape[0] for g in groups]
            n_full = int(np.sum(sizes))
            keep = self._select(n_full, P)       # None => all
            offs = np.concatenate([[0], np.cumsum(sizes)])
            for k in self._fields:
                if k not in f[groups[0]]:
                    self._log(f'  patch {P}: field "{k}" absent, skipped')
                    continue
                blocks = []
                for gi, g in enumerate(groups):
                    if keep is None:
                        blocks.append(f[g][k][...])
                    else:                        # rows of this group that survive
                        lo, hi = offs[gi], offs[gi+1]
                        sel = keep[(keep >= lo) & (keep < hi)] - lo
                        if sel.size:
                            blocks.append(f[g][k][...][sel])
                self.cat.setdefault(k, []).append(
                    np.concatenate(blocks, axis=0) if blocks
                    else np.empty((0,), f[groups[0]][k].dtype))
        self._n_full += n_full                      # for the completeness weight
        fr = np.arange(n_full) if keep is None else keep
        self._keep[P] = fr
        self._file_row.append(fr)
        self._patch_id.append(np.full(fr.size, P, dtype=np.int32))
        self._log(f'  patch {P}: {fr.size:,} rows'
                  + ('' if keep is None else f' (of {n_full:,})'))

    def _read_flat_key(self, kind, stem, key, chunk=2_000_000):
        """One dataset (survey matrix / lum column) across patches, with the
        per-patch downsample selection applied.  Reads in row slabs so peak
        memory stays ~chunk rows even when the kept subset spans the file."""
        blocks = []
        for P in self.patches:
            path = self._path(P, kind, stem)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with h5py.File(path, 'r') as f:
                if key not in f:
                    raise KeyError(f'{key!r} not in {os.path.basename(path)} '
                                   f'(have {list(f.keys())})')
                ds = f[key]
                fr = self._keep[P]
                if fr.size == ds.shape[0]:                # keeping all rows
                    blocks.append(ds[...])
                    continue
                parts = []
                for s in range(0, ds.shape[0], chunk):
                    e = min(s + chunk, ds.shape[0])
                    sel = fr[(fr >= s) & (fr < e)] - s
                    if sel.size:
                        parts.append(ds[s:e][sel])
                blocks.append(np.concatenate(parts, axis=0))
        return np.concatenate(blocks, axis=0)

    def _register_seds(self, P):
        path = self._path(P, 'seds', 'seds')
        if os.path.exists(path):
            self._sed_paths[P] = path
        else:
            self._log(f'  patch {P}: no SED file, skipped')

    # ----------------------------------------------------------- catalog access
    def field(self, name):
        """Any loaded catalog field (scalar or 2-D, e.g. 'sfh')."""
        if name not in self.cat:
            raise KeyError(f'{name!r} not loaded; add via extra_fields=')
        return self.cat[name]

    def __getattr__(self, name):
        cat = self.__dict__.get('cat', {})
        if name in cat:
            return cat[name]
        raise AttributeError(name)

    @property
    def logM(self):
        """log10 stellar mass (a=1 matched-UM mass)."""
        return np.log10(np.clip(self.cat['stellar_mass'], 1, None))

    @property
    def logM_zobs(self):
        """log10 observed-epoch stellar mass (SFH integrated to z_obs)."""
        return np.log10(np.clip(self.cat['stellar_mass_zobs'], 1, None))

    @property
    def patch_id(self):
        return self._patch_id

    # -------------------------------------------------------- photometry / lum
    def survey(self, name):
        """Full (N, nband) photometry matrix for a survey (lazy, cached)."""
        if not self._has_phot:
            raise RuntimeError('opened with photometry=False')
        if name not in self.phot:
            if self.root is None:                 # snapshot: only saved surveys
                raise KeyError(f'survey {name!r} not in this snapshot '
                               f'(saved: {list(self.phot)}); re-save with it')
            self.phot[name] = self._read_flat_key('photometry', 'photometry', name)
        return self.phot[name]

    def bands(self, name):
        """Band names (column labels) for a survey."""
        return BAND_NAMES[name]

    def wavelengths(self, name):
        """Effective wavelength [micron] of each band of a survey (x-axis for
        photometric SEDs, e.g. the 102 SPHEREx channels)."""
        wl = self.__dict__.get('_wl')
        if wl is None:
            wl = dict(np.load(os.path.join(_DATA, 'band_wavelengths.npz')))
            self._wl = wl
        return wl[name]

    def bandpass(self, survey, band=None):
        """Filter response curve(s) used to convolve the SEDs into photometry.
        Returns (wavelength[micron], transmission) for one `band`, or a dict
        {band_name: (wave, trans)} for the whole survey if band is None.
        Transmission is the per-photon relative response (peak-normalised)."""
        bp = self.__dict__.get('_bp')
        if bp is None:
            bp = {}
            with h5py.File(os.path.join(_DATA, 'bandpasses.h5'), 'r') as f:
                for s in f:
                    bp[s] = {k: f[s][k][...] for k in f[s]}
            self._bp = bp
        g = bp[survey]
        if band is None:
            return {k: (v[0], v[1]) for k, v in g.items()}
        key = band if band in g else None
        if key is None:
            for cand in (f'{survey}_{band}', f'{survey}_{band}_DEcam'):
                if cand in g:
                    key = cand
                    break
        if key is None:
            hits = [k for k in g if k.split('_')[-1] == band or k.endswith('_'+band)]
            key = hits[0] if len(hits) == 1 else None
        if key is None:
            raise KeyError(f'band {band!r} not in {survey}; have {list(g)[:6]}...')
        d = g[key]
        return d[0], d[1]

    def mag(self, survey, band):
        """AB mag in one band, e.g. mag('LSST','i') or mag('WISE','W1')."""
        return self.survey(survey)[:, _band_index(BAND_NAMES[survey], survey, band)]

    def luminosity(self, name):
        """Luminosities column: 'L_BOL_LSUN','L_8_33UM_LSUN','M_ABS_SDSS',
        'M_ABS_WISE','redshift' (lazy, cached)."""
        if not self._has_lum:
            raise RuntimeError('opened with luminosities=False')
        if name not in self.lum:
            if self.root is None:
                raise KeyError(f'luminosity {name!r} not in this snapshot')
            self.lum[name] = self._read_flat_key('luminosities', 'luminosities', name)
        return self.lum[name]

    def abs_mag(self, survey, band):
        """Rest-frame absolute mag, survey in {'SDSS','WISE'}."""
        key = f'M_ABS_{survey}'
        return self.luminosity(key)[:, BAND_NAMES[f'_LUM_{key}'].index(band)]

    # ------------------------------------------------------------------ SEDs
    def sed_wavelength(self):
        P = next(iter(self._sed_paths))
        with h5py.File(self._sed_paths[P], 'r') as f:
            return f['wave_rest'][...]

    def seds(self, rows=None, chunk=20000):
        """(wave_rest, SED[nrows, nwave]) for `rows` (global indices into this
        SuperMock).  SEDs are f_nu [Jy] on the REST grid; observed wavelength is
        wave_rest*(1+z).  Full SEDs are ~1.6 TB/patch -- always pass `rows`."""
        # snapshot with a stored SED subset
        cache = self.__dict__.get('_cache_sed')
        if cache is not None:
            wave, crows, csed = cache
            if rows is None:
                return wave, csed
            pos = {int(r): i for i, r in enumerate(crows)}
            miss = [int(r) for r in np.asarray(rows) if int(r) not in pos]
            if miss:
                raise KeyError(f'{len(miss)} requested rows not in the snapshot '
                               'SED subset; re-save with those rows')
            idx = [pos[int(r)] for r in np.asarray(rows)]
            return wave, csed[idx]
        if not self._sed_paths:
            raise RuntimeError('open with SuperMock(..., seds=True) first')
        rows = np.arange(self.n) if rows is None else np.asarray(rows)
        wave = self.sed_wavelength()
        out = np.empty((rows.size, wave.size), dtype=np.float32)
        for P in self.patches:
            if P not in self._sed_paths:
                continue
            pm = self._patch_id[rows] == P
            if not pm.any():
                continue
            g_rows = rows[pm]
            file_rows = self._file_row[g_rows]           # -> SED file rows
            order = np.argsort(file_rows)
            fr = file_rows[order]
            with h5py.File(self._sed_paths[P], 'r') as f:
                ds = f['SED']
                for s in range(0, fr.size, chunk):
                    idx = fr[s:s+chunk]
                    block = ds[idx.min():idx.max()+1][idx - idx.min()]
                    out[np.flatnonzero(pm)[order[s:s+chunk]]] = block
        return wave, out

    # ------------------------------------------------------------- helpers
    def sample(self, n, seed=0, mask=None):
        """Random global row indices (optionally within a boolean mask)."""
        rng = np.random.default_rng(seed)
        pool = np.arange(self.n) if mask is None else np.flatnonzero(mask)
        return pool if pool.size <= n else np.sort(rng.choice(pool, n, replace=False))

    @property
    def sed_rows(self):
        """Global rows for which SEDs are available: all rows for a live
        SuperMock(seds=True), or just the stored subset for a snapshot;
        None if no SEDs.  SED plots sample within this."""
        cache = self.__dict__.get('_cache_sed')
        if cache is not None:
            return cache[1]
        if self._sed_paths:
            return np.arange(self.n)
        return None

    def sample_seds(self, n, seed=0, mask=None):
        """Like sample(), but restricted to rows that HAVE SEDs (respects a
        snapshot's stored SED subset)."""
        sr = self.sed_rows
        if sr is None:
            raise RuntimeError('no SEDs; open with seds=True or a snapshot')
        m = np.zeros(self.n, bool)
        m[sr] = True
        if mask is not None:
            m &= mask
        return self.sample(n, seed=seed, mask=m)

    def _verify_sed_alignment(self, nchk=100000):
        ct = self.cat['core_tag']
        for P in self.patches:
            if P not in self._sed_paths:
                continue
            pm = self._patch_id == P
            fr = self._file_row[pm]
            with h5py.File(self._sed_paths[P], 'r') as f:
                if f['core_tag'].shape[0] < (int(fr[-1])+1):
                    raise ValueError(f'patch {P}: SED file shorter than catalog')
                k = min(nchk, fr.size)
                lo, hi = int(fr[0]), int(fr[k-1])+1
                sed_ct = f['core_tag'][lo:hi][fr[:k]-lo]
            if not np.array_equal(sed_ct, ct[pm][:k]):
                raise ValueError(f'patch {P}: SED/catalog core_tag mismatch')
        self._log('  SED row alignment verified (core_tag)')

    @property
    def area_deg2(self):
        """Total footprint (sum of per-patch wrap-aware bbox areas).  Uses the
        FULL catalog ra/dec, so it is unaffected by downsampling."""
        if self.__dict__.get('_area') is not None:      # snapshot
            return self._area
        return float(sum(_patch_area(self.root, P) for P in self.patches))

    @property
    def completeness(self):
        """Fraction of the full population retained (1.0 unless downsampled).
        Number densities must divide by this: an object represents
        1/completeness galaxies of the full catalog."""
        nf = self.__dict__.get('_n_full', 0)
        return 1.0 if not nf else self.n / nf

    @property
    def weight(self):
        """Per-object weight so downsampled counts recover full densities:
        weight = 1/completeness for every row (a scalar broadcast)."""
        return 1.0 / self.completeness

    # ------------------------------------------------- simulation time grids
    def _grids(self):
        g = self.__dict__.get('_tg')
        if g is None:
            g = dict(np.load(os.path.join(_DATA, 'time_grids.npz')))
            self._tg = g
        return g

    @property
    def mah_time(self):
        """Cosmic age [Gyr] of the 101 MAH columns (Last Journey snapshots).
        See also mah_redshift.  `sm.field('mah')` is M_halo(t) [Msun/h]."""
        return self._grids()['mah_age_gyr']

    @property
    def mah_redshift(self):
        return self._grids()['mah_redshift']

    @property
    def sfh_time(self):
        """Cosmic age [Gyr] of the 117 SFH columns (SMDPL bins).
        `sm.field('sfh')` is SFR(t) [Msun/yr]."""
        return self._grids()['sfh_age_gyr']

    @property
    def sfh_redshift(self):
        return self._grids()['sfh_redshift']

    def catalog_fields(self):
        """List every catalog field on disk (loaded or not)."""
        P = self.patches[0]
        path = self._path(P, 'lightcone_catalogs', 'lightcone_galaxies')
        with h5py.File(path, 'r') as f:
            g = sorted(f.keys())[0]
            return {k: (f[g][k].shape[1:], str(f[g][k].dtype))
                    for k in sorted(f[g].keys())}

    def loaded_fields(self):
        """Catalog fields currently in memory."""
        return sorted(self.cat)

    # -------------------------------------------------- fast-reload cache I/O
    def save(self, path, surveys=('LSST', 'WISE'), seds_rows=None):
        """Write a compact, UNCOMPRESSED snapshot (catalog + the named
        photometry surveys + luminosities + optional SED subset) for instant
        reload with SuperMock.from_file().  The recommended workflow: build a
        downsampled SuperMock once (slow, gzip), snapshot it, then explore the
        snapshot.  SEDs are stored only for `seds_rows` (global indices)."""
        with h5py.File(path, 'w') as f:
            f.attrs['patches'] = self.patches
            f.attrs['area_deg2'] = self.area_deg2
            f.attrs['n'] = self.n
            f.attrs['n_full'] = self.__dict__.get('_n_full', self.n)
            gc = f.create_group('catalog')
            for k, v in self.cat.items():
                gc.create_dataset(k, data=v)
            gc.create_dataset('_patch_id', data=self._patch_id)
            if self._has_phot:
                gp = f.create_group('photometry')
                for s in surveys:
                    gp.create_dataset(s, data=self.survey(s))
            if self._has_lum:
                gl = f.create_group('luminosities')
                for k in ('L_BOL_LSUN', 'L_8_33UM_LSUN', 'M_ABS_SDSS',
                          'M_ABS_WISE', 'redshift'):
                    try:
                        gl.create_dataset(k, data=self.luminosity(k))
                    except KeyError:
                        pass
            if seds_rows is not None and self._sed_paths:
                wave, sed = self.seds(rows=np.asarray(seds_rows))
                gs = f.create_group('seds')
                gs.create_dataset('SED', data=sed)
                gs.create_dataset('wave_rest', data=wave)
                gs.create_dataset('rows', data=np.asarray(seds_rows))
        self._log(f'snapshot -> {path} '
                  f'({os.path.getsize(path)/1e6:.0f} MB, surveys={list(surveys)})')

    @classmethod
    def from_file(cls, path, verbose=True):
        """Load a snapshot written by .save() (no access to the patch files)."""
        self = cls.__new__(cls)
        self._verbose = verbose
        self.root = None
        self.phot = {}
        self.lum = {}
        self.cat = {}
        self._sed_paths = {}
        self._cache_sed = None
        with h5py.File(path, 'r') as f:
            self.patches = [int(p) for p in f.attrs['patches']]
            self._area = float(f.attrs['area_deg2'])
            self._n_full = int(f.attrs.get('n_full', f.attrs['n']))
            for k in f['catalog']:
                self.cat[k] = f['catalog'][k][...]
            self._patch_id = self.cat.pop('_patch_id')
            self._has_phot = 'photometry' in f
            if self._has_phot:
                for s in f['photometry']:
                    self.phot[s] = f['photometry'][s][...]
            self._has_lum = 'luminosities' in f
            if self._has_lum:
                for k in f['luminosities']:
                    self.lum[k] = f['luminosities'][k][...]
            if 'seds' in f:
                self._cache_sed = (f['seds']['wave_rest'][...],
                                   f['seds']['rows'][...], f['seds']['SED'][...])
        self.n = self._patch_id.size
        self._downsample = None
        self._log(f'loaded snapshot {path}: {self.n:,} galaxies, '
                  f'surveys={list(self.phot)}')
        return self

    def __repr__(self):
        return (f'SuperMock(patches={self.patches}, n={getattr(self,"n","?"):,}, '
                f'seds={bool(self._sed_paths) or self.__dict__.get("_cache_sed") is not None})')


def _band_index(names, survey, band):
    for cand in (band, f'{survey}_{band}', f'{survey}_{band}_DEcam'):
        if cand in names:
            return names.index(cand)
    hits = [i for i, nm in enumerate(names)
            if nm.split('_')[-1] == band or nm.endswith('_' + band)]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f'band {band!r} not found in {survey} bands {names}')


def _patch_area(root, P):
    """Wrap-aware bbox area from ra/dec (matches the report's area)."""
    path = os.path.join(root, 'lightcone_catalogs',
                        f'lightcone_galaxies_skypatch_{P}.h5')
    with h5py.File(path, 'r') as f:
        gs = sorted(f.keys())
        ra = np.concatenate([f[g]['ra'][...] for g in gs])
        dec = np.concatenate([f[g]['dec'][...] for g in gs])
    ra_w = min(np.ptp(ra), np.ptp((ra + 180) % 360))
    dec_w = np.ptp(dec)
    return float(ra_w * dec_w * np.cos(np.deg2rad(0.5*(dec.min()+dec.max()))))
