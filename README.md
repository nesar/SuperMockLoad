# SuperMockLoad

Load and plot the **SuperMock** galaxy catalogs — synthetic
galaxies painted onto the HACC *Last Journey* simulation (redshifts, positions,
stellar masses, SFHs, full optical–IR SEDs, and multi-survey photometry).

The package ships a sample of the **observational comparison data** (SDSS, DEEP2,
WISE×SuperCOSMOS, AllWISE, GAMA — all downsampled, ~3 MB) and the plotting code
to reproduce the validation report, but **not** the synthetic catalogs
themselves (they are large; point the loader at wherever they live).

```python
from supermockload import SuperMock, plots

sm = SuperMock(3)                      # one skypatch (catalog + photometry + lum)
plots.gsmf(sm)                         # stellar mass function vs Baldry+12
plots.luminosity_function(sm)          # r-band LF vs Blanton+03
plots.number_counts(sm, 'LSST', 'i')

wave, sed = sm.seds(rows=sm.sample(100))   # SEDs are opt-in and pulled by row
```

## Install

```bash
pip install -e .          # from the repo root  (or: pip install .)
```

Requires numpy, h5py, matplotlib, astropy.

## Pointing at the data

A *patch* is four row-aligned HDF5 files under a data root:

```
<root>/lightcone_catalogs/lightcone_galaxies_skypatch_<P>.h5   catalog (per-core groups)
<root>/photometry/photometry_skypatch_<P>.h5                   AB mags, per survey
<root>/luminosities/luminosities_skypatch_<P>.h5              rest-frame luminosities
<root>/seds/seds_skypatch_<P>.h5                               f_nu SEDs (LARGE, ~1.6 TB/patch)
```

The loader finds them via `--root` / `SUPERMOCK_DATA`, defaulting to the
directory this package sits in:

```python
SuperMock(3, root='/path/to/Mocks_v3_data')       # explicit
# or:  export SUPERMOCK_DATA=/path/to/Mocks_v3_data
from supermockload import available_patches
available_patches()                               # -> [3, ...]
```

## Loading options

```python
SuperMock(3)                                  # all rows of one patch
SuperMock([3, 5])                             # several patches, combined
SuperMock(3, downsample=500_000)              # random 500k rows (fast, low memory)
SuperMock(3, downsample=0.1)                  # random 10%
SuperMock(3, seds=True)                        # also open SED handles
SuperMock(3, photometry=False)                # catalog only (skip the mag files)
SuperMock(3, extra_fields=('sfh', 'mah'))     # pull the SFH / MAH arrays too
```

Loading is **lazy**: only catalog scalars load up front; each photometry
survey / luminosity column / SED block reads on first access and caches.

### Fast repeat loads (snapshots)

The catalog and photometry files are gzip-compressed, so a first full read is
slow. For interactive work, downsample once and snapshot to an uncompressed
file that reloads instantly:

```python
sm = SuperMock(3, downsample=200_000, seds=True)
rows = sm.sample(3000)
sm.save('patch3_200k.h5', surveys=('LSST','WISE'), seds_rows=rows)

sm = SuperMock.from_file('patch3_200k.h5')    # instant; carries the SED subset
```

## Accessors

| call | returns |
|---|---|
| `sm.redshift`, `sm.ra`, `sm.dec` | catalog scalars (attribute access) |
| `sm.logM`, `sm.logM_zobs` | log10 stellar mass (a=1 / observed-epoch) |
| `sm.mag('LSST','i')` | AB mag in one band |
| `sm.survey('SPHEREx')` | full `(N, nband)` matrix; `sm.bands('SPHEREx')` for labels |
| `sm.luminosity('L_BOL_LSUN')` | a luminosities column |
| `sm.abs_mag('SDSS','r')` | rest-frame absolute mag |
| `sm.seds(rows=...)` | `(wave_rest, f_nu[Jy])`; observed λ = wave_rest·(1+z) |
| `sm.field('sfh')` | any loaded catalog field (2-D too) |
| `sm.area_deg2` | footprint (wrap-aware) |
| `sm.patch_id` | per-row patch id (for multi-patch) |

Surveys: `LSST WISE SPHEREx COSMOS LEGACYSURVEY 2MASS F784` (all AB).

## Plots

`supermockload.plots` reproduces the report panels against the bundled
observations. Cheap panels (photometry/luminosities only):

```python
plots.redshift_distribution(sm)   plots.gsmf(sm)          plots.smhm(sm)
plots.luminosity_function(sm)     plots.number_counts(sm) plots.optical_colors(sm,'SDSS')
plots.wise_w1w2(sm)               plots.report(sm)        # compact grid
```

SED panels (need `seds=True` or a snapshot with an SED subset):

```python
plots.wise_colors(sm)     # W2-W3 vs XSCOS (projects SEDs through WISE RSRs)
plots.example_seds(sm)    plots.gama_stacks(sm, z0=0.1, logm=10.5)
```

Each takes an optional `ax=` and returns the Axes, so they compose into your
own figures.

## Notebooks

- `notebooks/01_quickstart.ipynb` — load a patch, basic plots, snapshots.
- `notebooks/02_report_panels.ipynb` — every report panel vs observations.
- `notebooks/03_multipatch_and_seds.ipynb` — combine patches, work with SEDs.

## Notes

- One simulation today (Last Journey); the patch-oriented API extends to the
  full sky unchanged — `SuperMock([...])` concatenates and tracks `patch_id`.
- SEDs are `f_nu` in Jy on the **rest** wavelength grid; multiply the grid by
  `(1+z)` for observed wavelength (SEDs already include IGM + the observed-frame
  transform in their normalization).
- `stellar_mass` is the matched-UM a=1 mass; `stellar_mass_zobs` is the
  observed-epoch mass (SFH integrated to the galaxy's redshift) — use the
  latter for mass functions.
- Cosmology throughout is SMDPL flat ΛCDM, H0=67.77, Ωm=0.307115.
- The observed WISE×SCOS sample is W3+W4-detected (star-forming-biased); a
  W1-selected mock keeps quiescent galaxies, so the mock W2−W3 median sits
  bluer — a selection effect, not a model error. See `plots.wise_colors`.

See `data/observations/README.md` for the provenance of the bundled data.
