# Bundled observational comparison data

Small, downsampled to what the SuperMock validation compares against.  Loaded
via `supermockload.observations`.

| file | contents | provenance |
|---|---|---|
| `sdss.npz`  | colors [u-g,g-i,r-i,i-z,mag_i] (AB), err_colors, zspec; 40k | SDSS spectroscopic, downsampled |
| `deep2.npz` | same schema; ~13k | DEEP2 DR4 spectroscopic |
| `xscos_wise.npz` | Vega colors [W1-W2,W2-W3,W3-W4,W1], err_colors, zphot; ~4.3k | WISE×SuperCOSMOS photo-z (W3+W4 detected -> star-forming biased) |
| `allwise_locus.npz` | Vega colors [W1-W2,W2-W3,W3-W4,W1], err_colors; 40k | AllWISE galaxy locus (SNR cuts + W2-W3>min), NO redshifts |
| `gama_median_stacks.csv` | median observed-frame SED stacks (mJy) in z x logM* bins | GAMA (Richard's median stacks, 2022) |

`data/band_names.json` — column labels for each survey's photometry matrix.
`data/filters/RSR-W[1-4].txt` — WISE relative spectral responses, for
projecting mock SEDs to WISE colors (`plots.project_wise_vega`).

These are comparison/reference data only; the synthetic catalogs are not
included in the repository.
