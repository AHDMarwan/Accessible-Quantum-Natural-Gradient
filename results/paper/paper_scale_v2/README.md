# AQNG paper-scale results (frozen campaign + SGD/Adam follow-up)

This directory is the durable result package for the frozen AQNG paper-scale campaign.
It contains the complete aggregate terminal results, calibration/orientation diagnostics, lossless-compressed training curves and metric diagnostics, corrected paper summaries, and the matched SGD/Adam large/deep follow-up.

## Provenance

- Paper-scale workflow run: `31480796466` at `1aad0e2413436084d84c9b013ef4b75a32771d06`.
- SGD/Adam follow-up workflow run: `31522334431` at `82616cabacec2f8740a8955e0c750e169112c9eb`.
- Paper-scale matrix: 470 cells = 320 primary + 40 scaling + 30 depth + 40 readout-order + 40 finite-shot.
- Terminal paper-scale rows: 3160.
- Follow-up: 15 matched cells, 30 terminal rows (SGD and Adam only).

## Generic primary benchmark (U(1) negative control excluded)

| method | mean test loss | mean test accuracy |
|---|---:|---:|
| AQNG-aligned | 0.594494 | 0.7947 |
| AQNG-random | 0.595566 | 0.7914 |
| SGD | 0.608363 | 0.7858 |
| Adam | 0.620272 | 0.7731 |
| AQNG-physical | 0.622907 | 0.7878 |
| AQNG-Z0 | 0.635241 | 0.7828 |
| Full-QNG | 0.656169 | 0.7689 |
| Block-QNG | 0.678055 | 0.7606 |

## Large/deep matched comparison

| setting | method | mean test loss | mean test accuracy |
|---|---|---:|---:|
| n10_L3 | AQNG-physical | 0.241032 | 0.9600 |
| n10_L3 | AQNG-random | 0.252036 | 0.9600 |
| n10_L3 | AQNG-aligned | 0.253791 | 0.9867 |
| n10_L3 | Adam | 0.264865 | 0.9600 |
| n10_L3 | SGD | 0.407165 | 0.9333 |
| n10_L3 | Full-QNG | 0.429528 | 0.8533 |
| n6_L5 | AQNG-random | 0.126763 | 1.0000 |
| n6_L5 | AQNG-aligned | 0.138588 | 1.0000 |
| n6_L5 | AQNG-physical | 0.183283 | 1.0000 |
| n6_L5 | Adam | 0.184247 | 1.0000 |
| n6_L5 | SGD | 0.367199 | 0.9867 |
| n6_L5 | Full-QNG | 0.388011 | 0.9600 |
| n8_L3 | AQNG-aligned | 0.167762 | 1.0000 |
| n8_L3 | AQNG-random | 0.209617 | 0.9733 |
| n8_L3 | Adam | 0.231153 | 1.0000 |
| n8_L3 | AQNG-physical | 0.237676 | 0.9600 |
| n8_L3 | SGD | 0.425057 | 0.9333 |
| n8_L3 | Full-QNG | 0.476202 | 0.9333 |

## Files

- `raw/results_all.csv`: all 3160 terminal rows from the frozen paper-scale matrix.
- `raw/calibration_all.csv`: all 470 calibration rows.
- `raw/orientation_all.csv`: orientation diagnostics.
- `raw/curves_all.csv.gz`: complete training curves (lossless gzip).
- `raw/metric_diags_all.csv.gz`: complete metric diagnostics (lossless gzip).
- `summaries/`: corrected block summaries, geometry summaries, resource accounting, and planned paired tests.
- `followup/`: complete SGD/Adam follow-up aggregates plus the merged AQNG/Full-QNG/SGD/Adam comparison.
- `report.json`: provenance, completeness checks, and orientation/task correlation diagnostics.

## Reporting fix

The completed source workflow produced valid raw aggregate CSVs but empty block-summary CSVs because the reporting parser searched for tags such as `paper-primary`, while the actual artifact tags are `paper-run-primary-...`. This package regenerates those summaries from the unchanged raw tables.
