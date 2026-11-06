# Data

Raw telemetry of two LoRaWAN soil-moisture / rain nodes on the same 33° hillslope. The files are as logged, before any cleaning. All cleaning is done in the notebook main cell (see the main README).

| File | Node | Records | Period | SHA-256 |
|---|---|---|---|---|
| `dev_108_updated.csv` | Dev108 (source node) | 144,847 | 1 Nov 2025 – 29 Sep 2026 | `dcbd17de2429541329844714af57770d64d40001d31f98cd70b54f7f82eb4a74` |
| `dev_107_updated.csv` | Dev107 (transfer target) | 116,733 | 1 Nov 2025 – 29 Sep 2026 | `cbb4776490a5816119265a774afccab28be05badf8a7e3217a27b0682e5f2212` |

## Columns

| Column | Unit | Meaning |
|---|---|---|
| `timestamp` | local time (UTC+7) | logger time stamp, nominal 1-min interval with outages |
| `devID` | – | node number (108 or 107) |
| `theta` | m³ m⁻³ | volumetric water content at 0.30 m depth |
| `rain` | mm | tipping-bucket total since the last counter reset (0.2794 mm per tip). It rises while it rains, holds after the rain stops and drops at a reset. |

## Known features of the raw files, handled in the code

- **Duplicate timestamps:** 612 rows in Dev108, 462 in Dev107. Removed.
- **Sensor warm-up:** the first 268 rows of Dev108 (θ ≈ 0.05 rising to 0.41). Trimmed.
- **Logger infill:** straight-line runs written by the logger between uploads. Flagged and removed: 5,253 Dev108 and 6,935 Dev107 values on the 2-min grid.
- **Outages:**
  - shared outages 6 Jun – 12 Jul and 9 Aug – 3 Sep 2026;
  - longest gap 35.2 d (Dev108) and 93.8 d (Dev107);
  - outages longer than 30 min split the record into segments.

## Soil and node properties used in the model

| | Dev108 | Dev107 |
|---|---|---|
| Texture | sandy clay loam (30.48 % clay, 59.52 % sand) | sandy clay loam (29.48 % clay, 58.52 % sand) |
| θr | 0.05 | 0.05 |
| θs | 0.55 | 0.463 |
| α (m⁻¹) | 5.9 | 5.9 |
| n | 1.48 | 1.48 |
| Ks (m h⁻¹) | 0.0043 | 0.0041 |
| Sensor depth (m) | 0.30 | 0.30 |

- θs is anchored to each probe's sustained maximum in saturating rain.
- θr, α and n are shared because the two nodes have the same texture on the same slope.
- Ks is each node's pedotransfer estimate.

## Terms of use

The data are provided for peer review and for reproducing the results of the paper. Please cite the paper when using them.

## External records (`external/`)

Public-domain USGS data used for the transfer tests to other sites (manuscript Section 2.5; scripts in `external_sites/`). Columns: `timestamp` (UTC), `theta` (m³ m⁻³), `rain` (cumulative, mm).

| File | Site | Probe depth | Step | Period | Source |
|---|---|---|---|---|---|
| `utuado_formatted.csv` | Utuado, Puerto Rico | 0.27 m (SP1) | 15 min | 1 Dec 2018 – 31 Aug 2019 | Smith et al. (2020), https://doi.org/10.5066/P9548YK2 |
| `california_formatted.csv` | Two Towers landslide, California | 0.19 m | 1 h | 19 Jan 2017 – 29 Apr 2020 | Schulz and Smith (2021), https://doi.org/10.5066/P9NSUMOY |

Utuado soil parameters, from the laboratory tests of the same release (drying branch, sample at 0.22 m):
- θs 0.46, θr 0.10;
- α 0.042 cm⁻¹, n 1.586;
- Ks 1.0 × 10⁻⁴ cm s⁻¹.

The models use θr 0.05, because the record minimum is 0.086.
