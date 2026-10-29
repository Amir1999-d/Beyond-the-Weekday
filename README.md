# Beyond the Weekday — holiday transit equity framework

Code for the MSc thesis *Beyond the Weekday: An Open-Data Framework for Holiday
Transit Equity, with an Application to Milan* (Amirhassan Darvishzade,
Politecnico di Milano, 2026; supervisor Prof. Fabio Borghetti; operator
co-advisor Alberto Bertolin, ATM).

Public transport is usually assessed on weekday service. This framework
measures how the reduced service run on public holidays is distributed across
populations with different levels of transit dependence, where a high holiday
service burden coincides with poor walk-up access to essential services, and
how much of a modelled budget-neutral reallocation of service survives the
operator's mode and depot structure.

The open-data core uses only GTFS timetables, small-area census data and
OpenStreetMap. Optional extensions use private data from ATM, Milan's
operator, which are not distributed.

## Reproducing the thesis

`thesis_code.ipynb` runs the whole pipeline. Its final cell compares 35
published figures with the run and writes `outputs/thesis_reproduction_check.csv`.
With the thesis inputs (see `data/README.md`) all 35 match. 24 of the checks
use open data only; the other 11 need the private ATM workbook and are
reported as `n/a` without it.

Selected figures it checks:

| Figure | Thesis value |
|---|---|
| Analytical network (stops) | 3,187 |
| Holiday/weekday social-burden ratio: served, stop-hour basis / penalised, stop level | 1.49 / 1.62 |
| Double-excluded stops (top-quartile holiday burden, and outside the walking catchment of at least one of the three essential-service categories) | 415 |
| Priority stops (double-excluded and reallocation recipients) | 184 |
| Moderate scenario (20% donor reduction): donors / recipients | 336 / 361 |
| Moderate scenario: service equivalents moved per day | 10,015 |
| Moderate scenario: Gini improvement / total burden reduction | 20.8% / 13.2% |
| Ceiling sweep: peak Gini improvement, at donor reduction | 22.9%, at 41% |
| Depot extension (ATM data), Moderate Gini improvement: bus and tram only / 2,000-equivalent inter-depot allowance / no inter-depot transfer | 15.9% / 14.5% / 10.2% |

The thesis run used Python 3.11.9, pandas 2.2.3 and numpy 2.2.4.

## Installation

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install notebook               # only to run thesis_code.ipynb
```

Place the inputs in `data/` as described in `data/README.md`, then check them:

```
python tools/make_data_manifest.py --check
```

## Running

Open `thesis_code.ipynb`, restart the kernel and run all cells. To run the
scripts directly, use the notebook's order:

| # | Script | Thesis | What it does |
|---|---|---|---|
| 1 | `build_stop_table.py` | Phases 1–2, §3.2–3.3 | Vulnerability index, calendar reconstruction, hourly headways, social burden |
| 2 | `optimise.py` | Phase 5, §3.6, §5.5.2 | Budget-neutral stop-level reallocation for the named scenarios |
| 3 | `sweep.py` | §5.5.3 | Dense sweep of donor-reduction rates; reallocation ceiling |
| 4 | `qgis_layers.py` | §5.2.1 | Map layers, hourly headway profile, holiday-versus-Sunday deterioration |
| 5 | `hour_diff.py` | §5.2.1 | Per-stop headway difference between two hours |
| 6 | `phase1_corridors.py` | §5.2 | Regular urban network, corridors of exclusion, 30-minute service floor |
| 7 | `phase3_optimisation_map.py` | Phase 5, §5.5 | Before/after reallocation map and roles |
| 8 | `phase2_holiday_disaggregation.py` | Phase 3, §3.4, §5.3 | Per-holiday comparison across the four seasonal feeds |
| 9 | `phase4_essential_services.py` | Phase 4, §3.5, §5.4 | Essential-service access, double exclusion, priority set |
| 10 | `sensitivity_dynamic_burden.py` | §5.5.4 | Static versus dynamic burden weights |
| 11 | `sensitivity_suite.py` | §5.5.4 | Weight simplex (36 vectors), service-floor thresholds, tertile cuts, catchments, bootstrap |
| 12 | `robustness_checks.py` | §5.4, §5.5.3, §6.3.2 | Ten consolidated diagnostics; `--list` shows them, `--skip-osm` and `--skip-atm` skip the live-OSM and ATM checks |

Operator-data extensions (§5.6; need the private ATM workbook):

| Script | Thesis | What it does |
|---|---|---|
| `demand_validation.py` | §5.6.1 | Metro gate entries versus scheduled supply retention (consistency check) |
| `depot_feasibility.py`, `depot_feasibility_map.py` | §5.6.2 | Post-analysis depot compatibility screen and its stop-level map |
| `depot_constrained_optimise.py` | §5.6.3 | Reallocation with mode and depot constraints (strict, allowance, mode-only) |
| `depot_line_transfer_list.py` | §5.6.3 | Strict depot-constrained result translated from stops to lines |

`python test_depot_constrained.py` runs 13 checks of the constrained allocator
on synthetic data; it needs no input files.

All parameters are in `config.py`, tagged `[FRAMEWORK]` (general design choice)
or `[INSTANTIATION]` (Milan-specific value). The pipeline scripts write a
`run_parameters_<script>.json` record of the full configuration next to their
outputs in `outputs/`.

### Phase numbering

The thesis has five phases: (1) vulnerability, (2) scheduled supply,
(3) per-holiday comparison, (4) essential access and double exclusion,
(5) reallocation. Some file names predate that numbering and were kept so that
thesis-era run records stay valid: `phase1_corridors.py` reports §5.2 results
of Phases 1–2, `phase2_holiday_disaggregation.py` is thesis Phase 3, and
`phase3_optimisation_map.py` maps the Phase 5 reallocation. Console messages
follow the same mixed naming.

## Known issues and corrections

**Zero-population census sections (kept for reproducibility).** Milan's 6,055
census sections in the ISTAT table include sections with no resident
population. `build_vulnerability()` sets their undefined shares to 0 instead of
excluding them, so they carry a vulnerability index computed from zero shares,
and stops inside them take that index rather than the index of a populated
neighbouring section. Excluding these sections before normalisation (`P1 > 0`)
reduces the analytical network from 3,187 to 3,161 stops and changes the
headline optimisation results. This release keeps the submitted behaviour so
that it reproduces the thesis.

**One-direction sensitivity (thesis §6.3.2).** The thesis reports a Moderate
Gini improvement of 18.8% when headways are rescaled to one direction per
route. That figure came from an implementation that rescaled headways and
burdens but kept the original pooled service equivalents in the optimiser.
Recomputing them from the rescaled headways, which is internally consistent,
gives 21.2%. Robustness check 8 reports both. The principal 20.8% result is
unaffected.

**Reporting bases.** The thesis reports the served burden ratio on a stop-hour
basis (1.49) and the penalised ratio at stop level (1.62). On a stop-hour basis
the penalised ratio is 1.63. The builder's validation block prints all three,
labelled. It also distinguishes the stops located in the study area (3,260)
from the analytical network (3,187): the difference is stops with no scheduled
service between 06:00 and 22:00 on a baseline weekday, Sunday or core holiday.

## Methodological notes

- **Scheduled supply only.** Burden is measured from timetabled headways. The
  framework does not measure delivered service, realised waiting times,
  ridership or achieved accessibility.
- **Headway definition.** A stop's day-type headway is the median of its served
  hourly headways, so a shorter service day is not penalised as such; only
  stops with no holiday service at all take the disclosed 120-minute penalty
  (Phase 4 and Phase 5). The per-holiday comparison (thesis Phase 3) instead
  uses the operational window divided by that day's departures.
- **Double exclusion** requires top-quartile holiday burden and failure of the
  constraint-limited access test, i.e. at least one of the three categories
  (pharmacy 400 m, supermarket 400 m, hospital or clinic 800 m) lies outside
  its catchment. It does not mean that all three are out of reach.
- **Model units.** Reallocated service is counted in stop-service equivalents
  within a fixed budget. The depot extension tests structural compatibility; it
  is not a timetable, vehicle schedule or crew plan.
- **Essential access** is geometric proximity to OpenStreetMap features (distance
  to the feature boundary for polygons), not network walking time or verified
  entrances. Robustness check 9 tests the effect of the municipal boundary using
  a buffered, current OpenStreetMap download.

## Applying the framework to another city

Start from `config.py`: GTFS feeds and the primary feed, the holiday list
(including the local patron-saint day), the census comune code, vulnerability
variables and weights, the census zone ID column, the local UTM zone
(`CRS_METRIC`), the operator's night and interurban line codes, and
`OSM_PLACE_NAME` (point `FROZEN_OSM_SERVICES` to a new snapshot or to a path
that does not exist). Then check:

- **Outputs** carry a `milan_` prefix; use a separate `OUTPUT_DIR` per city.
- **Census input** must follow the ISTAT layout: one row per section, with
  numerator and denominator columns for each share.
- **Comune codes** are compared as text: a leading zero (for example `001272`)
  does not match `1272`. The builder stops with an error if no row matches.
- **Holidays** must fall inside a feed's service window, and a holiday that
  falls on a Sunday cannot separate the holiday effect from the Sunday effect.
- **Calendars:** the per-holiday feed diagnostic table reads only
  `calendar_dates.txt`, so feeds built on `calendar.txt` are listed there as
  "not in feed", although the computation itself uses the full calendar.
- **Route types:** metro is identified as GTFS `route_type` 1 and transferable
  service as 0 and 3; extended route types (for example 700 for bus, 900 for
  tram) are not recognised.
- **Blank stop times:** rows with an empty `arrival_time` are dropped, which
  undercounts service in feeds that leave non-timepoint times blank.
- **Comparability:** the vulnerability index is min-max normalised within the
  study area, so index values and burden levels are not comparable between
  cities without a common normalisation.

## Release 1.0 (October 2026)

Changes from the code used for the submitted thesis are limited to
documentation, comments, console labels, input guards and configuration
plumbing:

- `config.py`: added `OSM_PLACE_NAME` (same value previously written in Phase 4
  and robustness check 9); the run record now stores the framework version
  (1.1) instead of a fixed "1.0"; `HEADWAY_METHOD` accepts only the implemented
  "frequency" method.
- `build_stop_table.py`: clearer validation labels, a printed stop-level
  penalised ratio and analytical-network count (also added to the run record),
  and an error when the comune filter matches no rows.
- `depot_feasibility.py`, `demand_validation.py`: the ATM workbook path, sheet
  and column names now come from `config.py` (same values).
- `thesis_code.ipynb`: outputs cleared; the robustness log is written to
  `outputs/`; robustness check 10 is skipped when the ATM workbook is absent.
- Corrected comments on census-section handling (`config.py`,
  `build_stop_table.py`) and the docstrings of `phase3_optimisation_map.py` and
  `phase4_essential_services.py`; removed unused variables and placeholder-free
  f-strings.
- Added `tools/make_data_manifest.py` and `data/MANIFEST.sha256` (input
  checksums).

Before release, the original and edited code were run on the same synthetic
dataset: all 91 output files were identical (byte for byte, or in content for
files that embed timestamps), the notebook ran without errors with and without
the operator workbook, and `test_depot_constrained.py` passes.

## Licence and citation

Code: MIT licence (`LICENSE`). Data keep their own terms: the OpenStreetMap
snapshot is © OpenStreetMap contributors under the ODbL; GTFS and ISTAT data
are subject to their providers' terms; the ATM workbook is private and is not
distributed.

To cite, use `CITATION.cff` or:

> Darvishzade, A. (2026). *Beyond the Weekday: An Open-Data Framework for
> Holiday Transit Equity, with an Application to Milan*. MSc thesis, Politecnico
> di Milano.
