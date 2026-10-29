# Input data

The pipeline reads its inputs from this folder, using the file names set in
`config.py`. Only this README, the checksum manifest and the OpenStreetMap
service snapshot are published; everything else must be obtained from its
source and placed here under the names below.

| Path in `data/` | Content | Source | Published here |
|---|---|---|---|
| `gtfs/` | ATM GTFS feed, service window 28 Mar – 2 May 2026. Primary feed: weekday baseline and the core holidays (Easter Sunday, Easter Monday, Liberation Day; Labour Day is computed but excluded from the ranking) | Comune di Milano open-data portal, dataset *Orari del trasporto pubblico locale nel Comune di Milano in formato GTFS* ([ds929](https://dati.comune.milano.it/dataset/ds929-orari-del-trasporto-pubblico-locale-nel-comune-di-milano-in-formato-gtfs)) | No |
| `gtfs_june/` | ATM GTFS feed, 4 May – 12 Jun 2026 (Republic Day) | as above | No |
| `gtfs_ferragosto/` | ATM GTFS feed, August 2025 window (Ferragosto) | as above | No |
| `gtfs_december/` | ATM GTFS feed, December 2025 window (Christmas, Santo Stefano) | as above | No |
| `R03_DatiCPA_2021.csv.xlsx` | ISTAT 2021 census variables by census section, Lombardy (`R03_DatiCPA_2021.csv`), saved as `.xlsx` | ISTAT, [Basi territoriali e variabili censuarie](https://www.istat.it/notizia/basi-territoriali-e-variabili-censuarie/), 2021 regional census-section data | No |
| `sezioni_censimento_2021.geojson` | 2021 census-section boundaries, Lombardy, converted to GeoJSON | ISTAT, same page, 2021 territorial bases for Lombardy | No |
| `osm_services_2026-07-05.gpkg` | Essential-service snapshot (pharmacies, supermarkets and convenience stores, hospitals and clinics) used for the thesis run | © OpenStreetMap contributors, available under the [Open Database License](https://opendatacommons.org/licenses/odbl/) | Yes |
| `Tornellati-LineeDepostio_-_Tesista.xlsx` | ATM operator workbook: metro gate entries and the line-to-depot register | Provided by ATM for the thesis | **Never** (private) |

Each GTFS feed is unzipped into its own folder. Feeds are replaced by the
operator over time; the [Mobility Database](https://mobilitydatabase.org/feeds/gtfs/mdb-2666)
keeps an archive of the ATM feed, but its history does not reach back to every
window used here.

## Checksums

`MANIFEST.sha256` lists the SHA-256 checksum of every input file used for the
thesis run (the ATM workbook excluded). To verify local copies:

```
python tools/make_data_manifest.py --check
```

If a GTFS or ISTAT file differs, the results will not match the thesis.

## Without the private ATM workbook

The open-data core (thesis §§5.1–5.5) runs without it. The operator-data
extensions of §5.6 (`depot_feasibility.py`, `depot_feasibility_map.py`,
`demand_validation.py`, `depot_constrained_optimise.py`,
`depot_line_transfer_list.py`) and robustness check 10 need it; the notebook
skips them when the file is absent.

## Without the OpenStreetMap snapshot

Phase 4 then downloads current OpenStreetMap data for `config.OSM_PLACE_NAME`.
OpenStreetMap changes continuously, so the double-exclusion counts (415
double-excluded stops, 184 priority stops) can drift.
