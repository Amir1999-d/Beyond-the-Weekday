"""
================================================================================
config.py  —  Single source of truth for the Holiday Transit Equity framework
================================================================================
An Open-Data Framework for Holiday Transit Equity, with an Application to Milan.
Politecnico di Milano, 2026.

This module is the EXECUTABLE form of the framework's parameter contract
(methodology chapter, Table 3.1). Every analysis script imports its framework
and instantiation parameters from here; only diagnostic grids (the sweep range
and the sensitivity grids) are set in their own scripts. To instantiate the
open-data core for a different city, an analyst starts from this file; README.md
lists the remaining Milan-specific elements (output file prefixes, the ISTAT
input layout and the operator-data extensions).

Each parameter is annotated as one of:
    [FRAMEWORK]     general design choice, literature-anchored, transferable as-is
    [INSTANTIATION] Milan/ATM/ISTAT-specific value, must be re-derived per city

The split mirrors the thesis structure: [FRAMEWORK] parameters are described
in Chapter 3 (the reusable method); [INSTANTIATION] values are justified in
Chapter 4 (the Milan case study).

Author: Amirhassan Darvishzade  ·  Framework version: 1.1
================================================================================
"""

import os
import json
import datetime

FRAMEWORK_VERSION = "1.1"

# =============================================================================
# 1. PATHS  [INSTANTIATION]
# -----------------------------------------------------------------------------
# Paths are relative to this file: input data go in ./data (not part of the
# repository) and every derived file is written to ./outputs.
# GTFS feeds are kept in separate sibling folders so each can cover a different
# part of the calendar year (the framework supports multiple seasonal feeds).
# =============================================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(PROJECT_DIR, "data")

# GTFS feeds — one folder per seasonal window. The framework reads whichever
# feeds are present; a holiday is only analysable if SOME feed covers its date.
GTFS_FEEDS = {
    "spring":     os.path.join(BASE_PATH, "gtfs"),            # primary feed (28 Mar – 2 May 2026)
    "ferragosto": os.path.join(BASE_PATH, "gtfs_ferragosto"), # August 2025 window
    "december":   os.path.join(BASE_PATH, "gtfs_december"),   # December 2025 window (Christmas/Santo Stefano)
    "june":       os.path.join(BASE_PATH, "gtfs_june"),       # 4 May–12 Jun 2026; supplies Republic Day
}

# The "primary" feed is the one used to build the core weekday/holiday baseline
# (Phases 1–2). Seasonal feeds are used in Phase 3 disaggregation for holidays
# the primary feed's date window does not cover.
PRIMARY_FEED = "spring"

# Census / demographic inputs  [INSTANTIATION]
ISTAT_XLSX = os.path.join(BASE_PATH, "R03_DatiCPA_2021.csv.xlsx")
ZONES_GEOJSON = os.path.join(BASE_PATH, "sezioni_censimento_2021.geojson")
# Restrict ISTAT to the study-area comune BEFORE normalising. [INSTANTIATION]
ISTAT_COMUNE_COLUMN = "PROCOM"      # comune code column in the ISTAT table
ISTAT_COMUNE_CODE   = "15146"        # Milan. None = use all rows (no filter).
# Output directory for all derived files. Kept distinct so a re-run can be
# date-stamped and frozen without overwriting a previous run.
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")  # or a dated subfolder to freeze a run, e.g. os.path.join(PROJECT_DIR, "outputs", "run_2026-05-27")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional frozen OSM service snapshot used for exact Phase 4 reproduction.
# If absent, Phase 4 falls back to a live OSM query and warns that counts may drift.
FROZEN_OSM_SERVICES = os.path.join(BASE_PATH, "osm_services_2026-07-05.gpkg")


# =============================================================================
# 2. VULNERABILITY INDEX (PTD)  —  Phase 1
# -----------------------------------------------------------------------------
# The composite Public Transport Dependency index is a weighted sum of
# normalised population SHARES (ratios), NOT raw counts. Ratios make zones of
# different population sizes comparable. [FRAMEWORK]
#
# The specific dimensions and weights are Milan's instantiation. A different
# city re-derives both the variable list and the weights to reflect its own
# captive-rider structure. [INSTANTIATION]
# =============================================================================

# Mapping of each vulnerability dimension to its ISTAT 2021 source columns and
# the denominator that turns it into a share.  [INSTANTIATION]
#   numerator columns are summed; result divided by the denominator column.
VULNERABILITY_VARIABLES = {
    "pct_elderly": {"numerator": ["P27", "P28", "P29"], "denominator": "P1"},   # age >= 65 / total pop
    "pct_foreign": {"numerator": ["ST1"],               "denominator": "P1"},   # foreign residents / total pop
    "pct_single":  {"numerator": ["PF3"],               "denominator": "PF1"},  # single-person HH / total HH
}

# Weights per dimension. MUST sum to 1.0. [INSTANTIATION]
# Basis: elderly weighted highest as the most service-frequency-sensitive captive
# cohort (Mouwen 2015 [6]); foreign residents and single-person households equal,
# as car-access and intra-household-sharing constraints (Cebollada 2006 [8];
# Titheridge 2014 [9]). Weights were SYNTHESISED, not copied from a single source;
# an equal-weight (1/3 each) sensitivity check is run and reported.
VULNERABILITY_WEIGHTS = {
    "pct_elderly": 0.40,
    "pct_foreign": 0.30,
    "pct_single":  0.30,
}

# Minimum population for a section's v_index to be treated as statistically
# reliable. Small denominators make the ratio-based shares unstable, and
# low-population sections are disproportionately likely to land in the top
# v_index quantiles as an artefact of that instability (quantified per run
# by the builder diagnostic; see run_parameters_builder.json of the frozen
# run rather than any number embedded here). Flagged, not excluded, since
# some low-population sections reflect genuine small residential pockets
# (e.g. care facilities) rather than pure artefact. [FRAMEWORK]
MIN_POPULATION_FOR_VINDEX = 20
FLAG_LOW_POPULATION_SECTIONS = True
POPULATION_COLUMN = "P1"

# Run the equal-weight sensitivity variant alongside the main index? [FRAMEWORK]
RUN_EQUAL_WEIGHT_SENSITIVITY = True

# Census-zone ID column name in both the ISTAT table and the GeoJSON.
# IDs are normalised by stripping any trailing ".0" decimal artefact. [INSTANTIATION]
ZONE_ID_COLUMN = "SEZ21_ID"


# =============================================================================
# 3. TEMPORAL / SUPPLY ANALYSIS  —  Phase 2
# -----------------------------------------------------------------------------
# Day-type definitions, the operational window, and the headway method.
# =============================================================================

# Operational window [start, end): 06:00 <= time < 22:00. [FRAMEWORK]
# Rationale: 06:00–22:00 captures the span over which a service standard is
# meaningfully expected; night service is structurally different and excluded.
OPERATIONAL_HOUR_START = 6
OPERATIONAL_HOUR_END = 22
OPERATIONAL_HOURS = OPERATIONAL_HOUR_END - OPERATIONAL_HOUR_START  # 16
OPERATIONAL_MINUTES = OPERATIONAL_HOURS * 60                       # 960


def operational_hour_filter(hour_series):
    """
    Boolean mask for hourly bins inside the operational window.

    The window is HALF-OPEN: [OPERATIONAL_HOUR_START, OPERATIONAL_HOUR_END),
    i.e. bins 6..21, whose 60-minute spans exactly tile 06:00-22:00 = 960 min.
    An inclusive filter (6..22) would admit a 17th bin (22:00-22:59) and make
    every headway derived as OPERATIONAL_MINUTES / trips inconsistent with the
    trip count it divides. Every script MUST use this helper (or the identical
    half-open comparison) so the window is defined in exactly one place.
    """
    return (hour_series >= OPERATIONAL_HOUR_START) & \
           (hour_series < OPERATIONAL_HOUR_END)

# Weekday baseline = Tue/Wed/Thu only. [FRAMEWORK]
# Rationale: Monday and Friday carry bridge-day / early-closure risk that
# contaminates a "typical working day" baseline; midweek is the clean baseline.
WEEKDAY_DAYS = ["Tuesday", "Wednesday", "Thursday"]

# Whether Sunday is carried as a distinct day-type. [FRAMEWORK decision]
# If True, the builder computes and propagates hw_sun so Phase 1's service-floor
# table has a REAL Sunday row (not a holiday proxy). If False, the Sunday row is
# omitted entirely rather than proxied. We keep it True: Sunday is the natural
# comparator that isolates the *holiday-specific* penalty from the weekend penalty.
INCLUDE_SUNDAY = True

# Headway computation method. [FRAMEWORK]
# The implemented method is hourly frequency-derived headway:
#   headway = 60 / (scheduled stop events in the hour / relevant service days).
# Stop-level day-type headway is then the median of served hourly headways.
HEADWAY_METHOD = "frequency"

# No-service handling. [FRAMEWORK]
# Canonical internal state for an unserved stop-hour is NaN (no imputation).
# Two reported quantities are derived from it:
#   - served-only median headway  (the honest "when served, how long" figure)
#   - coverage statistic          (% of stop-hours / stops with NO service)
# NO_SERVICE_PENALTY_MIN is applied only where a continuous stop-level value is
# required, notably the Phase 4 burden classification and Phase 5 reallocation;
# its use is always disclosed.
NO_SERVICE_PENALTY_MIN = 120
# Report the coverage statistic as a first-class finding (not just a footnote)?
REPORT_COVERAGE_SEPARATELY = True


# =============================================================================
# 4. SERVICE FLOOR  —  Phase 1 / Phase 2
# -----------------------------------------------------------------------------
# Threshold above which a stop "fails" the minimum acceptable service standard.
# =============================================================================

# [FRAMEWORK] 30-minute descriptive floor. It is deliberately more permissive
# than the range normally associated with spontaneous/walk-up use and therefore
# flags clear low-frequency cases rather than a regulatory minimum.
SERVICE_FLOOR_MIN = 30

# [FRAMEWORK] Vertical-equity tertile threshold: stops whose v_index falls in
# the top tertile are reported separately from the full set so that disparate
# treatment is explicit (Litman; Pereira et al.). 0.67 is the empirical-tertile
# convention; transferable as-is.
HIGH_VULN_TERTILE = 0.67

# [FRAMEWORK] Regular-network scope filter (Phase 1).
# The corridors-of-exclusion concept measures *holiday reduction from a usable
# weekday baseline*. Stops without a regular weekday baseline service belong to
# a different category — night lines, interurban coaches, special-event
# services — and analysing them in the same frame conflates distinct service
# types (Mouratidis on disaggregation [27]). A stop qualifies as part of the
# regular urban network when its median weekday headway is at most
# REGULAR_NETWORK_HW_MAX_MIN minutes AND its route set contains at least one
# non-excluded route.
REGULAR_NETWORK_HW_MAX_MIN = 30

# [INSTANTIATION] Route-name prefixes/exact-matches to exclude as non-urban.
# Apply to GTFS route_short_name. Tune per operator.
#   "N"  -> ATM night lines (NM1, NM2, N15, ...)
# Lines 172 and 174 are excluded from the regular urban network on
# service-pattern grounds (thesis §5.2). This is the thesis's own network
# definition, independent of ATM's contractual "ambito" in the depot register.
NON_URBAN_ROUTE_PREFIXES = ("N",)
NON_URBAN_ROUTE_EXACT = ("172", "174")


# =============================================================================
# 5. HOLIDAYS  —  Phase 2 / Phase 3
# -----------------------------------------------------------------------------
# The civil public-holiday set. The METHOD ("analyse each civil holiday in the
# feed window") is [FRAMEWORK]; the specific Italian list and which feed covers
# each date is [INSTANTIATION].
#
# Each holiday: label -> {date: YYYYMMDD, feed: which GTFS_FEEDS key covers it}.
# A holiday whose feed does not actually contain its date is reported as
# "not in feed" by the Phase 2 diagnostic and skipped.
# =============================================================================

HOLIDAYS = {
    # Spring feed (primary)
    "Easter Sunday (5 Apr)":   {"date": "20260405", "feed": "spring"},
    "Easter Monday (6 Apr)":   {"date": "20260406", "feed": "spring"},
    "Liberation Day (25 Apr)": {"date": "20260425", "feed": "spring"},
    "Labour Day (1 May)":      {"date": "20260501", "feed": "spring"},
    "Republic Day (2 Jun)":    {"date": "20260602", "feed": "june"},
    # Summer feed
    "Ferragosto (15 Aug)":     {"date": "20250815", "feed": "ferragosto"},
    # December feed
    "Christmas (25 Dec)":      {"date": "20251225", "feed": "december"},
    "Santo Stefano (26 Dec)":  {"date": "20251226", "feed": "december"},
}

# The set of holiday dates used to build the CORE holiday baseline (Phases 1–2).
# Only holidays present in the PRIMARY feed's window belong here — they share one
# feed and one denominator. Derived automatically below; do not edit by hand.
CORE_HOLIDAY_DATES = [
    h["date"] for h in HOLIDAYS.values() if h["feed"] == PRIMARY_FEED
]

# ALL civil-holiday dates, regardless of which feed is declared to cover them.
# Used as the EXCLUSION set when classifying weekday/Sunday dates: a holiday
# must never be counted as a regular weekday or Sunday even if it happens to
# fall inside the primary feed's window while being assigned to another feed.
# The rule is general: in the Milan instantiation the spring window ends on
# 2 May 2026, so Republic Day (2 Jun) falls outside it and is analysed from the
# june feed. Derived automatically; do not edit by hand.
ALL_HOLIDAY_DATES = [h["date"] for h in HOLIDAYS.values()]

# Holidays EXCLUDED from the cross-holiday "which holiday cuts deepest"
# comparison, with the documented reason. [INSTANTIATION]
# They are still COMPUTED and shown, but flagged and dropped from the ranking,
# so the exclusion is transparent rather than silent.
#   Labour Day: the archive's coverage collapses on this date (service is
#               enumerated at only a small fraction of stops), and 1 May 2026
#               falls on a Friday, the bridge-day review flag. Both grounds are
#               documented in the thesis (§5.3). (Methodological decision #3.)
# Santo Stefano (26 Dec 2025) also falls on a Friday: reviewed in Phase 2 and
# cleared (see CROSS_HOLIDAY_REVIEW below).
CROSS_HOLIDAY_EXCLUSIONS = {
    "Labour Day (1 May)": "Coverage collapse in the archive on this date, with the Friday bridge-day flag — not a representative festivo pattern.",
}

# [INSTANTIATION] R1 — whether a date excluded from the Phase 3 per-holiday
# comparison is ALSO removed from the Phase 2 aggregate holiday day type.
# Holiday headways are computed as summed operations / number of holiday dates,
# so a date on which most stops are unserved lengthens every unserved stop's
# aggregate holiday headway by n/(n-1). Set True to keep Phases 2 and 3
# consistent; False reproduces the original behaviour.
EXCLUDE_CROSS_HOLIDAY_FROM_CORE = True

if EXCLUDE_CROSS_HOLIDAY_FROM_CORE:
    CORE_HOLIDAY_DATES = [
        h["date"] for name, h in HOLIDAYS.items()
        if h["feed"] == PRIMARY_FEED and name not in CROSS_HOLIDAY_EXCLUSIONS
    ]
# Holidays to inspect for a possible Friday/operational caveat before trusting
# them in the uniformity comparison (computed, but reviewed manually).
# Review outcomes belong in the Phase 2 run record (run_parameters_phase2.json)
# and the thesis text, not in this file: once a holiday is cleared, remove it
# from this list; if a caveat is confirmed, promote it to
# CROSS_HOLIDAY_EXCLUSIONS with a documented reason.
# Santo Stefano (26 Dec 2025, a Friday) was reviewed in Phase 2 and cleared:
# its profile matches the regular holiday template, so no bridge-day caveat
# applies (numeric evidence lives in run_parameters_phase2.json, not here).
# Ferragosto (15 Aug 2025) ALSO falls on a Friday and has not yet received the
# same review; it is listed until the Phase 2 profile is inspected, so the
# Friday rule is applied consistently rather than only to Labour Day.
# Ferragosto cleared 2026-07-10: 16.0 min / 0.101 / 3.9% — within the
# six-holiday band (15.2–16.0 min, 3.0–3.9%); see run_parameters_phase2.json
CROSS_HOLIDAY_REVIEW = []

# =============================================================================
# 6. OPTIMISATION ("Robin Hood" reallocation)  —  Phase 5
# -----------------------------------------------------------------------------
# Stop-level reallocation of modelled service equivalents from low-vulnerability,
# high-frequency DONOR stops to high-vulnerability, low-frequency RECIPIENT
# stops, weighted by current burden and limited by a recipient headway cap.
# =============================================================================

# Donor/recipient classification cut-points (quantiles of v_index and of
# holiday headway). [FRAMEWORK] — the tertile design is general; the 0.33/0.67
# values are the instantiation and are sensitivity-testable.
ROLE_VULN_LOW_Q = 0.33   # donors:    v_index <= this quantile (low vulnerability)
ROLE_VULN_HIGH_Q = 0.67  # recipients: v_index >= this quantile (high vulnerability)
ROLE_HW_LOW_Q = 0.33     # donors:    hw_hol  <= this quantile (already high frequency)
ROLE_HW_HIGH_Q = 0.67    # recipients: hw_hol  >= this quantile (poorly served)

# Reduction scenarios: fraction of donor trips removed and reallocated. [INSTANTIATION]
# The graduated set (conservative/moderate/ambitious) is [FRAMEWORK]; the specific
# rates are the instantiation. "Moderate" is the recommended scenario.
SCENARIOS = {
    "Conservative": 0.10,
    "Moderate":     0.20,
    "Ambitious":    0.35,
    "Aggressive":   0.50,
    "Extreme":      0.65,
}
RECOMMENDED_SCENARIO = "Moderate"

# Stop-level recipient headway cap (minutes). [INSTANTIATION]
# Informed by the shortest observed holiday headways on high-frequency surface
# services. It is a modelling cap, not a line-level operational constraint.
# As the scenario rate rises, more recipients reach the cap and marginal equity
# gain diminishes.
MIN_HEADWAY_FLOOR_MIN = 8

# Overflow rule once a recipient hits the floor. [FRAMEWORK]
#   "redistribute_then_cap": cap the saturated recipient at the floor and pass
#       its residual freed-trips to the next-priority unsaturated recipient;
#       once ALL recipients are saturated, any remaining freed trips are
#       UNALLOCATED — they leave the system, so the network loses service and
#       the scenario is no longer strictly budget-neutral. The optimiser
#       reports this explicitly (unallocated_trips, budget_neutral flag).
#   "return_residual_to_donors": as above, but any freed trips that no
#       recipient can absorb are returned to donors pro-rata, so total trips
#       are conserved EXACTLY at every rate. This isolates the pure
#       saturation mechanism from the trip-destruction artefact and is the
#       recommended robustness variant for the ceiling-curve analysis: past
#       the saturation onset the two rules diverge, and only the residual-
#       return curve attributes the Gini bend to saturation alone.
OVERFLOW_RULE = "redistribute_then_cap"

# Strict budget neutrality: sum of donor trips freed must equal sum of recipient
# trips gained (within tolerance). Asserted at runtime. [FRAMEWORK]
BUDGET_NEUTRALITY_TOLERANCE = 1.0  # modelled stop-service equivalents/day


# -----------------------------------------------------------------------------
# 6b. DEPOT- AND MODE-CONSTRAINED REALLOCATION  —  operator-data extension (thesis §5.6.3)
# -----------------------------------------------------------------------------
# Read ONLY by depot_constrained_optimise.py. No frozen pipeline script reads
# these parameters, so adding them changes no figure reported in the thesis.
# The add-on moves the depot screen of §5.6.2 from an ex-post check into the
# optimisation itself: only surface service can be freed, freed service is
# pooled by depot, and a recipient draws only from its own lines' depots.
# =============================================================================

# ATM line-to-depot register (July 2026), §4.2.4. Operator-provided; not
# redistributed. [INSTANTIATION]
# Structure verified against the register (September 2026).
# The file name keeps the operator's spelling ("Depostio"): do not correct it,
# or it will no longer match the file on disk.
DEPOT_REGISTRY = os.path.join(BASE_PATH, "Tornellati-LineeDepostio_-_Tesista.xlsx")
DEPOT_REGISTRY_SHEET = "Linee-Deposito"   # the workbook's other sheet, "Tornellati", holds the entrances
DEPOT_REGISTRY_LINE_COL = "linea_carto"   # line code, matches GTFS route_short_name
DEPOT_REGISTRY_DEPOT_COL = "nomedepo"     # depot name
DEPOT_REGISTRY_AMBITO_COL = "ambito"      # Urbano / Area Urbana

# Service that can be moved between surface stops, as GTFS route_type values
# (read as strings): 0 = tram, 3 = bus. Metro (1) and rail (2) capacity cannot
# be converted into surface service (§5.6.2), so it is never surrendered. [FRAMEWORK]
TRANSFERABLE_ROUTE_TYPES = {"0", "3"}

# Operator-mandated exclusions (§4.2.4, §5.6.2): line 161, a temporary
# tram-replacement code, and the BDQ/Q-series Radiobus di Quartiere codes. [INSTANTIATION]
DEPOT_EXCLUDED_LINE_CODES = {"161"}
DEPOT_EXCLUDED_LINE_PREFIXES = ("BDQ", "Q")

# Night and interurban lines (NON_URBAN_ROUTE_PREFIXES / _EXACT above) are left
# out of the regular urban network by the thesis. True applies the same
# definition to the depot extension: those lines can neither give nor receive,
# so no night line (NM1, N15, ...) is assigned daytime holiday service.
# [FRAMEWORK]
DEPOT_EXCLUDE_NON_URBAN_LINES = True

# stop -> line attribution for the whole depot extension. "trips": each line's
# share of a stop's change follows its own departures there. "equal": each line
# gets an equal share, the rule of the post-analysis depot screen (§5.6.2).
# Donors are attributed through their
# holiday lines, recipients through their weekday lines. [FRAMEWORK]
STOP_LINE_ATTRIBUTION = "trips"

# Inter-depot transfer. [FRAMEWORK]
#   "strict"    no service crosses depots; what a depot cannot place is
#               returned to its own donors (exact budget neutrality)
#   "allowance" up to DEPOT_TRANSFER_ALLOWANCE may cross depots
#   "free"      depots unconstrained; only the mode restriction applies
DEPOT_TRANSFER_MODE = "strict"

# Maximum modelled stop-service equivalents per day allowed to cross depots
# under "allowance". A scenario choice for sensitivity, not an ATM figure.
# [INSTANTIATION]
DEPOT_TRANSFER_ALLOWANCE = 2000.0

# Line-level transfer list (depot_line_transfer_list.py): the strict
# depot-constrained result translated from stops to lines, depot by depot.
# Scenario to translate. [FRAMEWORK]
DEPOT_LIST_SCENARIO = RECOMMENDED_SCENARIO

# Attribution override for the line list only. None = follow
# STOP_LINE_ATTRIBUTION, so the list and the constrained results come from the
# same model. Set "equal" only to reproduce the thesis screen's rule. [FRAMEWORK]
DEPOT_LIST_ATTRIBUTION = None


# =============================================================================
# 7. ESSENTIAL SERVICES & DOUBLE EXCLUSION  —  Phase 4
# -----------------------------------------------------------------------------
# Walking catchments to essential-service categories, and the double-exclusion
# classification. These catchments are 15-Minute-City SERVICE-access thresholds,
# DISTINCT from SDG 11.2.1's transit-access standard (500 m bus / 1 km rail),
# which governs the transit-access dimension elsewhere. Do not conflate.
# =============================================================================

# OSM amenity/shop tags per essential-service category. [INSTANTIATION]
ESSENTIAL_SERVICE_TAGS = {
    "hospital":    {"amenity": ["hospital", "clinic"]},
    "pharmacy":    {"amenity": "pharmacy"},
    "supermarket": {"shop": ["supermarket", "convenience"]},
}

# OpenStreetMap place name. Used by Phase 4 ONLY when the frozen service
# snapshot (FROZEN_OSM_SERVICES) is absent, and by the optional boundary-buffer
# robustness check. [INSTANTIATION]
OSM_PLACE_NAME = "Milan, Metropolitan City of Milan, Italy"

# Per-category walking catchment radius (metres). [INSTANTIATION, literature-bounded]
# Basis: 15-Minute-City proximity literature uses a spread of 300–900 m for
# daily-need destinations, with shorter thresholds for the most frequent needs.
#   400 m (~7–10 min walk) for daily-need pharmacy/supermarket
#   800 m (~15 min walk)    for the lower-frequency hospital/clinic
# NOTE: keep these identical to the values stated in Literature Review §2.8.
SERVICE_CATCHMENTS_M = {
    "pharmacy":    400,
    "supermarket": 400,
    "hospital":    800,
}
# Catchment used for the "near ANY essential service" composite flag.
CATCHMENT_ANY_M = 400

# SDG 11.2.1 transit-access reference (NOT a service catchment). [FRAMEWORK]
# Documented here for provenance; used only where transit-access is discussed.
SDG_TRANSIT_LOW_CAPACITY_M = 500   # bus / tram
SDG_TRANSIT_HIGH_CAPACITY_M = 1000  # metro / rail

# Double-exclusion classification. [FRAMEWORK]
# A stop is "double excluded" if BOTH hold:
#   (a) its holiday social burden is in the top quartile of the analytical
#       network (cut below), AND
#   (b) it FAILS the constraint-limited essential-access criterion (outside
#       at least one category's catchment).
# The access dimension is deliberately the categorical catchment-fail flag,
# NOT a normalised-distance quantile: normalising distances against the
# observed maximum is outlier-sensitive and non-transferable between cities
# (see Phase 4 methodology block). A former DOUBLE_EXCLUSION_DISTANCE_Q
# parameter reflecting the abandoned distance-quantile variant has been
# removed — it was read by no script and contradicted the implemented
# definition.
DOUBLE_EXCLUSION_BURDEN_Q = 0.75


# =============================================================================
# 7b. REGRESSION ANCHORS  —  VALIDATION ONLY, NEVER USED IN COMPUTATION
# -----------------------------------------------------------------------------
# Optional expected values from a FROZEN reference run, used exclusively by
# the builder's validation block to print "computed vs expected" so a re-run
# on the same inputs can be regression-checked. These values are NEVER read
# by any analytical computation; emptying this dict changes no result.
# For a new city or a fresh run, set it to {} until a run is frozen.
# [INSTANTIATION — values below are the consolidated Milan thesis run]
# =============================================================================

# Regression anchors intentionally EMPTY: the reproducibility reference is
# the frozen run record (run_parameters_builder.json, 2026-07-10T11:57:16),
# not values pinned here. An empty dict changes no computation and suppresses
# the "[frozen-run expectation]" annotations in the builder validation block.
VALIDATION_EXPECTATIONS = {}
# =============================================================================
# 8. SPATIAL REFERENCE SYSTEMS  [FRAMEWORK / local UTM = INSTANTIATION]
# -----------------------------------------------------------------------------
# Geographic CRS for I/O; projected metric CRS for distance computation.
# A different city sets CRS_METRIC to its local UTM zone.
# =============================================================================

CRS_GEOGRAPHIC = "EPSG:4326"   # WGS84 lat/lon — for GTFS, GeoJSON, exports
CRS_METRIC = "EPSG:32632"      # UTM 32N — Milan's metric zone, for distances

# Two-stage stop-to-zone assignment cap. [FRAMEWORK]
# Stops not inside any census section that carries a v_index take the v_index
# of the NEAREST such section, if within this distance. NOTE: sections that
# appear in the ISTAT table with zero resident population DO carry a v_index
# (their undefined shares are set to 0 in build_vulnerability), so stops inside
# them are assigned in stage 1, not snapped. This is the submitted thesis
# behaviour and is kept for reproducibility; see README, "Known issues". The cap
# was chosen from the 2026-07-10 diagnostic: the distance distribution shows
# a sharp plateau (468 stops <= 25 m, 565 <= 100 m, only +43 more by 200 m)
# before the genuinely-external mass (1,444 stops > 500 m, the hinterland
# network). 100 m sits on the plateau shoulder. Without the snap stage, the
# strict within-join silently drops the largest interchange nodes (Duomo,
# Cairoli, Centrale FS, San Siro, ...) — a systematic bias against
# high-importance locations, first detected via the metro demand-validation
# layer. Stops farther than the cap from any section with a v_index lie outside
# the municipality's residential fabric and are excluded; this exclusion
# supersedes the former dissolved-boundary clip, which was built from
# populated sections only and therefore shared the same flaw.
ZONE_SNAP_MAX_DIST_M = 100


# =============================================================================
# 9. RUN-PARAMETER RECORD  [FRAMEWORK — reproducibility]
# -----------------------------------------------------------------------------
# write_run_parameters() stamps the EXACT parameter state into the output folder
# alongside the results, so every figure/table is traceable to the config that
# produced it. This is the concrete evidence for the "reproducible open-data
# framework" contribution. Call it once at the start of each pipeline script.
# =============================================================================

def _config_snapshot():
    """Collect ALL UPPER_CASE config constants into a JSON-serialisable dict.

    Tuples become lists and sets become sorted lists, so no parameter is left
    out of the run record (previously NON_URBAN_ROUTE_PREFIXES and
    NON_URBAN_ROUTE_EXACT, which define the regular urban network, were
    silently dropped).
    """
    import sys
    module = sys.modules[__name__]
    snapshot = {}
    for name in dir(module):
        if name.isupper() and not name.startswith("_"):
            value = getattr(module, name)
            if isinstance(value, (set, frozenset)):
                value = sorted(value)
            elif isinstance(value, tuple):
                value = list(value)
            # Keep only JSON-serialisable primitives/containers
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                snapshot[name] = value
    return snapshot


def write_run_parameters(script_name, extra=None):
    """
    Write a timestamped record of the full config state plus any script-specific
    extras (e.g. derived denominators, row counts) to run_parameters_<script>.json
    in OUTPUT_DIR. Returns the path written.

    Usage at the top of any pipeline script:
        import config
        config.write_run_parameters("builder", extra={"n_weekday_days": n_wd})
    """
    record = {
        "script": script_name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "framework_version": FRAMEWORK_VERSION,
        "config": _config_snapshot(),
    }
    if extra:
        record["run_specific"] = extra

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"run_parameters_{script_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


# =============================================================================
# 10. VALIDATION  —  fail fast on inconsistent config  [FRAMEWORK]
# -----------------------------------------------------------------------------
# validate() checks the internal consistency a re-user could plausibly break.
# Call it once at import-time of any entry script.
# =============================================================================

def validate():
    """Raise ValueError on any internally inconsistent parameter setting."""
    errors = []

    # Weights must sum to 1 and match the variable set
    w_sum = sum(VULNERABILITY_WEIGHTS.values())
    if abs(w_sum - 1.0) > 1e-9:
        errors.append(f"VULNERABILITY_WEIGHTS sum to {w_sum}, must sum to 1.0")
    if set(VULNERABILITY_WEIGHTS) != set(VULNERABILITY_VARIABLES):
        errors.append("VULNERABILITY_WEIGHTS keys must match VULNERABILITY_VARIABLES keys")

    # Operational window sanity
    if OPERATIONAL_HOUR_START >= OPERATIONAL_HOUR_END:
        errors.append("OPERATIONAL_HOUR_START must be < OPERATIONAL_HOUR_END")

    # Headway method must be known
    # Only the frequency-derived method is implemented (build_stop_table.py).
    if HEADWAY_METHOD not in ("frequency",):
        errors.append(f"HEADWAY_METHOD '{HEADWAY_METHOD}' not recognised")

    # Quantile cut-points in (0,1) and low < high
    if not (0 < ROLE_VULN_LOW_Q < ROLE_VULN_HIGH_Q < 1):
        errors.append("ROLE_VULN quantiles must satisfy 0 < low < high < 1")
    if not (0 < ROLE_HW_LOW_Q < ROLE_HW_HIGH_Q < 1):
        errors.append("ROLE_HW quantiles must satisfy 0 < low < high < 1")

    # Scenarios in (0,1)
    for name, rate in SCENARIOS.items():
        if not (0 < rate < 1):
            errors.append(f"Scenario '{name}' rate {rate} must be in (0,1)")
    if RECOMMENDED_SCENARIO not in SCENARIOS:
        errors.append(f"RECOMMENDED_SCENARIO '{RECOMMENDED_SCENARIO}' not in SCENARIOS")

    # Recipient cap must be positive and below the service-failure threshold.
    if MIN_HEADWAY_FLOOR_MIN <= 0:
        errors.append("MIN_HEADWAY_FLOOR_MIN must be positive")
    if MIN_HEADWAY_FLOOR_MIN >= SERVICE_FLOOR_MIN:
        errors.append("MIN_HEADWAY_FLOOR_MIN should be well below SERVICE_FLOOR_MIN")

    if OVERFLOW_RULE not in ("redistribute_then_cap", "return_residual_to_donors"):
        errors.append(f"OVERFLOW_RULE '{OVERFLOW_RULE}' not recognised")

    # Catchment keys must match service tags
    if set(SERVICE_CATCHMENTS_M) != set(ESSENTIAL_SERVICE_TAGS):
        errors.append("SERVICE_CATCHMENTS_M keys must match ESSENTIAL_SERVICE_TAGS keys")

    # Primary feed must exist in the feed dict
    if PRIMARY_FEED not in GTFS_FEEDS:
        errors.append(f"PRIMARY_FEED '{PRIMARY_FEED}' not in GTFS_FEEDS")

    # Cross-holiday exclusions/review must reference real holidays
    for label in list(CROSS_HOLIDAY_EXCLUSIONS) + CROSS_HOLIDAY_REVIEW:
        if label not in HOLIDAYS:
            errors.append(f"Cross-holiday list references unknown holiday '{label}'")

    # Depot extension (§5.6.3): recognised modes and a non-negative allowance
    if STOP_LINE_ATTRIBUTION not in ("equal", "trips"):
        errors.append(f"STOP_LINE_ATTRIBUTION '{STOP_LINE_ATTRIBUTION}' not recognised")
    if DEPOT_TRANSFER_MODE not in ("strict", "allowance", "free"):
        errors.append(f"DEPOT_TRANSFER_MODE '{DEPOT_TRANSFER_MODE}' not recognised")
    if DEPOT_TRANSFER_ALLOWANCE < 0:
        errors.append("DEPOT_TRANSFER_ALLOWANCE must be non-negative")

    if DEPOT_LIST_ATTRIBUTION not in (None, "equal", "trips"):
        errors.append(f"DEPOT_LIST_ATTRIBUTION '{DEPOT_LIST_ATTRIBUTION}' not recognised")
    if DEPOT_LIST_SCENARIO not in SCENARIOS:
        errors.append(f"DEPOT_LIST_SCENARIO '{DEPOT_LIST_SCENARIO}' not in SCENARIOS")
    if not isinstance(DEPOT_EXCLUDE_NON_URBAN_LINES, bool):
        errors.append("DEPOT_EXCLUDE_NON_URBAN_LINES must be True or False")
    if errors:
        raise ValueError("config.validate() failed:\n  - " + "\n  - ".join(errors))
    return True


# Validate immediately on import so misconfiguration fails before any analysis.
if __name__ != "__main__":
    validate()

if __name__ == "__main__":
    # Running `python config.py` prints a human-readable summary and validates.
    validate()
    print("config.validate() passed.\n")
    print("=" * 70)
    print("FRAMEWORK CONFIGURATION SUMMARY")
    print("=" * 70)
    snap = _config_snapshot()
    for k in sorted(snap):
        print(f"{k:32s} = {snap[k]}")
    print("=" * 70)
    print(f"Total parameters: {len(snap)}")