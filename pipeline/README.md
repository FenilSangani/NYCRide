# Data pipeline

Turns the raw downloads in `DataSets/` into cleaned files the models and the site read.
Every script writes a `*_report.json` next to its output recording what it dropped and why.

## Run

```bash
pip install -r requirements.txt
cd pipeline
python run_all.py              # clean everything
python clean_trips.py 2025-12  # one month
python export_web_data.py      # the JSON the site reads
python train_fare.py           # the fare model + range + distance lookup
python train_demand.py         # the demand model
python build_knowledge.py      # the facts the Ask page answers from
```

Models and their metrics land in `models/`. The API loads them at start-up.

## Sources

| Data | Source | Licence / terms |
|---|---|---|
| Trips (Uber + Lyft) | [NYC TLC High Volume FHV trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), monthly parquet | TLC states the data comes from the companies and it cannot guarantee accuracy; no licence stated on the page |
| Taxi zones | TLC zone shapefile + lookup (same page) | as above |
| Events | [NYC Permitted Event Information – Historical](https://data.cityofnewyork.us/d/bkfu-528j) | licence unspecified on the portal |
| Weather | [Open-Meteo archive API](https://open-meteo.com/) | free, no key; free tier is non-commercial |

## What each script does

**`clean_trips.py`** — reads one parquet row group at a time, so a 20M-row month never
has to fit in memory. Applies the rules below, derives time/trip/money features, writes
`<month> (cleaned).parquet`.

**`clean_events.py`** — parses the permit dates, tidies the text columns, and builds an
hourly borough table. Events have no coordinates, so `borough × hour` is the finest join
the source supports.

**`clean_zones.py`** — converts the shapefile to GeoJSON in lat/lon (a browser cannot draw
a shapefile) and writes a flat CSV of names and centroids.

**`build_weather.py`** — merges the monthly Open-Meteo JSON into one hourly table keyed on
`hour_ts`, which joins to `pickup_hour_ts` in the trip files.

## Trip cleaning rules

Thresholds live in `config.py` and were picked by profiling the data, not guessed.

| Rule | Keep | Why |
|---|---|---|
| `bad_license` | HV0003 (Uber), HV0005 (Lyft) | the project compares these two |
| `unknown_zone` | not 264/265 | "Unknown"/"Outside NYC" have no shape to draw |
| `bad_miles` | 0.1 – 100 mi | zero-distance records and impossible distances |
| `bad_time` | 60 s – 6 h | sub-minute and runaway trips |
| `bad_fare` | $1 – $1000 | negative fares are refunds, not rides |
| `bad_pay` | ≥ 0 | negative driver pay is an adjustment |
| `bad_wait` | 0 – 60 min | negative wait means the timestamps disagree |
| `bad_vtat` | 0 ≤ arrival ≤ pickup | a driver cannot arrive after the pickup |
| `dur_mismatch` | \|dropoff − pickup − trip_time\| ≤ 5 min | the two clocks disagree |
| `bad_ppm` | $1 – $100 /mile | the raw tail reaches $3,951/mile |
| `bad_speed` | 1 – 60 mph | NYC highways cap at 55, so faster is a glitch |
| `wrong_month` | matches the file's month | stray timestamps |

About 6.1–6.5% of rows are dropped per month, most of them unknown zones.

## No scaling here

Values stay in real units. Tree models do not need scaling, the site has to display
`$24.50` rather than `0.37`, and a scaler must be fitted on the training split only —
so normalisation belongs inside the model pipeline, not in the stored data.

## Training

**`train_fare.py`** predicts `base_fare` for one trip. Gradient boosted trees, because
the relationships are not linear — airport trips are near flat-rate, and rush hour bites
differently in each zone. A linear model needs those interactions written out by hand;
trees find them.

Split is **by month**, never random: train on Sep 2025, Dec 2025 and Jan 2026, test on
Apr 2026. A random split lets the model see the future and flatters the score.

Plain baselines are trained first on purpose. A model that cannot beat a straight line
through `trip_miles` is not worth shipping, and without the baselines an R² means nothing.

The range comes from two more models with `loss="quantile"` at the 10th and 90th
percentile — same algorithm, different loss. Coverage is checked on the unseen month:
roughly 80% of real fares should land inside the band.

**Two things the model is never given:** `trip_min`, `tips`, `driver_pay` and
`price_per_mile`. All of them are only known once the ride is over, so they would leak
the answer straight into the prediction.

### Where the mileage comes from

The site only knows two zones, but the model wants miles. `train_fare.py` builds a lookup
of the **median miles actually driven** between every zone pair with at least 20 trips,
and measures a **detour factor** for the rest — the ratio of driven miles to the straight
line between zone centroids, which came out at about 1.38 for NYC. Guessing that number
would have been the easy option; measuring it is the honest one.

**`test_weather.py`** is the experiment that kept weather *out* of the model. Rain does lift
the citywide mean fare 5–6% in September and April, so adding temperature, rainfall and wind
looked obvious. On the same rows and the same split it moved the unseen-month error from
$5.080 to $5.089 and R² from 0.799 to 0.794 — very slightly worse. That lift is already
carried by hour and zone. The columns are not used.

**`train_demand.py`** predicts pickups per zone-hour. It ships two models: one with lag
features (last hours, yesterday, last week) that shows what live data would buy, and one
without, which is what a static site can actually serve. The seasonal average is a strong
baseline here — about R² 0.94 — so the model has to earn its place.

## Known limitations

- **Events are a weak signal.** Street-closure permits (parades, street festivals) only
  appear up to 2016; the 2025–26 months contain Parks Department permits only. The
  all-zero `closures_active` column is dropped rather than fed to a model as noise.
- **Weather is one point for the whole city.** NYC is small, but local showers are missed.
  `visibility_m` comes back empty from the archive and is dropped.
- **Events join on borough, not zone**, because the source has no coordinates.
- **Four months only** (Sep 2025, Dec 2025, Jan 2026, Apr 2026), chosen to cover autumn,
  winter/holidays, cold and rain. Add a month by dropping the parquet into `DataSets/`,
  registering it in `config.TRIP_MONTHS`, and fetching its weather JSON.
- **`tips` is in the cleaned files but must not be a model input** — it is only known
  after the ride.

## The Ask fact base

**`build_knowledge.py`** writes `models/knowledge.json`, the 312 facts the Ask page
answers from. Nothing in it is typed by hand: the model scores, every baseline, the
cleaning thresholds, the rejected experiments and all 261 zones are read out of the
files the rest of this pipeline writes. Retrain and re-run it, and the answers move
with the model instead of quietly going stale.

Two things are worth knowing before editing it:

- **One fact per chunk.** Retrieval scores whole chunks, so a chunk that needs its
  neighbours to make sense will be quoted alone and read as nonsense.
- **`KEYWORDS` exists for vocabulary, not content.** It attaches words a person is
  likely to use to the fact that answers them, indexed but never shown. "Which
  algorithm did you use" matched nothing until `algorithm` was attached to the fact
  about gradient boosting, because the fact itself never used the word. The script
  fails loudly if `KEYWORDS` names a fact that no longer exists.
