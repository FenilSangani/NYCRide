# NYCRide

What a New York ride costs, and when — a fare predictor built on 78 million real
NYC taxi trips.

Pick two zones and a time, and the site prices the trip, shows the band around that
price, prices all 24 hours and all 7 days so you can see when to travel, compares Uber
against Lyft on that exact route, and shows what the model got right and wrong.

**The point of the project is honesty about the model, not just the prediction.** Every
score on the site is read from the trained model's own metrics file, plain baselines are
shown next to it, and the two experiments that failed are reported alongside the ones
that worked.

## What it does

| Section | What it answers |
|---|---|
| Estimated fare | what this trip costs, with a 10–90 band and both providers |
| Best time to leave | the same trip priced at all 24 hours |
| Pick-ups nearby | would walking to the next zone be cheaper? |
| Across the week | the same trip priced on each weekday |
| What would make it cheaper | one change at a time, ranked by saving |
| Weather | an experiment that was run and then rejected |
| Model lab | scores, baselines, feature importance, and how it was built |

## Results

Trained on September 2025, December 2025 and January 2026. Tested on **April 2026**, a
month no model ever saw.

| Model | MAE | R² |
|---|---|---|
| mean fare | $13.73 | −0.000 |
| linear regression, miles only | $7.03 | 0.671 |
| linear regression, + hour/day/airport | $7.00 | 0.681 |
| gradient boosting | $5.24 | 0.803 |
| **gradient boosting on log(fare)** | **$5.02** | **0.804** |

The baselines are the point of that table. A model that cannot beat a straight line
through distance is not worth shipping, and an R² on its own says nothing about whether
0.80 was hard to reach.

The 10–90 band holds **79.5%** of real fares against a target of 80%, so the uncertainty
the site shows is calibrated rather than decorative.

## Two things that did not work, and are reported anyway

**Weather was dropped.** Rain lifts the citywide mean fare 5–6%, so temperature, rainfall
and wind looked obvious. On the same split they moved the error from $5.080 to $5.089 and
R² from 0.799 to 0.794 — slightly worse. Hour and zone already carry that signal.

**The demand model lost to a lookup table.** With lag features it reaches R² 0.972, but
lags need the last few hours of live counts and this site has no live feed. Without them
it scored 0.915 against 0.951 for a plain zone × hour × weekday average. The average
ships, and the page says so.

## Running it

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m uvicorn app.main:app --port 5173
```

Open <http://localhost:5173>. The trained models and the web data files are in the
repository, so the site runs without the raw data.

To work on the pipeline as well, add its dependencies:

```bash
.venv/Scripts/python.exe -m pip install -r requirements-pipeline.txt
```

## Deploying to Vercel

The whole site runs as one Vercel Function. No configuration is needed to find it:
Vercel looks for a FastAPI instance named `app` at entrypoints like `app/main.py`,
which is exactly where this one lives.

```bash
npm i -g vercel
vercel          # preview deployment
vercel --prod   # production
```

Or connect the GitHub repository in the Vercel dashboard and it deploys on push.

Three things in this repository exist for that deployment:

| File | Why |
|---|---|
| `vercel.json` | raises the function timeout to 30s; the nearby-pickup and scenario endpoints price dozens of zones in one request |
| `.vercelignore` | keeps the 4.5 GB of trip parquet out of the upload when deploying from the CLI |
| `requirements.txt` | runtime dependencies only — `requirements-pipeline.txt` holds the rest |

### What to know before you deploy

**Bundle size.** The function is about **267 MB** unzipped, against Vercel's 500 MB
limit for Python. scipy (113 MB), pandas (63 MB) and scikit-learn (42 MB) are most of
it. That is why pyarrow is not in `requirements.txt`: it is 87 MB on its own and only
the pipeline imports it. If you add a runtime dependency, check what it costs first.

**Versions are pinned on purpose.** `models/fare.joblib` is a pickle written by
scikit-learn 1.9.1. A different version can refuse to load it, or load it and behave
slightly differently. Re-pin `requirements.txt` whenever you retrain.

**Cold starts.** The first request after an idle period loads 7 MB of models, which
takes a second or two. Subsequent requests are about a millisecond, because nothing
at request time touches the parquet files.

**Static files are served by the CDN.** There is no cache-control middleware in
`app/main.py`, and that is deliberate: any top-level middleware stops Vercel promoting
mounted static files to the CDN, which would make every stylesheet request wake a
Python function. Cache busting is the `?v=N` on the asset tags instead — bump it on
both pages when you change CSS or JavaScript.

## Rebuilding from raw data

The trip files are 4.5 GB and are not in the repository. Download the High Volume FHV
monthly parquet files from the
[TLC trip record page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
put them where `pipeline/config.py` expects them, then:

```bash
cd pipeline
python run_all.py          # clean trips, events, zones, weather
python train_fare.py       # fare model + quantile band + distance lookup
python train_demand.py     # demand baselines and the seasonal lookup
python export_web_data.py  # the JSON the pages read
```

## Layout

| Folder | What is in it |
|---|---|
| `pipeline/` | cleaning and training scripts — see [pipeline/README.md](pipeline/README.md) |
| `app/` | the FastAPI service, `main.py` |
| `web/` | the two pages — see [web/README.md](web/README.md) |
| `models/` | trained models, lookups and metrics (checked in, 7 MB) |
| `DataSets/` | raw and cleaned data (not in the repository) |

## Built with

Python, scikit-learn (`HistGradientBoostingRegressor`), pandas, PyArrow, FastAPI,
and plain HTML/CSS/JavaScript with Leaflet for the maps. No frontend framework and no
paid APIs.

## Honest limitations

- **NYC only, four months.** September, December, January and April were chosen to cover
  autumn, winter holidays, cold and rain.
- **The data cannot explain everything.** Traffic, surge multipliers, the exact route and
  driver supply are what really move a fare, and none of them are in a post-trip TLC
  record. About $5 of error is close to the ceiling for these features.
- **No vehicle type, no passenger count, no surge column, no cancellations.**
- **Events are a weak signal.** Street-closure permits stop appearing after 2016, so the
  2025–26 months carry Parks Department permits only.
- **Weather is a single city-wide reading**, so local showers are missed.
