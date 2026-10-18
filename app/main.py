"""NYCRide API.

Loads the trained models once at start-up and answers the questions the pages
ask. Every endpoint takes two zone ids and a time; nothing here touches the
78M-row parquet files, which is why a request costs about a millisecond.

Run from the project root:
    .venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 5173
"""
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

# Imported as a package by uvicorn (app.main) but as a plain module by some
# hosts, so the sibling import has to work either way.
try:
    from . import rag
except ImportError:                                   # pragma: no cover
    import rag

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
WEB = ROOT / "web"


def load_dotenv():
    """Read .env for local runs, without a dependency and without overriding.

    A real environment variable always wins, so the host's own settings are
    untouched - .env only fills in what is missing, and it is git-ignored.
    """
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


load_dotenv()

AIRPORT_ZONES = {1, 132, 138}      # Newark, JFK, LaGuardia
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
NEARBY_KM = 2.0                    # how far a rider is assumed to be willing to walk

app = FastAPI(title="NYCRide", docs_url="/api/docs")

# ---------------------------------------------------------------- load once

fare_bundle = joblib.load(MODELS / "fare.joblib")
range_bundle = joblib.load(MODELS / "fare_range.joblib")

FARE_MODEL = fare_bundle["model"]
FARE_LOG = fare_bundle["log_target"]
FEATURES = fare_bundle["features"]
LO_MODEL, HI_MODEL = range_bundle["lo"], range_bundle["hi"]

# Demand is served from a lookup, not a model, and deliberately so: without live
# lag features the trained model scored R2 0.915 against the plain
# zone x hour x weekday average at 0.951. The average wins, so the average ships.
DEMAND_TABLE = json.loads((MODELS / "demand_seasonal.json").read_text())

_pair_file = json.loads((MODELS / "zone_pair_miles.json").read_text())
PAIR_MILES = _pair_file["pairs"]
DETOUR = _pair_file["detour_factor"]

ZONES = pd.read_csv(ROOT / "DataSets" / "Zones" / "nyc_zones (cleaned).csv").set_index("zone_id")
ZONE_STATS = json.loads((WEB / "data" / "zone_stats.json").read_text())
SUMMARY = json.loads((WEB / "data" / "summary.json").read_text())
FARE_METRICS = json.loads((MODELS / "fare_metrics.json").read_text())
DEMAND_METRICS = json.loads((MODELS / "demand_metrics.json").read_text())


# ---------------------------------------------------------------- helpers

def haversine_mi(a, b):
    p = math.pi / 180
    s = (math.sin((b[0] - a[0]) * p / 2) ** 2
         + math.cos(a[0] * p) * math.cos(b[0] * p) * math.sin((b[1] - a[1]) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(s)) * 0.621371


def check_zone(z):
    if z not in ZONES.index:
        raise HTTPException(404, f"zone {z} is not a NYC taxi zone")
    return int(z)


@lru_cache(maxsize=8192)
def trip_miles(pu, do):
    """Miles for a zone pair: what was actually driven if we have enough trips,
    otherwise the straight line stretched by the measured detour factor."""
    if pu == do:
        # within one zone there is no centroid distance to fall back on; use the
        # pair's own median, else a short default
        return PAIR_MILES.get(f"{pu}-{do}", 1.2), "same-zone median"
    hit = PAIR_MILES.get(f"{pu}-{do}")
    if hit is not None:
        return hit, "zone-pair median"
    a = (ZONES.loc[pu, "lat"], ZONES.loc[pu, "lon"])
    b = (ZONES.loc[do, "lat"], ZONES.loc[do, "lon"])
    return round(haversine_mi(a, b) * DETOUR, 2), "straight line x detour factor"


def make_rows(pu, do, miles, hours, dows, is_uber, shared):
    """One row per (hour, dow) combination, in the column order the model wants.

    The airport flags are built from the zone ids exactly as training built them,
    so the model is never handed a feature it was not taught on.
    """
    rows = []
    for h in hours:
        for d in dows:
            rows.append({
                "pu_zone": pu, "do_zone": do, "trip_miles": miles,
                "hour": h, "dow": d, "is_weekend": int(d >= 5),
                "is_holiday": 0, "is_holiday_eve": 0,
                "is_uber": is_uber, "shared_req": shared,
                "pu_airport": int(pu in AIRPORT_ZONES),
                "do_airport": int(do in AIRPORT_ZONES),
            })
    return pd.DataFrame(rows)[FEATURES]


def predict_fare(X):
    p = FARE_MODEL.predict(X)
    return np.exp(p) if FARE_LOG else p


def zones_within(pu, km):
    """Zones whose centroid is within `km` of this pick-up, nearest measure first.

    Both the nearby-pickup and the scenario endpoints need the same shortlist,
    and a zone with no stats of its own is skipped: it would price off nothing.
    """
    here = (ZONES.loc[pu, "lat"], ZONES.loc[pu, "lon"])
    out = []
    for z, row in ZONES.iterrows():
        if z == pu or str(z) not in ZONE_STATS:
            continue
        away = haversine_mi((row["lat"], row["lon"]), here) / 0.621371
        if away <= km:
            out.append((int(z), row["zone"], row["borough"], away))
    return out


def zone_card(z):
    s = ZONE_STATS.get(str(z))
    row = ZONES.loc[z]
    return {
        "zone_id": z, "zone": row["zone"], "borough": row["borough"],
        "lat": float(row["lat"]), "lon": float(row["lon"]),
        "trips_per_hour": s["trips_per_hour"] if s else None,
        "median_fare": s["median_fare"] if s else None,
        "median_ppm": s["median_ppm"] if s else None,
        "mean_wait": s["mean_wait"] if s else None,
    }


# ---------------------------------------------------------------- endpoints

@app.get("/api/estimate")
def estimate(pu: int, do: int, hour: int = Query(17, ge=0, le=23),
             dow: int = Query(4, ge=0, le=6), shared: int = Query(0, ge=0, le=1)):
    """A fare for one trip, with the range around it and both providers."""
    pu, do = check_zone(pu), check_zone(do)
    miles, source = trip_miles(pu, do)

    uber_row = make_rows(pu, do, miles, [hour], [dow], 1, shared)
    lyft_row = make_rows(pu, do, miles, [hour], [dow], 0, shared)
    out = {"uber": round(float(predict_fare(uber_row)[0]), 2),
           "lyft": round(float(predict_fare(lyft_row)[0]), 2)}

    # the band comes from its own pair of quantile models, so it can land either
    # side of the estimate; widen it rather than show a band that excludes it
    lo = float(LO_MODEL.predict(uber_row)[0])
    hi = float(HI_MODEL.predict(uber_row)[0])
    mid = out["uber"]

    return {
        "pickup": zone_card(pu), "dropoff": zone_card(do),
        "hour": hour, "dow": dow, "day": DAYS[dow], "shared": shared,
        "miles": miles, "miles_source": source,
        "fare": mid,
        "low": round(min(lo, mid), 2), "high": round(max(hi, mid), 2),
        "per_mile": round(mid / miles, 2) if miles else None,
        "uber": out["uber"], "lyft": out["lyft"],
        "cheaper": "lyft" if out["lyft"] < out["uber"] else "uber",
    }


@app.get("/api/hourly")
def hourly(pu: int, do: int, dow: int = Query(4, ge=0, le=6), shared: int = Query(0, ge=0, le=1)):
    """The same trip priced at all 24 hours, so the page can say when to leave."""
    pu, do = check_zone(pu), check_zone(do)
    miles, source = trip_miles(pu, do)
    X = make_rows(pu, do, miles, range(24), [dow], 1, shared)
    fares = predict_fare(X)
    best, worst = int(np.argmin(fares)), int(np.argmax(fares))
    return {
        "miles": miles, "miles_source": source, "dow": dow, "day": DAYS[dow],
        "hours": [{"hour": h, "fare": round(float(f), 2),
                   "per_mile": round(float(f) / miles, 2) if miles else None}
                  for h, f in enumerate(fares)],
        "cheapest_hour": best, "dearest_hour": worst,
        "cheapest_fare": round(float(fares[best]), 2),
        "dearest_fare": round(float(fares[worst]), 2),
        "spread_pct": round(float(fares[worst] / fares[best] - 1) * 100, 1),
    }


@app.get("/api/weekly")
def weekly(pu: int, do: int, hour: int = Query(17, ge=0, le=23), shared: int = Query(0, ge=0, le=1)):
    """The same trip priced on each day of the week."""
    pu, do = check_zone(pu), check_zone(do)
    miles, _ = trip_miles(pu, do)
    X = make_rows(pu, do, miles, [hour], range(7), 1, shared)
    fares = predict_fare(X)
    best = int(np.argmin(fares))
    return {
        "miles": miles, "hour": hour,
        "days": [{"dow": d, "day": DAYS[d], "fare": round(float(f), 2)}
                 for d, f in enumerate(fares)],
        "cheapest_dow": best, "cheapest_day": DAYS[best],
    }


@app.get("/api/nearby")
def nearby(pu: int, do: int, hour: int = Query(17, ge=0, le=23),
           dow: int = Query(4, ge=0, le=6), shared: int = Query(0, ge=0, le=1),
           radius_km: float = Query(2.0, gt=0, le=5)):
    """Would starting from a neighbouring zone be cheaper? Priced, not guessed.

    Takes `shared` so the comparison is against the same kind of ride the rider
    actually asked for - pricing the alternatives solo while the headline is a
    shared fare would make the savings meaningless.
    """
    pu, do = check_zone(pu), check_zone(do)

    base_miles, _ = trip_miles(pu, do)
    base = float(predict_fare(make_rows(pu, do, base_miles, [hour], [dow], 1, shared))[0])

    options = []
    for z, name, borough, km in zones_within(pu, radius_km):
        if z == do:
            continue
        miles, source = trip_miles(z, do)
        fare = float(predict_fare(make_rows(z, do, miles, [hour], [dow], 1, shared))[0])
        options.append({
            "zone_id": z, "zone": name, "borough": borough,
            "walk_m": int(round(km * 1000)),
            "fare": round(fare, 2), "miles": miles, "miles_source": source,
            "saving": round(base - fare, 2),
        })

    options.sort(key=lambda o: -o["saving"])
    return {
        "from": zone_card(pu), "to": zone_card(do),
        "base_fare": round(base, 2), "base_miles": base_miles,
        "radius_km": radius_km,
        "options": options[:6],
        "best_saving": options[0]["saving"] if options else 0,
    }


@app.get("/api/demand")
def demand(hour: int = Query(17, ge=0, le=23), dow: int = Query(4, ge=0, le=6)):
    """Expected pickups per zone for one hour - the map's demand layer."""
    out = {}
    for z, by_dow in DEMAND_TABLE.items():
        row = by_dow.get(str(dow))
        if row:
            out[z] = row[hour]
    return {
        "hour": hour, "dow": dow, "day": DAYS[dow],
        "source": "zone x hour x weekday average from the training months",
        "zones": out,
    }


@app.get("/api/scenarios")
def scenarios(pu: int, do: int, hour: int = Query(17, ge=0, le=23),
              dow: int = Query(4, ge=0, le=6), shared: int = Query(0, ge=0, le=1)):
    """The same trip under a handful of changes, so the saving is one glance.

    Everything is priced by the same model - these are not rules of thumb.
    """
    pu, do = check_zone(pu), check_zone(do)
    miles, _ = trip_miles(pu, do)

    def price(h=None, d=None, sh=None, uber=1, from_zone=None):
        z = from_zone if from_zone is not None else pu
        m = trip_miles(z, do)[0]
        row = make_rows(z, do, m, [hour if h is None else h], [dow if d is None else d],
                        uber, shared if sh is None else sh)
        return float(predict_fare(row)[0])

    baseline = price()

    # cheapest hour and day for this route, found by pricing all of them
    hour_fares = predict_fare(make_rows(pu, do, miles, range(24), [dow], 1, shared))
    best_hour = int(np.argmin(hour_fares))
    day_fares = predict_fare(make_rows(pu, do, miles, [hour], range(7), 1, shared))
    best_dow = int(np.argmin(day_fares))

    # the nearest zone that prices cheaper, if there is one
    best_zone, best_zone_fare = None, baseline
    for z, name, _borough, _km in zones_within(pu, NEARBY_KM):
        if z == do:
            continue
        f = price(from_zone=z)
        if f < best_zone_fare:
            best_zone, best_zone_fare = (z, name), f

    out = [{
        "label": "As you picked it",
        "detail": f"{DAYS[dow]} {hour:02d}:00 · {'shared' if shared else 'solo'} · Uber",
        "fare": round(baseline, 2), "delta": 0.0, "baseline": True,
    }]

    def add(label, detail, fare):
        out.append({"label": label, "detail": detail, "fare": round(fare, 2),
                    "delta": round(fare - baseline, 2), "baseline": False})

    if best_hour != hour:
        add("Leave at the cheapest hour", f"{DAYS[dow]} {best_hour:02d}:00",
            float(hour_fares[best_hour]))
    if best_dow != dow:
        add("Travel on the cheapest day", f"{DAYS[best_dow]} {hour:02d}:00",
            float(day_fares[best_dow]))
    add("Switch to Lyft", f"{DAYS[dow]} {hour:02d}:00", price(uber=0))
    add("Shared instead of solo" if not shared else "Solo instead of shared",
        "same time, same zones", price(sh=1 - shared))
    if best_zone:
        add(f"Start from {best_zone[1]}", f"within {NEARBY_KM:g} km of your pick-up",
            best_zone_fare)

    out[1:] = sorted(out[1:], key=lambda o: o["delta"])
    return {"baseline": round(baseline, 2), "options": out}


ASK = rag.Ask(
    MODELS / "knowledge.json",
    [(int(z), r["zone"], r["borough"]) for z, r in ZONES.iterrows()],
    # the router prices a trip through the same endpoint the pages use, so an
    # answer can never quote a number the site would not show
    lambda pu, do, hour, dow: estimate(pu, do, hour, dow, 0),
)


@app.get("/api/ask")
def ask(q: str = Query(..., min_length=2, max_length=500)):
    """Answer a question about the project, its data or a particular trip."""
    return ASK.answer(q)


@app.get("/api/meta")
def meta():
    """Everything the Model Lab section prints."""
    return {
        "fare": FARE_METRICS,
        "demand": DEMAND_METRICS,
        "data": {
            "total_trips": SUMMARY["total_trips"],
            "zones": SUMMARY["zones"],
            "months": SUMMARY["months"],
            "provider_share": SUMMARY["provider_share"],
        },
        "distance": {"zone_pairs": len(PAIR_MILES), "detour_factor": DETOUR},
        "knowledge_facts": len(ASK.facts),
    }


# The pages are plain files; serving them from the same origin means fetch()
# needs no CORS and no second host.
#
# There is deliberately no cache-control middleware here. One used to send
# no-cache on every page response, which solved a local problem - an edited
# stylesheet kept being served from the browser cache - but it is the wrong
# trade in production, and on a serverless host any top-level middleware stops
# static files being promoted to the CDN, so every stylesheet request would wake
# a Python function. The ?v=N query on the asset tags is what busts the cache;
# bump it when editing CSS or JS.
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
