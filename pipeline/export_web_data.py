"""Export the small JSON files the website reads.

The site must not load 78M rows, so the heavy aggregation happens here, once,
and the browser gets a few hundred KB.

Writes into web/data/:
  nyc_zones.geojson  - zone polygons (copied from the cleaned pipeline output)
  zone_stats.json    - per zone: pickups, median fare, median $/mile, mean wait
  summary.json       - headline numbers for the welcome page

Usage: python export_web_data.py
"""
import json
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as cfg

WEB_DATA = cfg.ROOT / "web" / "data"
COLS = ["pu_zone", "base_fare", "price_per_mile", "wait_min", "trip_miles", "hour", "provider"]


def main():
    WEB_DATA.mkdir(parents=True, exist_ok=True)

    sums = {}          # zone -> [trips, fare_sum, miles_sum, wait_sum]
    fare_samples = {}  # zone -> list of sampled fares (medians need values, not sums)
    ppm_samples = {}
    hour_trips = np.zeros(24, dtype=np.int64)
    hour_fare = np.zeros(24)
    hour_wait = np.zeros(24)
    hour_ppm_samples = {h: [] for h in range(24)}
    provider_counts = {"Uber": 0, "Lyft": 0}
    total_trips = 0
    rng = np.random.default_rng(7)

    for key, raw in cfg.TRIP_MONTHS.items():
        path = cfg.cleaned_path(raw)
        pf = pq.ParquetFile(path)
        print(f"  {key}: {pf.metadata.num_rows:,} rows", flush=True)
        for i in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(i, columns=COLS).to_pandas()
            total_trips += len(t)
            hour_trips += np.bincount(t["hour"].to_numpy(), minlength=24)
            hr = t.groupby("hour", observed=True).agg(fare=("base_fare", "sum"), wait=("wait_min", "sum"))
            for h, row in hr.iterrows():
                hour_fare[int(h)] += float(row["fare"])
                hour_wait[int(h)] += float(row["wait"])
            vc = t["provider"].value_counts()
            for name in provider_counts:
                if name in vc.index:
                    provider_counts[name] += int(vc[name])

            g = t.groupby("pu_zone", observed=True).agg(
                trips=("base_fare", "size"), fare=("base_fare", "sum"),
                miles=("trip_miles", "sum"), wait=("wait_min", "sum"))
            for zone, row in g.iterrows():
                acc = sums.setdefault(int(zone), [0, 0.0, 0.0, 0.0])
                acc[0] += int(row["trips"])
                acc[1] += float(row["fare"])
                acc[2] += float(row["miles"])
                acc[3] += float(row["wait"])

            # reservoir-ish sample per zone, enough for a stable median
            s = t.sample(frac=0.02, random_state=int(rng.integers(1e9)))
            for zone, sub in s.groupby("pu_zone", observed=True):
                fare_samples.setdefault(int(zone), []).extend(sub["base_fare"].tolist()[:400])
                ppm_samples.setdefault(int(zone), []).extend(sub["price_per_mile"].tolist()[:400])
            for h, sub in s.groupby("hour", observed=True):
                hour_ppm_samples[int(h)].extend(sub["price_per_mile"].tolist()[:1500])
            del t

    zones = pd.read_csv(cfg.ZONES_CSV)
    stats = {}
    for _, z in zones.iterrows():
        zid = int(z["zone_id"])
        acc = sums.get(zid)
        if not acc or acc[0] == 0:
            continue
        trips, fare_sum, miles_sum, wait_sum = acc
        stats[zid] = {
            "zone": z["zone"],
            "borough": z["borough"],
            "trips": trips,
            "trips_per_hour": round(trips / (len(cfg.TRIP_MONTHS) * 30 * 24), 1),
            "mean_fare": round(fare_sum / trips, 2),
            "mean_miles": round(miles_sum / trips, 2),
            "mean_wait": round(wait_sum / trips, 2),
            "median_fare": round(float(np.median(fare_samples.get(zid, [fare_sum / trips]))), 2),
            "median_ppm": round(float(np.median(ppm_samples.get(zid, [0]))), 2),
        }

    (WEB_DATA / "zone_stats.json").write_text(json.dumps(stats, separators=(",", ":")))

    ppm_by_hour = [round(float(np.median(hour_ppm_samples[h])), 2) if hour_ppm_samples[h] else None
                   for h in range(24)]
    fare_by_hour = [round(hour_fare[h] / hour_trips[h], 2) if hour_trips[h] else None for h in range(24)]
    wait_by_hour = [round(hour_wait[h] / hour_trips[h], 2) if hour_trips[h] else None for h in range(24)]
    cheap_hour = int(np.argmin([p if p is not None else 1e9 for p in ppm_by_hour]))
    dear_hour = int(np.argmax([p if p is not None else -1 for p in ppm_by_hour]))

    ranked = sorted(stats.items(), key=lambda kv: -kv[1]["trips"])
    summary = {
        "total_trips": total_trips,
        "zones": len(stats),
        "months": sorted(cfg.TRIP_MONTHS),
        "providers": provider_counts,
        "provider_share": {k: round(v / total_trips * 100, 1) for k, v in provider_counts.items()},
        "busiest_hour": int(hour_trips.argmax()),
        "quietest_hour": int(hour_trips.argmin()),
        "trips_by_hour": hour_trips.tolist(),
        "ppm_by_hour": ppm_by_hour,
        "fare_by_hour": fare_by_hour,
        "wait_by_hour": wait_by_hour,
        "cheapest_hour": cheap_hour,
        "dearest_hour": dear_hour,
        "cheapest_ppm": ppm_by_hour[cheap_hour],
        "dearest_ppm": ppm_by_hour[dear_hour],
        "top_zones": [{"zone_id": z, "zone": s["zone"], "borough": s["borough"],
                       "trips": s["trips"], "share": round(s["trips"] / total_trips * 100, 2)}
                      for z, s in ranked[:8]],
        "max_zone_trips": ranked[0][1]["trips"],
    }
    (WEB_DATA / "summary.json").write_text(json.dumps(summary, indent=2))

    shutil.copyfile(cfg.ZONES_GEOJSON, WEB_DATA / "nyc_zones.geojson")

    print(f"\ntotal trips {total_trips:,} | zones with trips {len(stats)}")
    print(f"busiest hour {summary['busiest_hour']}:00, quietest {summary['quietest_hour']}:00")
    print("provider share:", summary["provider_share"])
    print("top zones:", [(z["zone"], z["share"]) for z in summary["top_zones"][:5]])
    for f in sorted(WEB_DATA.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
