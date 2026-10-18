"""Clean the NYC TLC High Volume FHV (Uber + Lyft) monthly trip files.

Reads one row group at a time so a 20M-row month never has to fit in memory.
Values stay in real units - scaling belongs inside the model pipeline, fitted on
the training split only, so the site can still show "$24.50" rather than "0.37".

Usage:
    python clean_trips.py            # every month in config.TRIP_MONTHS
    python clean_trips.py 2025-12    # one month
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config as cfg

READ_COLS = ["hvfhs_license_num", "request_datetime", "on_scene_datetime", "pickup_datetime",
             "dropoff_datetime", "PULocationID", "DOLocationID", "trip_miles", "trip_time",
             "base_passenger_fare", "tolls", "bcf", "sales_tax", "congestion_surcharge",
             "airport_fee", "tips", "driver_pay", "shared_request_flag", "shared_match_flag",
             "wav_request_flag", "cbd_congestion_fee"]

RULES = ["bad_license", "unknown_zone", "bad_miles", "bad_time", "bad_fare", "bad_pay",
         "bad_wait", "bad_vtat", "dur_mismatch", "bad_ppm", "bad_speed", "wrong_month"]


def clean_month(in_path, out_path):
    pf = pq.ParquetFile(in_path)
    drops = dict.fromkeys(RULES, 0)
    rows_in = kept = 0
    writer = None
    month = None
    t0 = time.time()

    for i in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(i, columns=READ_COLS).to_pandas()
        rows_in += len(df)
        if month is None:
            month = df["pickup_datetime"].dt.to_period("M").mode()[0]

        wait = (df["pickup_datetime"] - df["request_datetime"]).dt.total_seconds() / 60
        vtat = (df["on_scene_datetime"] - df["request_datetime"]).dt.total_seconds() / 60
        dur_s = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()
        ppm = df["base_passenger_fare"] / df["trip_miles"].replace(0, np.nan)
        speed = df["trip_miles"] / (df["trip_time"].replace(0, np.nan) / 3600)

        checks = {
            "bad_license": ~df["hvfhs_license_num"].isin(["HV0003", "HV0005"]),
            "unknown_zone": (df["PULocationID"].isin(cfg.UNKNOWN_ZONES)
                             | df["DOLocationID"].isin(cfg.UNKNOWN_ZONES)),
            "bad_miles": ~df["trip_miles"].between(cfg.MILES_MIN, cfg.MILES_MAX),
            "bad_time": ~df["trip_time"].between(cfg.TIME_MIN_S, cfg.TIME_MAX_S),
            "bad_fare": ~df["base_passenger_fare"].between(cfg.FARE_MIN, cfg.FARE_MAX),
            "bad_pay": df["driver_pay"] < 0,
            "bad_wait": ~wait.between(cfg.WAIT_MIN, cfg.WAIT_MAX),
            "bad_vtat": (vtat < 0) | (vtat > wait + 1),        # driver cannot arrive after pickup
            "dur_mismatch": (dur_s - df["trip_time"]).abs() > 300,
            "bad_ppm": ~ppm.between(cfg.PPM_MIN, cfg.PPM_MAX),
            "bad_speed": ~speed.between(cfg.SPEED_MIN, cfg.SPEED_MAX),
            "wrong_month": df["pickup_datetime"].dt.to_period("M") != month,
        }
        bad = pd.Series(False, index=df.index)
        for name, mask in checks.items():
            mask = mask.fillna(True)
            drops[name] += int((mask & ~bad).sum())            # count each row once
            bad |= mask

        keep = ~bad
        df, wait, vtat, ppm, speed = df[keep], wait[keep], vtat[keep], ppm[keep], speed[keep]
        if df.empty:
            continue

        pick = df["pickup_datetime"]
        date_str = pick.dt.strftime("%Y-%m-%d")
        total_cost = (df["base_passenger_fare"] + df["tolls"] + df["bcf"] + df["sales_tax"]
                      + df["congestion_surcharge"] + df["airport_fee"] + df["cbd_congestion_fee"])

        out = pd.DataFrame({
            "provider": pd.Categorical(df["hvfhs_license_num"].map({"HV0003": "Uber", "HV0005": "Lyft"}),
                                       categories=["Uber", "Lyft"]),
            "request_datetime": df["request_datetime"],
            "pickup_datetime": pick,
            "pickup_hour_ts": pick.dt.floor("h"),              # join key for weather
            "hour": pick.dt.hour.astype("int8"),
            "dow": pick.dt.dayofweek.astype("int8"),
            "day": pick.dt.day.astype("int8"),
            "month": pick.dt.month.astype("int8"),
            "is_weekend": (pick.dt.dayofweek >= 5).astype("int8"),
            "is_holiday": date_str.isin(cfg.HOLIDAYS).astype("int8"),
            "is_holiday_eve": date_str.isin(cfg.HOLIDAY_EVES).astype("int8"),
            "pu_zone": df["PULocationID"].astype("int16"),
            "do_zone": df["DOLocationID"].astype("int16"),
            "same_zone": (df["PULocationID"] == df["DOLocationID"]).astype("int8"),
            "trip_miles": df["trip_miles"].astype("float32"),
            "trip_min": (df["trip_time"] / 60).astype("float32"),
            "wait_min": wait.astype("float32"),                # request -> pickup
            "vtat_min": vtat.astype("float32"),                # request -> driver on scene
            "speed_mph": speed.astype("float32"),
            "base_fare": df["base_passenger_fare"].astype("float32"),
            "total_cost": total_cost.astype("float32"),
            "price_per_mile": ppm.astype("float32"),
            "driver_pay": df["driver_pay"].astype("float32"),
            "pay_share": (df["driver_pay"] / df["base_passenger_fare"]).astype("float32"),
            "tips": df["tips"].astype("float32"),              # known only after the ride
            "tolls": df["tolls"].astype("float32"),
            "congestion_surcharge": df["congestion_surcharge"].astype("float32"),
            "cbd_congestion_fee": df["cbd_congestion_fee"].astype("float32"),
            "is_airport": (df["airport_fee"] > 0).astype("int8"),
            "shared_req": (df["shared_request_flag"] == "Y").astype("int8"),
            "shared_match": (df["shared_match_flag"] == "Y").astype("int8"),
            "wav_req": (df["wav_request_flag"] == "Y").astype("int8"),
        })

        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
        kept += len(out)
        print(f"    group {i+1}/{pf.metadata.num_row_groups}: kept {len(out):,} (total {kept:,})", flush=True)
        del df, out, table

    if writer:
        writer.close()

    report = {"input": str(in_path), "output": str(out_path), "month": str(month),
              "rows_in": rows_in, "rows_kept": kept, "rows_dropped": rows_in - kept,
              "drop_pct": round((rows_in - kept) / rows_in * 100, 2),
              "dropped_by_rule": drops, "seconds": round(time.time() - t0, 1)}
    report_path = out_path.with_name(out_path.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(cfg.TRIP_MONTHS)
    for key in wanted:
        src = cfg.TRIP_MONTHS[key]
        print(f"\n=== {key}  {src.name}")
        rep = clean_month(src, cfg.cleaned_path(src))
        print(f"  kept {rep['rows_kept']:,} of {rep['rows_in']:,} "
              f"({rep['drop_pct']}% dropped) in {rep['seconds']}s")
