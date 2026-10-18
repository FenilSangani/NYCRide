"""Clean the NYC Permitted Events CSV and build an hourly per-borough table.

Events carry a borough but no coordinates, so the hourly table keys on
borough x hour. That is the finest join the source supports.

Usage: python clean_events.py
"""
import json

import numpy as np
import pandas as pd

import config as cfg

FMT = "%m/%d/%Y %I:%M:%S %p"
MAX_HOURS = 24 * 14        # longer than this is a season-long permit, not an event
CROWD_TYPES = {"Parade", "Street Festival", "Single Block Festival", "Block Party",
               "Religious Event", "Special Event", "Production Event"}
COUNT_COLS = ["events_active", "events_starting", "closures_active", "crowd_events_active"]

RENAME = {"Event ID": "event_id", "Event Name": "event_name", "Start Date/Time": "start",
          "End Date/Time": "end", "Event Agency": "agency", "Event Type": "event_type",
          "Event Borough": "borough", "Event Location": "location",
          "Event Street Side": "street_side", "Street Closure Type": "closure_type",
          "Community Board": "community_board", "Police Precinct": "police_precinct",
          "CEMSID": "cems_id"}


def clean_events():
    raw = pd.read_csv(cfg.EVENTS_CSV, low_memory=False)
    n_in = len(raw)
    df = raw.rename(columns=RENAME)
    drops = {}

    df["start"] = pd.to_datetime(df["start"], format=FMT, errors="coerce")
    df["end"] = pd.to_datetime(df["end"], format=FMT, errors="coerce")

    bad = df["start"].isna() | df["end"].isna()
    drops["unparseable_dates"] = int(bad.sum())
    df = df[~bad]

    dur_h = (df["end"] - df["start"]).dt.total_seconds() / 3600
    bad = dur_h <= 0
    drops["end_before_or_equal_start"] = int(bad.sum())
    df, dur_h = df[~bad], dur_h[~bad]

    bad = df["borough"].isna()
    drops["missing_borough"] = int(bad.sum())
    df, dur_h = df[~bad], dur_h[~bad]

    # "5," / "20, 26, 28," -> "5" / "20,26,28"
    for c in ["community_board", "police_precinct"]:
        df[c] = (df[c].fillna("").astype(str)
                 .str.replace(r"[,\s]+$", "", regex=True)
                 .str.replace(r"\s*,\s*", ",", regex=True)
                 .replace("", np.nan))
    for c in ["event_name", "location", "event_type", "agency", "borough", "closure_type", "street_side"]:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})
    df["location"] = df["location"].str.replace(r"\s{2,}", " ", regex=True)

    clean = pd.DataFrame({
        "event_id": df["event_id"].astype("int64"),
        "event_name": df["event_name"],
        "event_type": df["event_type"].astype("category"),
        "agency": df["agency"].astype("category"),
        "borough": df["borough"].astype("category"),
        "location": df["location"],
        "start": df["start"],
        "end": df["end"],
        "start_date": df["start"].dt.date.astype("string"),
        "start_hour_ts": df["start"].dt.floor("h"),
        "hour": df["start"].dt.hour.astype("int8"),
        "dow": df["start"].dt.dayofweek.astype("int8"),
        "duration_h": dur_h.astype("float32"),
        "is_long_permit": (dur_h > MAX_HOURS).astype("int8"),
        "has_closure": df["closure_type"].notna().astype("int8"),
        "is_full_closure": df["closure_type"].fillna("").str.contains("Full Street", case=False).astype("int8"),
        "closure_type": df["closure_type"].astype("category"),
        "street_side": df["street_side"].astype("category"),
        "is_crowd_event": df["event_type"].isin(CROWD_TYPES).astype("int8"),
        "community_board": df["community_board"],
        "police_precinct": df["police_precinct"],
    }).sort_values("start").reset_index(drop=True)

    clean.to_parquet(cfg.EVENTS_OUT, index=False, compression="snappy")
    return clean, n_in, drops


def build_hourly(clean):
    short = clean[clean["is_long_permit"] == 0]
    frames = []
    for key in cfg.TRIP_MONTHS:
        period = pd.Period(key, freq="M")
        hours = pd.date_range(period.start_time, period.end_time.floor("h"), freq="h")
        live = short[(short["end"] >= hours[0]) & (short["start"] <= hours[-1])]
        hv = hours.values.astype("datetime64[ns]")
        for borough in clean["borough"].cat.categories:
            b = live[live["borough"] == borough]
            if b.empty:
                continue
            starts = b["start"].values.astype("datetime64[ns]")
            ends = b["end"].values.astype("datetime64[ns]")
            closure = b["has_closure"].values.astype(bool)
            crowd = b["is_crowd_event"].values.astype(bool)
            hour_step = np.timedelta64(1, "h")
            active = (starts[None, :] <= hv[:, None] + hour_step) & (ends[None, :] > hv[:, None])
            begins = (starts[None, :] >= hv[:, None]) & (starts[None, :] < hv[:, None] + hour_step)
            frames.append(pd.DataFrame({
                "hour_ts": hours,
                "borough": borough,
                "events_active": active.sum(1).astype("int16"),
                "events_starting": begins.sum(1).astype("int16"),
                "closures_active": (active & closure[None, :]).sum(1).astype("int16"),
                "crowd_events_active": (active & crowd[None, :]).sum(1).astype("int16"),
            }))

    hourly = pd.concat(frames, ignore_index=True)
    # a count that is zero in every hour carries no signal - drop it rather than feed the model noise
    dead = [c for c in COUNT_COLS if hourly[c].max() == 0]
    hourly = hourly.drop(columns=dead)
    hourly["borough"] = hourly["borough"].astype("category")
    hourly["hour"] = hourly["hour_ts"].dt.hour.astype("int8")
    hourly["dow"] = hourly["hour_ts"].dt.dayofweek.astype("int8")
    hourly = hourly.sort_values(["borough", "hour_ts"]).reset_index(drop=True)
    hourly.to_parquet(cfg.EVENTS_HOURLY_OUT, index=False, compression="snappy")
    return hourly, dead


if __name__ == "__main__":
    clean, n_in, drops = clean_events()
    hourly, dead = build_hourly(clean)

    closure_years = sorted(clean.loc[clean["has_closure"] == 1, "start"].dt.year.unique().tolist())
    report = {
        "source": str(cfg.EVENTS_CSV), "rows_in": n_in, "rows_kept": len(clean),
        "rows_dropped": n_in - len(clean), "drop_pct": round((n_in - len(clean)) / n_in * 100, 2),
        "dropped_by_rule": drops,
        "date_range": [str(clean["start"].min()), str(clean["start"].max())],
        "long_permits_flagged": int(clean["is_long_permit"].sum()),
        "events_with_street_closure": int(clean["has_closure"].sum()),
        "closure_data_years": closure_years,
        "hourly_rows": len(hourly), "hourly_columns": list(hourly.columns),
        "hourly_columns_dropped_all_zero": dead,
        "note": "Street-closure permits (parades, street festivals) only appear up to 2016, so for the "
                "2025-2026 trip months the events are Parks Department permits only - a weak demand signal.",
    }
    path = cfg.EVENTS_OUT.with_name(cfg.EVENTS_OUT.stem + "_report.json")
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
