"""Build the cleaned hourly weather table from Open-Meteo archive JSON.

The raw JSON lives in DataSets/Weather/raw, one file per month, fetched free and
without an API key. To add a month, download it with curl and re-run this script:

  curl -o "DataSets/Weather/raw/2026-05.json" "https://archive-api.open-meteo.com/v1/archive\
?latitude=40.7128&longitude=-74.0060&start_date=2026-05-01&end_date=2026-05-31\
&hourly=temperature_2m,apparent_temperature,precipitation,rain,snowfall,wind_speed_10m,\
relative_humidity_2m,cloud_cover,visibility,weather_code&timezone=America%2FNew_York"

Usage: python build_weather.py
"""
import json

import pandas as pd

import config as cfg

# WMO weather codes -> short labels (https://open-meteo.com/en/docs)
WMO = {0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Rime fog",
       51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle", 56: "Freezing drizzle",
       57: "Freezing drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
       66: "Freezing rain", 67: "Freezing rain", 71: "Light snow", 73: "Snow", 75: "Heavy snow",
       77: "Snow grains", 80: "Light showers", 81: "Showers", 82: "Violent showers",
       85: "Snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
       96: "Thunderstorm w/ hail", 99: "Thunderstorm w/ hail"}

RENAME = {"time": "hour_ts", "temperature_2m": "temp_c", "apparent_temperature": "feels_like_c",
          "precipitation": "precip_mm", "rain": "rain_mm", "snowfall": "snow_cm",
          "wind_speed_10m": "wind_kmh", "relative_humidity_2m": "humidity_pct",
          "cloud_cover": "cloud_pct", "visibility": "visibility_m"}

NUMERIC = ["temp_c", "feels_like_c", "precip_mm", "rain_mm", "snow_cm", "wind_kmh",
           "humidity_pct", "cloud_pct", "visibility_m"]


def main():
    files = sorted(cfg.WEATHER_RAW.glob("*.json"))
    if not files:
        raise SystemExit(f"no JSON files in {cfg.WEATHER_RAW}")

    frames, meta = [], None
    for p in files:
        j = json.loads(p.read_text())
        if meta is None:
            meta = {k: j[k] for k in ("latitude", "longitude", "timezone") if k in j}
        frames.append(pd.DataFrame(j["hourly"]))
        print(f"  {p.name}: {len(j['hourly']['time'])} hours")

    w = pd.concat(frames, ignore_index=True).rename(columns=RENAME)
    w["hour_ts"] = pd.to_datetime(w["hour_ts"])
    dupes = int(w["hour_ts"].duplicated().sum())
    w = w.drop_duplicates("hour_ts").sort_values("hour_ts").reset_index(drop=True)

    w[NUMERIC] = w[NUMERIC].apply(pd.to_numeric, errors="coerce")   # JSON nulls arrive as object
    # the archive does not serve every variable at every location; an empty column is not data
    dead = [c for c in NUMERIC if w[c].isna().all()]
    w = w.drop(columns=dead)
    num = [c for c in NUMERIC if c not in dead]
    nulls_before = int(w[num].isna().sum().sum())
    w[num] = w[num].interpolate(limit=3).astype("float32")          # short gaps only

    w["weather_code"] = w["weather_code"].astype("int16")
    w["condition"] = w["weather_code"].map(WMO).fillna("Unknown").astype("category")
    w["is_raining"] = (w["precip_mm"] >= 0.1).astype("int8")
    w["is_heavy_rain"] = (w["precip_mm"] >= 2.0).astype("int8")
    w["is_snowing"] = (w["snow_cm"] > 0).astype("int8")
    w["is_freezing"] = (w["temp_c"] <= 0).astype("int8")
    w["rain_band"] = pd.cut(w["precip_mm"], [-1, 0.05, 0.5, 2, 1e6],
                            labels=["dry", "drizzle", "rain", "heavy"]).astype("category")
    w["hour"] = w["hour_ts"].dt.hour.astype("int8")
    w["dow"] = w["hour_ts"].dt.dayofweek.astype("int8")

    w.to_parquet(cfg.WEATHER_OUT, index=False, compression="snappy")

    months = sorted(w["hour_ts"].dt.to_period("M").astype(str).unique())
    expected = sum(pd.Period(m).days_in_month * 24 for m in months)
    report = {
        "source": "Open-Meteo historical archive API (free, no key required)",
        "point": meta, "months": months,
        "rows": len(w), "expected_rows": expected, "missing_hours": expected - len(w),
        "duplicate_hours_removed": dupes, "columns_dropped_all_null": dead,
        "nulls_interpolated": nulls_before, "nulls_remaining": int(w[num].isna().sum().sum()),
        "columns": list(w.columns),
        "rainy_hours": int(w["is_raining"].sum()), "heavy_rain_hours": int(w["is_heavy_rain"].sum()),
        "snow_hours": int(w["is_snowing"].sum()), "freezing_hours": int(w["is_freezing"].sum()),
        "join_key": "hour_ts matches pickup_hour_ts in the cleaned trip files",
        "note": "One weather point for all of NYC - the city is small, but local showers are not captured.",
    }
    path = cfg.WEATHER_OUT.with_name(cfg.WEATHER_OUT.stem + "_report.json")
    path.write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
