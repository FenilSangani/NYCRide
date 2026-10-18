"""Does weather actually help the fare model?

An A/B on the same rows, the same split and the same settings - the only
difference is four extra columns. If the test score does not move, the feature
does not ship, however good "we model the weather" sounds.

Usage: python test_weather.py [--sample 0.05]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import config as cfg

TRAIN_MONTHS = ["2025-09", "2025-12", "2026-01"]
TEST_MONTH = "2026-04"
AIRPORT_ZONES = {1, 132, 138}

BASE = ["pu_zone", "do_zone", "trip_miles", "hour", "dow", "is_weekend",
        "is_holiday", "is_holiday_eve", "is_uber", "shared_req", "pu_airport", "do_airport"]
WEATHER = ["temp_c", "precip_mm", "is_raining", "wind_kmh"]

READ = ["pu_zone", "do_zone", "trip_miles", "hour", "dow", "is_weekend", "is_holiday",
        "is_holiday_eve", "provider", "shared_req", "base_fare", "pickup_hour_ts"]


def load(key, frac, seed, weather):
    pf = pq.ParquetFile(cfg.cleaned_path(cfg.TRIP_MONTHS[key]))
    rng = np.random.default_rng(seed)
    parts = [pf.read_row_group(i, columns=READ).to_pandas()
             .sample(frac=frac, random_state=int(rng.integers(1e9)))
             for i in range(pf.metadata.num_row_groups)]
    df = pd.concat(parts, ignore_index=True)
    df["is_uber"] = (df["provider"] == "Uber").astype("int8")
    df["pu_airport"] = df["pu_zone"].isin(AIRPORT_ZONES).astype("int8")
    df["do_airport"] = df["do_zone"].isin(AIRPORT_ZONES).astype("int8")
    df = df.merge(weather, left_on="pickup_hour_ts", right_on="hour_ts", how="left")
    return df.drop(columns=["provider", "pickup_hour_ts", "hour_ts"])


def fit_score(cols, train, test, label):
    t0 = time.time()
    m = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=0,
    ).fit(train[cols], np.log(train["base_fare"]))
    pred = np.exp(m.predict(test[cols]))
    mae = mean_absolute_error(test["base_fare"], pred)
    r2 = r2_score(test["base_fare"], pred)
    print(f"  {label:28s} MAE ${mae:6.3f}   R2 {r2:.5f}   ({m.n_iter_} trees, {time.time()-t0:.0f}s)")
    return {"label": label, "mae": float(mae), "r2": float(r2)}


def main(sample):
    weather = pd.read_parquet(cfg.WEATHER_OUT)[
        ["hour_ts", "temp_c", "precip_mm", "is_raining", "wind_kmh"]]

    print(f"Loading (sample {sample:.0%})")
    train = pd.concat([load(k, sample, 5 + i, weather) for i, k in enumerate(TRAIN_MONTHS)],
                      ignore_index=True)
    test = load(TEST_MONTH, sample, 77, weather)
    print(f"  train {len(train):,}   test {len(test):,}")
    print(f"  rainy rows in train: {train['is_raining'].mean():.1%}")

    print("\nSame rows, same split, same settings - only the columns differ")
    without = fit_score(BASE, train, test, "without weather")
    with_w = fit_score(BASE + WEATHER, train, test, "with weather")

    d_mae = without["mae"] - with_w["mae"]
    d_r2 = with_w["r2"] - without["r2"]
    print(f"\n  weather changes MAE by ${d_mae:+.3f} and R2 by {d_r2:+.5f}")

    # a cent of MAE on a $25 fare is noise, not a finding
    keep = d_mae > 0.02
    print(f"  verdict: {'keep it' if keep else 'drop it - the gain is inside the noise'}")

    out = {"sample": sample, "train_rows": len(train), "test_rows": len(test),
           "without_weather": without, "with_weather": with_w,
           "delta_mae": round(d_mae, 4), "delta_r2": round(d_r2, 5),
           "keep": bool(keep),
           "note": "Rain lifts the citywide mean fare 5-6% in September and April, but that "
                   "lift is already carried by hour and zone, which the model has. The test "
                   "here is whether weather adds anything on top."}
    (cfg.ROOT / "models" / "weather_experiment.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=0.05)
    main(ap.parse_args().sample)
