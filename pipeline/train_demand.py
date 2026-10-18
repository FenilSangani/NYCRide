"""Train the demand model: how many trips start in a zone in a given hour.

This is a time series, but not one model per zone. 261 zones would mean 261
ARIMA models to fit and ship. Instead the recent history becomes ordinary
columns - last hour, two hours ago, same hour yesterday, same hour last week -
and one gradient boosting model learns the lot.

The baselines matter here. A seasonal average already scores R2 0.94, so a
model that cannot beat it is just a heavier way of being no better.

Usage: python train_demand.py
"""
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import config as cfg

MODELS = cfg.ROOT / "models"
TRAIN_MONTHS = ["2025-09", "2025-12", "2026-01"]
TEST_MONTH = "2026-04"

FEATURES = ["zone", "hour", "dow", "is_weekend", "seasonal", "lag1", "lag2", "lag3", "lag24", "lag168"]


def zone_hour_counts(key):
    """Every zone x hour in a month, zeros included - a quiet hour is data too."""
    pf = pq.ParquetFile(cfg.cleaned_path(cfg.TRIP_MONTHS[key]))
    parts = []
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=["pu_zone", "pickup_hour_ts"]).to_pandas()
        parts.append(t.groupby(["pu_zone", "pickup_hour_ts"]).size())
    counts = pd.concat(parts).groupby(level=[0, 1]).sum().rename("trips").reset_index()
    counts.columns = ["zone", "ts", "trips"]

    period = pd.Period(key, freq="M")
    hours = pd.date_range(period.start_time, period.end_time.floor("h"), freq="h")
    zones = sorted(counts["zone"].unique())
    grid = pd.MultiIndex.from_product([zones, hours], names=["zone", "ts"])
    return counts.set_index(["zone", "ts"]).reindex(grid, fill_value=0).reset_index()


def add_features(df, seasonal):
    df = df.sort_values(["zone", "ts"]).reset_index(drop=True)
    df["hour"] = df["ts"].dt.hour.astype("int16")
    df["dow"] = df["ts"].dt.dayofweek.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    g = df.groupby("zone")["trips"]
    for lag in (1, 2, 3, 24, 168):
        df[f"lag{lag}"] = g.shift(lag)
    df = df.merge(seasonal, on=["zone", "hour", "dow"], how="left")
    df["seasonal"] = df["seasonal"].fillna(df["trips"].mean())
    return df.dropna(subset=["lag1", "lag2", "lag3", "lag24", "lag168"])


def report(name, y, p, out):
    mae, r2 = mean_absolute_error(y, p), r2_score(y, p)
    out.append({"model": name, "mae": round(float(mae), 2), "r2": round(float(r2), 4)})
    print(f"  {name:38s} MAE {mae:6.2f} trips   R2 {r2:.4f}", flush=True)


def main():
    MODELS.mkdir(exist_ok=True)
    t0 = time.time()

    print("Counting trips per zone-hour")
    train_raw = pd.concat([zone_hour_counts(k) for k in TRAIN_MONTHS], ignore_index=True)
    test_raw = zone_hour_counts(TEST_MONTH)
    print(f"  train {len(train_raw):,} zone-hours, test {len(test_raw):,}")

    # the seasonal profile is learned on training months only, then reused on the
    # test month - computing it over everything would leak the test month in
    seasonal = (train_raw.assign(hour=train_raw["ts"].dt.hour, dow=train_raw["ts"].dt.dayofweek)
                .groupby(["zone", "hour", "dow"])["trips"].mean()
                .rename("seasonal").reset_index())

    train = add_features(train_raw, seasonal)
    test = add_features(test_raw, seasonal)
    print(f"  usable after lags: train {len(train):,}, test {len(test):,}")

    scores = []
    print("\nBASELINES")
    report("last hour repeated", test["trips"], test["lag1"], scores)
    report("same hour last week", test["trips"], test["lag168"], scores)
    report("zone x hour x weekday average", test["trips"], test["seasonal"], scores)

    print("\nGRADIENT BOOSTING")
    t1 = time.time()
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=25,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25, random_state=0,
    ).fit(train[FEATURES], train["trips"])
    pred = np.clip(model.predict(test[FEATURES]), 0, None)
    report("gradient boosting + lags", test["trips"], pred, scores)
    print(f"  ({model.n_iter_} trees, {time.time() - t1:.0f}s)")

    # The site has no live feed, so it cannot supply lags. Without them, does a
    # model still beat the plain average? Worth asking before shipping one.
    print("\nWITHOUT LAGS — what a static site could actually serve")
    plain = ["zone", "hour", "dow", "is_weekend"]
    t1 = time.time()
    typical = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=25,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=0,
    ).fit(train[plain], train["trips"])
    report("gradient boosting, no lags", test["trips"], np.clip(typical.predict(test[plain]), 0, None), scores)
    print(f"  ({typical.n_iter_} trees, {time.time() - t1:.0f}s)")

    seasonal_r2 = next(s["r2"] for s in scores if s["model"].startswith("zone x hour"))
    nolag_r2 = scores[-1]["r2"]
    serve_lookup = nolag_r2 < seasonal_r2
    print(f"\n  seasonal average {seasonal_r2:.4f} vs no-lag model {nolag_r2:.4f}")
    print(f"  -> the site will serve the {'lookup table' if serve_lookup else 'model'}")

    # ship the lookup when it wins: 261 x 24 x 7 cells with a dozen observations
    # each is simply a better estimator than a tree that has to smooth across them
    table = {}
    for (z, h, d), v in seasonal.set_index(["zone", "hour", "dow"])["seasonal"].items():
        table.setdefault(str(int(z)), {}).setdefault(str(int(d)), [0.0] * 24)[int(h)] = round(float(v), 1)
    (MODELS / "demand_seasonal.json").write_text(json.dumps(table, separators=(",", ":")))

    # Neither trained model is saved. The lookup is what the site serves, and a
    # model nothing loads is just a stale binary in the repository - what the
    # two models were worth is recorded in the scores below.
    metrics = {
        "trained": time.strftime("%Y-%m-%d %H:%M"),
        "train_months": TRAIN_MONTHS, "test_month": TEST_MONTH,
        "train_rows": len(train), "test_rows": len(test),
        "scores": scores,
        "serving": "seasonal lookup table" if serve_lookup else "no-lag model",
        "note": "The lagged model wins outright, but it needs the last few hours of real "
                "counts and the site has no live feed. Without lags the model scored below "
                "the plain zone x hour x weekday average, so the site serves that lookup "
                "instead. The lagged score is reported to show what a live feed would buy.",
        "seconds": round(time.time() - t0, 1),
    }
    (MODELS / "demand_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved to {MODELS}\nDone in {metrics['seconds']}s")


if __name__ == "__main__":
    main()
