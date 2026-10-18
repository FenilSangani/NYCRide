"""Train the fare model.

What it does, in order:
  1. samples the cleaned trips and builds the model's features
  2. builds the lookups the website needs (zone-pair miles, detour factor)
  3. trains the plain baselines first, so the real model has something to beat
  4. trains gradient boosting, and the 10th/90th percentile versions for the range
  5. scores everything on a month the model has never seen
  6. saves the models, the lookups and a metrics report

Split is by month, never random: a random split lets the model peek at the
future and flatters the score.

Usage: python train_fare.py [--sample 0.10]
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import config as cfg

MODELS = cfg.ROOT / "models"

TRAIN_MONTHS = ["2025-09", "2025-12", "2026-01"]
TEST_MONTH = "2026-04"

READ_COLS = ["pu_zone", "do_zone", "trip_miles", "hour", "dow", "is_weekend",
             "is_holiday", "is_holiday_eve", "provider", "shared_req", "base_fare"]

AIRPORT_ZONES = {1, 132, 138}      # Newark, JFK, LaGuardia

# What the model actually sees. trip_min, tips, driver_pay and price_per_mile are
# all absent on purpose: they are only known once the ride is over, so feeding
# them in would leak the answer.
#
# The airport flags are split by direction and derived from the zone id, not from
# the airport_fee column. The site only knows the two zones, so the feature has to
# be computable from the zones alone - otherwise training and serving quietly
# disagree, which is exactly what made long airport fares read ~10% high.
FEATURES = ["pu_zone", "do_zone", "trip_miles", "hour", "dow", "is_weekend",
            "is_holiday", "is_holiday_eve", "is_uber", "shared_req",
            "pu_airport", "do_airport"]
TARGET = "base_fare"

MIN_PAIR_TRIPS = 20        # below this a zone pair's median mileage is too noisy to trust

MODEL_PARAMS = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                    min_samples_leaf=40, l2_regularization=1.0,
                    early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=25, random_state=0)

QUANTILE_PARAMS = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                       min_samples_leaf=40, early_stopping=True,
                       validation_fraction=0.1, n_iter_no_change=20, random_state=0)


def load_month(key, frac, seed):
    """Read one cleaned month and return a random sample of the columns we model on."""
    path = cfg.cleaned_path(cfg.TRIP_MONTHS[key])
    pf = pq.ParquetFile(path)
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=READ_COLS).to_pandas()
        parts.append(t.sample(frac=frac, random_state=int(rng.integers(1e9))))
    df = pd.concat(parts, ignore_index=True)
    df["is_uber"] = (df["provider"] == "Uber").astype("int8")
    df["pu_airport"] = df["pu_zone"].isin(AIRPORT_ZONES).astype("int8")
    df["do_airport"] = df["do_zone"].isin(AIRPORT_ZONES).astype("int8")
    return df.drop(columns=["provider"])


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def build_distance_lookup(train, centroids):
    """How far is a trip between two zones, when the rider has not told us?

    The site only knows two zones, but the model wants miles. Two answers, in
    order of preference:
      1. the median mileage actually driven between that pair, where we have enough trips
      2. straight-line distance scaled by a detour factor measured from the data
    """
    pair = (train.groupby(["pu_zone", "do_zone"])["trip_miles"]
            .agg(["median", "size"])
            .rename(columns={"median": "miles", "size": "trips"}))
    pair = pair[pair["trips"] >= MIN_PAIR_TRIPS]

    # measure the detour factor rather than guessing one
    idx = pair.index.to_frame(index=False)
    lat1 = idx["pu_zone"].map(centroids["lat"]).to_numpy()
    lon1 = idx["pu_zone"].map(centroids["lon"]).to_numpy()
    lat2 = idx["do_zone"].map(centroids["lat"]).to_numpy()
    lon2 = idx["do_zone"].map(centroids["lon"]).to_numpy()
    straight_mi = haversine_km(lat1, lon1, lat2, lon2) * 0.621371
    ratio = pair["miles"].to_numpy() / np.where(straight_mi > 0.2, straight_mi, np.nan)
    factor = float(np.nanmedian(ratio))

    lookup = {f"{int(a)}-{int(b)}": round(float(m), 2)
              for (a, b), m in pair["miles"].items()}
    return lookup, factor, len(pair)


def report(name, y_true, y_pred, out):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    out.append({"model": name, "mae": round(float(mae), 3), "r2": round(float(r2), 4)})
    print(f"  {name:38s} MAE ${mae:6.2f}   R2 {r2:.4f}", flush=True)


def main(sample):
    MODELS.mkdir(exist_ok=True)
    t0 = time.time()

    print(f"Loading trips (sample {sample:.0%} per month)")
    train = pd.concat([load_month(k, sample, 11 + i) for i, k in enumerate(TRAIN_MONTHS)],
                      ignore_index=True)
    test = load_month(TEST_MONTH, sample, 99)
    print(f"  train {len(train):,} rows from {', '.join(TRAIN_MONTHS)}")
    print(f"  test  {len(test):,} rows from {TEST_MONTH} (never seen during training)")

    zones = pd.read_csv(cfg.ZONES_CSV).set_index("zone_id")

    print("\nBuilding the zone-pair distance lookup (from training months only)")
    lookup, factor, n_pairs = build_distance_lookup(train, zones)
    print(f"  {n_pairs:,} zone pairs with >= {MIN_PAIR_TRIPS} trips")
    print(f"  detour factor (driven miles / straight line) = {factor:.3f}")

    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_te, y_te = test[FEATURES], test[TARGET]
    scores = []

    print("\nBASELINES — the real model has to beat these")
    report("mean fare", y_te, np.full(len(y_te), y_tr.mean()), scores)
    lin1 = LinearRegression().fit(X_tr[["trip_miles"]], y_tr)
    report("linear regression, miles only", y_te, lin1.predict(X_te[["trip_miles"]]), scores)
    lin2_cols = ["trip_miles", "hour", "dow", "do_airport"]
    lin2 = LinearRegression().fit(X_tr[lin2_cols], y_tr)
    report("linear regression, + hour/day/airport", y_te, lin2.predict(X_te[lin2_cols]), scores)

    print("\nGRADIENT BOOSTING")
    t1 = time.time()
    gbm = HistGradientBoostingRegressor(**MODEL_PARAMS).fit(X_tr, y_tr)
    report("gradient boosting", y_te, gbm.predict(X_te), scores)
    print(f"  ({gbm.n_iter_} trees, {time.time() - t1:.0f}s)")

    # fares are right-skewed, so a model on log(fare) is worth a try
    t1 = time.time()
    gbm_log = HistGradientBoostingRegressor(**MODEL_PARAMS).fit(X_tr, np.log(y_tr))
    report("gradient boosting on log(fare)", y_te, np.exp(gbm_log.predict(X_te)), scores)
    print(f"  ({gbm_log.n_iter_} trees, {time.time() - t1:.0f}s)")

    use_log = scores[-1]["mae"] < scores[-2]["mae"]
    best = gbm_log if use_log else gbm
    print(f"\n  keeping the {'log(fare)' if use_log else 'direct'} model")

    print("\nQUANTILES — the range around the estimate")
    quant = {}
    for q in (0.1, 0.9):
        t1 = time.time()
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                          **QUANTILE_PARAMS).fit(X_tr, y_tr)
        quant[q] = m
        print(f"  q{int(q * 100)} trained ({m.n_iter_} trees, {time.time() - t1:.0f}s)")

    lo = quant[0.1].predict(X_te)
    hi = quant[0.9].predict(X_te)
    inside = float(((y_te >= lo) & (y_te <= hi)).mean())
    width = float(np.mean(hi - lo))
    print(f"  {inside:.1%} of unseen fares fall inside the 10-90 band (target 80%)")
    print(f"  mean band width ${width:.2f}")

    print("\nWhich features the model leans on (permutation, 60k rows)")
    from sklearn.inspection import permutation_importance
    sub = test.sample(60000, random_state=3)
    imp = permutation_importance(best, sub[FEATURES],
                                 np.log(sub[TARGET]) if use_log else sub[TARGET],
                                 n_repeats=3, random_state=0, n_jobs=1)
    order = np.argsort(imp.importances_mean)[::-1]
    importance = [{"feature": FEATURES[i], "drop": round(float(imp.importances_mean[i]), 4)}
                  for i in order]
    for row in importance:
        print(f"  {row['feature']:16s} {row['drop']:.4f}")

    joblib.dump({"model": best, "log_target": use_log, "features": FEATURES},
                MODELS / "fare.joblib")
    joblib.dump({"lo": quant[0.1], "hi": quant[0.9], "features": FEATURES},
                MODELS / "fare_range.joblib")
    (MODELS / "zone_pair_miles.json").write_text(json.dumps(
        {"pairs": lookup, "detour_factor": round(factor, 4), "min_trips": MIN_PAIR_TRIPS},
        separators=(",", ":")))

    metrics = {
        "trained": time.strftime("%Y-%m-%d %H:%M"),
        "sample_fraction": sample,
        "train_rows": len(train), "test_rows": len(test),
        "train_months": TRAIN_MONTHS, "test_month": TEST_MONTH,
        "target": "log(base_fare)" if use_log else "base_fare",
        "features": FEATURES,
        "scores": scores,
        "range": {"coverage": round(inside, 4), "mean_width": round(width, 2)},
        "importance": importance,
        "zone_pairs": n_pairs, "detour_factor": round(factor, 4),
        "seconds": round(time.time() - t0, 1),
    }
    (MODELS / "fare_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved to {MODELS}")
    print(f"Done in {metrics['seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=0.10,
                    help="fraction of each month to train on (default 0.10)")
    main(ap.parse_args().sample)
