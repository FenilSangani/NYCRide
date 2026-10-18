"""Run the whole data pipeline: trips, events, zones, weather.

Usage: python run_all.py
"""
import time

import build_weather
import clean_events
import clean_trips
import clean_zones
import config as cfg

t0 = time.time()

print("=" * 60, "\nTRIPS")
for key, src in cfg.TRIP_MONTHS.items():
    print(f"\n--- {key}")
    rep = clean_trips.clean_month(src, cfg.cleaned_path(src))
    print(f"  kept {rep['rows_kept']:,} of {rep['rows_in']:,} ({rep['drop_pct']}% dropped)")

print("=" * 60, "\nEVENTS")
clean, n_in, drops = clean_events.clean_events()
hourly, dead = clean_events.build_hourly(clean)
print(f"  events kept {len(clean):,} of {n_in:,}; hourly rows {len(hourly):,}")

print("=" * 60, "\nZONES")
clean_zones.main()

print("=" * 60, "\nWEATHER")
build_weather.main()

print(f"\nPipeline finished in {round(time.time() - t0, 1)}s")
