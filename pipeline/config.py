"""Shared paths and cleaning thresholds for the data pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "DataSets"

TRIP_MONTHS = {
    "2025-09": DATA / "Sep 2025(Autumn)" / "Sep 2025.parquet",
    "2025-12": DATA / "Dec 2025(Winter & Holidays)" / "Dec 2025 Data.parquet",
    "2026-01": DATA / "Jan 2026(Cold)" / "Jan 2026.parquet",
    "2026-04": DATA / "Apr 2026(Rain)" / "Apr 2026.parquet",
}

EVENTS_CSV = DATA / "Events" / "NYC_Permitted_Event_Information_-_Historical.csv"
EVENTS_OUT = DATA / "Events" / "NYC Events (cleaned).parquet"
EVENTS_HOURLY_OUT = DATA / "Events" / "NYC Events Hourly (cleaned).parquet"

ZONES_RAW = DATA / "Zones" / "raw"
ZONES_GEOJSON = DATA / "Zones" / "nyc_zones (cleaned).geojson"
ZONES_CSV = DATA / "Zones" / "nyc_zones (cleaned).csv"

WEATHER_RAW = DATA / "Weather" / "raw"
WEATHER_OUT = DATA / "Weather" / "nyc_weather (cleaned).parquet"


def cleaned_path(raw_path: Path) -> Path:
    """Dec 2025 Data.parquet -> Dec 2025 Data (cleaned).parquet, same folder."""
    return raw_path.with_name(f"{raw_path.stem} (cleaned){raw_path.suffix}")


# ---- trip cleaning thresholds (chosen from the data, see pipeline/README.md) ----
UNKNOWN_ZONES = {264, 265}          # "Unknown" / "Outside NYC" - no shape on the map
MILES_MIN, MILES_MAX = 0.1, 100.0
TIME_MIN_S, TIME_MAX_S = 60, 6 * 3600
FARE_MIN, FARE_MAX = 1.0, 1000.0
WAIT_MIN, WAIT_MAX = 0.0, 60.0      # minutes between request and pickup
PPM_MIN, PPM_MAX = 1.0, 100.0       # $/mile
SPEED_MIN, SPEED_MAX = 1.0, 60.0    # mph; NYC highways cap at 55

# ---- calendar ----
HOLIDAYS = {"2025-09-01", "2025-11-27", "2025-12-25", "2026-01-01", "2026-01-19",
            "2026-04-03", "2026-04-05"}        # Labor Day, Thanksgiving, Christmas,
                                               # New Year, MLK, Good Friday, Easter
HOLIDAY_EVES = {"2025-12-24", "2025-12-31"}
