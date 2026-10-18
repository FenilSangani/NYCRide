"""Turn the TLC taxi-zone shapefile into web-ready map files.

A shapefile cannot be drawn by a browser, so this writes GeoJSON in lat/lon plus
a flat CSV of zone names and centroids. Zone ids match pu_zone / do_zone in the
cleaned trip files.

Usage: python clean_zones.py
"""
import json

import pandas as pd
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform

import config as cfg

SIMPLIFY_FT = 60        # source CRS is NY State Plane (US feet); 60ft is invisible at city zoom
SQFT_PER_SQMI = 27_878_400


def main():
    sf = shapefile.Reader(str(cfg.ZONES_RAW / "taxi_zones.shp"))
    src_crs = CRS.from_wkt((cfg.ZONES_RAW / "taxi_zones.prj").read_text())
    to_wgs = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True).transform
    fields = [f[0] for f in sf.fields[1:]]

    lookup = pd.read_csv(cfg.ZONES_RAW / "taxi_zone_lookup.csv")
    lookup.columns = [c.strip() for c in lookup.columns]
    lk = lookup.set_index("LocationID")

    features, rows, dropped, fixed = [], [], [], 0
    for sr in sf.shapeRecords():
        rec = dict(zip(fields, sr.record))
        zid = int(rec["LocationID"])
        if zid in cfg.UNKNOWN_ZONES:
            dropped.append(zid)
            continue

        geom = shape(sr.shape.__geo_interface__)
        if not geom.is_valid:
            geom = geom.buffer(0)       # self-intersections break area and simplify
            fixed += 1
        area_sqmi = geom.area / SQFT_PER_SQMI
        geom = geom.simplify(SIMPLIFY_FT, preserve_topology=True)
        lon, lat = to_wgs(geom.centroid.x, geom.centroid.y)

        props = {
            "zone_id": zid,
            "zone": str(lk.loc[zid, "Zone"]) if zid in lk.index else str(rec.get("zone", "")),
            "borough": str(lk.loc[zid, "Borough"]) if zid in lk.index else str(rec.get("borough", "")),
            "service_zone": str(lk.loc[zid, "service_zone"]) if zid in lk.index else "",
            "lat": round(lat, 6), "lon": round(lon, 6), "area_sqmi": round(area_sqmi, 4),
        }
        features.append({"type": "Feature", "properties": props,
                         "geometry": mapping(shp_transform(to_wgs, geom))})
        rows.append(props)

    cfg.ZONES_GEOJSON.write_text(json.dumps(
        {"type": "FeatureCollection",
         "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3/CRS84"}},
         "features": features}, separators=(",", ":")))

    df = pd.DataFrame(rows).sort_values("zone_id").reset_index(drop=True)
    df.to_csv(cfg.ZONES_CSV, index=False)

    report = {"source_shapefile": "taxi_zones.shp", "zones_in": len(sf), "zones_kept": len(df),
              "dropped_unknown": dropped, "invalid_geometry_fixed": fixed,
              "simplify_tolerance_ft": SIMPLIFY_FT, "crs": "EPSG:4326 (lat/lon)",
              "geojson_mb": round(cfg.ZONES_GEOJSON.stat().st_size / 1024 / 1024, 2),
              "note": "Zone ids match pu_zone / do_zone in the cleaned trip files."}
    path = cfg.ZONES_GEOJSON.with_name(cfg.ZONES_GEOJSON.stem + "_report.json")
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
