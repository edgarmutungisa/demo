"""Satellite grass greenness for the Time2Graze pilot area, published as small tiles.

Run by the scheduled job in the public demo repo (``scripts/greenness.py`` there, mirrored from
here) next to the NOTICES refresh. It needs Google Earth Engine credentials:

* ``GEE_SERVICE_ACCOUNT_KEY``: the service-account JSON key (the whole file, as a secret);
* ``GEE_PROJECT``: the Cloud project registered for Earth Engine.

For every 0.0025 degree cell (about 280 m) of Kasese and Isingiro districts (FAO GAUL boundaries,
plus the rest of each 0.1 degree tile they touch) it writes:

* the median Sentinel-2 NDVI of clear pixels (Cloud Score+ cs_cdf >= 0.60) over the last 30 days,
* only where at least half the cell is grass or shrub land (ESA WorldCover 2021, classes 30 and 20)
  outside protected areas (WDPA); other cells are empty.

Output: ``<out>/index.json`` (date, image window, tiles) and ``<out>/<tile>.json`` with NDVI x 100
as integers (-1 = no pasture or no clear image). The demo and ``grazing.rank_pasture_cells`` read
the cells within reach of a farm. Satellite images arrive every 5 days, so a run younger than
``--min-age-hours`` is skipped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

TILE_DEG = 0.1
STEP_DEG = 0.0025
CELLS = round(TILE_DEG / STEP_DEG)           # 40 x 40 cells per tile
DISTRICTS = ("Isingiro", "Kasese")
WINDOW_DAYS = 30
CLEAR = 0.60                                  # Cloud Score+ threshold, as domain/services/paddocks.py
MIN_PASTURE_SHARE = 0.5
SOURCE = ("Sentinel-2 L2A NDVI, median of clear pixels (Cloud Score+ >= 0.60) in the last 30 days; "
          "grass and shrub land from ESA WorldCover 2021; protected areas (WDPA) removed")


# ---------------------------------------------------------------- tile grid (pure)
def tile_index(lat: float, lon: float) -> Tuple[int, int]:
    return math.floor(round(lat / TILE_DEG, 9)), math.floor(round(lon / TILE_DEG, 9))


def tile_key(i: int, j: int) -> str:
    return f"t{i}_{j}"


def tiles_for_bounds(south: float, west: float, north: float, east: float) -> List[Tuple[int, int]]:
    i0, j0 = tile_index(south, west)
    i1, j1 = tile_index(north, east)
    return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]


def tiles_near(lat: float, lon: float, km: float) -> List[str]:
    dlat = km / 111.32
    dlon = dlat / max(0.2, math.cos(math.radians(lat)))
    return [tile_key(i, j) for i, j in tiles_for_bounds(lat - dlat, lon - dlon, lat + dlat, lon + dlon)]


def cells_from_tile(tile: dict) -> List[dict]:
    """Tile JSON -> [{"lat", "lon", "ndvi"}] cell centres with pasture."""
    out = []
    lat_top, lon0, step, cols = tile["lat0"] + TILE_DEG, tile["lon0"], tile["step"], tile["cols"]
    for k, v in enumerate(tile["ndvi"]):
        if v is None or v < 0:
            continue
        r, c = divmod(k, cols)
        out.append({"lat": round(lat_top - (r + 0.5) * step, 5), "lon": round(lon0 + (c + 0.5) * step, 5),
                    "ndvi": v / 100, "land_cover": "grass or shrub land"})
    return out


def encode(rows: List[List[Optional[float]]], share: List[List[Optional[float]]]) -> List[int]:
    """sampleRectangle arrays (north row first) -> flat NDVI x 100, -1 where not pasture or no image."""
    flat = []
    for r in range(len(rows)):
        for c in range(len(rows[r])):
            v, s = rows[r][c], share[r][c]
            ok = v is not None and s is not None and v > -1 and s >= MIN_PASTURE_SHARE
            flat.append(int(round(max(-1.0, min(1.0, v)) * 100)) if ok else -1)
    return flat


# ---------------------------------------------------------------- Earth Engine
def init_ee() -> None:
    import ee

    key = os.environ.get("GEE_SERVICE_ACCOUNT_KEY", "").strip()
    project = os.environ.get("GEE_PROJECT", "").strip() or None
    if not key:
        raise RuntimeError("GEE_SERVICE_ACCOUNT_KEY is not set")
    info = json.loads(key)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(info, f)
        path = f.name
    try:
        ee.Initialize(ee.ServiceAccountCredentials(info["client_email"], path), project=project or info.get("project_id"))
    finally:
        os.unlink(path)


def pasture_ndvi_image(end: dt.date, bounds):
    """(image with bands ndvi and share at the cell grid's native 10 m, latest image date)."""
    import ee

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(bounds)
          .filterDate((end - dt.timedelta(days=WINDOW_DAYS)).isoformat(), (end + dt.timedelta(days=1)).isoformat())
          .linkCollection(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"), ["cs_cdf"])
          .map(lambda i: i.updateMask(i.select("cs_cdf").gte(CLEAR))))
    lc = ee.ImageCollection("ESA/WorldCover/v200").first()
    # a composite has no native scale: give it WorldCover's 10 m grid so the 280 m mean is a true area mean
    ndvi = s2.median().normalizedDifference(["B8", "B4"]).rename("ndvi").setDefaultProjection(lc.projection())
    wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons").filterBounds(bounds)
    protected = ee.Image.constant(0).paint(wdpa, 1)
    pasture = lc.eq(20).Or(lc.eq(30)).And(protected.Not()).rename("share")
    latest = s2.aggregate_max("system:time_start")
    return ndvi.updateMask(pasture).addBands(pasture.toFloat()), latest


def fetch_tile(img, i: int, j: int) -> dict:
    import ee

    lat0, lon0 = round(i * TILE_DEG, 4), round(j * TILE_DEG, 4)
    region = ee.Geometry.Rectangle([lon0, lat0, lon0 + TILE_DEG, lat0 + TILE_DEG], proj="EPSG:4326", geodesic=False)
    grid = ee.Projection("EPSG:4326", [STEP_DEG, 0, lon0, 0, -STEP_DEG, lat0 + TILE_DEG])
    coarse = (img.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=2048).reproject(grid)
              .unmask(-9, sameFootprint=False))
    px = coarse.sampleRectangle(region=region, defaultValue=-9).getInfo()["properties"]
    nd = [[None if v == -9 else v for v in row[:CELLS]] for row in px["ndvi"][:CELLS]]
    sh = [[None if v == -9 else v for v in row[:CELLS]] for row in px["share"][:CELLS]]
    return {"key": tile_key(i, j), "lat0": lat0, "lon0": lon0, "step": STEP_DEG, "rows": len(nd), "cols": len(nd[0]) if nd else 0,
            "ndvi": encode(nd, sh)}


def district_bounds(names: Iterable[str]) -> Dict[str, List[float]]:
    import ee

    out: Dict[str, List[float]] = {}
    for level, field in (("level2", "ADM2_NAME"), ("level1", "ADM1_NAME")):   # same level as the app first
        missing = [n for n in names if n not in out]
        if not missing:
            break
        fc = (ee.FeatureCollection(f"FAO/GAUL/2015/{level}").filter(ee.Filter.eq("ADM0_NAME", "Uganda"))
              .filter(ee.Filter.inList(field, missing)))
        info = fc.map(lambda f: ee.Feature(None, {"name": f.get(field), "b": f.geometry().bounds(1000).coordinates()})).getInfo()
        for f in info["features"]:
            ring = f["properties"]["b"][0]
            lons, lats = [p[0] for p in ring], [p[1] for p in ring]
            box = [min(lats), min(lons), max(lats), max(lons)]
            old = out.get(f["properties"]["name"])
            out[f["properties"]["name"]] = box if old is None else [min(old[0], box[0]), min(old[1], box[1]), max(old[2], box[2]), max(old[3], box[3])]
    return out


def build(out_dir: Path, end: Optional[dt.date] = None, workers: int = 8) -> dict:
    import ee

    end = end or dt.date.today()
    bounds = district_bounds(DISTRICTS)
    if not bounds:
        raise RuntimeError("district boundaries not found in FAO GAUL")
    tiles = sorted({t for b in bounds.values() for t in tiles_for_bounds(*b)})
    s, w, n, e = (min(b[0] for b in bounds.values()), min(b[1] for b in bounds.values()),
                  max(b[2] for b in bounds.values()), max(b[3] for b in bounds.values()))
    img, latest = pasture_ndvi_image(end, ee.Geometry.Rectangle([w, s, e, n]))
    latest_ms = latest.getInfo()
    out_dir.mkdir(parents=True, exist_ok=True)
    done, failed = [], []

    def one(t):
        try:
            tile = fetch_tile(img, *t)
            (out_dir / f"{tile['key']}.json").write_text(json.dumps(tile, separators=(",", ":")), encoding="utf-8")
            return tile["key"], sum(v >= 0 for v in tile["ndvi"])
        except Exception as exc:  # one failed tile must not lose the others
            return tile_key(*t), f"error: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, res in pool.map(one, tiles):
            (failed if isinstance(res, str) else done).append((key, res))
    index = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
             "window": [(end - dt.timedelta(days=WINDOW_DAYS)).isoformat(), end.isoformat()],
             "latest_image": dt.datetime.fromtimestamp(latest_ms / 1000, dt.timezone.utc).date().isoformat() if latest_ms else None,
             "tile_deg": TILE_DEG, "step_deg": STEP_DEG, "districts": bounds, "tiles": sorted(k for k, _ in done),
             "pasture_cells": sum(c for _, c in done), "failed_tiles": [k for k, _ in failed], "source": SOURCE}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default="greenness")
    ap.add_argument("--min-age-hours", type=float, default=20.0)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    try:
        prev = json.loads((out / "index.json").read_text(encoding="utf-8"))
        age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(prev["generated_at"])
        if age < dt.timedelta(hours=a.min_age_hours) and not prev.get("failed_tiles"):
            print(f"skipped: last run {age} ago")
            return 0
    except (OSError, ValueError, KeyError):
        pass
    if not os.environ.get("GEE_SERVICE_ACCOUNT_KEY"):
        print("skipped: GEE_SERVICE_ACCOUNT_KEY secret is not set, so no greenness was published")
        return 0
    init_ee()
    idx = build(out)
    print(f"{len(idx['tiles'])} tiles, {idx['pasture_cells']} pasture cells, latest image {idx['latest_image']}, failed {idx['failed_tiles']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
