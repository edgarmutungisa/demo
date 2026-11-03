"""Ready-made rain forecasts for a grid across Uganda, so every answer has a prediction at once.

Run daily by the scheduled job in the public demo repo (``scripts/forecast_grid.py`` there, mirrored
from here; it needs ``scripts/prediction_models.py`` and ``scripts/uganda_climate.py``, mirrors of
``domain/services/prediction/models.py`` and ``domain/services/geospatial/uganda_climate.py``).
No keys are needed.

For each point of a 0.5 degree grid over Uganda (about 50 km; the sources are no finer):

* history: NASA POWER daily precipitation (PRECTOTCORR, MERRA-2 corrected with GPCP), maximum and
  minimum temperature and humidity at 2 m, 1981 to the latest day published;
* forecast: the ECMWF seasonal ensemble (Open-Meteo), daily rain for about 6 months;
* the same models as the farm tool: tercile outlook per month and for the next 3 months against
  the point's 1996-2025 rainfall, the analogue forecast from its own records with its
  leave-one-year-out test, the two combined, and the onset of the next rainy season;
* the climatology kept with it (monthly normals, past onset dates), so a usual-start prediction
  is still possible when the forecast is old.

Output: ``forecasts.json``. The page uses the nearest point, says how far away it is and when the
forecast was issued, and switches to the farm's own records whenever they have loaded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

try:  # inside DSTUganda
    from domain.services.geospatial import uganda_climate as uc
    from domain.services.prediction import models as pm
except ImportError:  # in the public demo repo (mirrors of the same two files)
    import prediction_models as pm  # type: ignore
    import uganda_climate as uc  # type: ignore

GRID_DEG = 0.5
BOUNDS = (-1.5, 29.5, 4.0, 35.0)              # Uganda: south, west, north, east
CLIMATE_START = 1981
BASELINE_YEARS = 30
SEASONS = {"bimodal": {"long_rains": (2, 15), "short_rains": (8, 1)}, "unimodal": {"rains": (3, 1)}}
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
SEASONAL_URL = "https://seasonal-api.open-meteo.com/v1/seasonal"
UA = {"User-Agent": "DSTUganda/1.0 (WWF Uganda Time2Graze; forecast grid)"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def grid_points(bounds=BOUNDS, step=GRID_DEG) -> List[Tuple[float, float]]:
    s, w, n, e = bounds
    pts, lat = [], s
    while lat <= n + 1e-9:
        lon = w
        while lon <= e + 1e-9:
            pts.append((round(lat, 2), round(lon, 2)))
            lon += step
        lat += step
    return pts


def monthly_totals(daily) -> Dict[Tuple[int, int], float]:
    acc: Dict[Tuple[int, int], List[float]] = {}
    for d, v in daily:
        acc.setdefault((d.year, d.month), []).append(v)
    return {k: sum(v) for k, v in acc.items() if len(v) >= 28}


def rain_pattern(normals: List[float]) -> str:
    """Two rainy seasons unless June-July is as wet as the drier of the two rainy seasons."""
    mam, jj, son = statistics.fmean(normals[2:5]), statistics.fmean(normals[5:7]), statistics.fmean(normals[8:11])
    return "unimodal" if jj >= 0.7 * min(mam, son) else "bimodal"


def onset_history(daily, windows: Dict[str, Tuple[int, int]], years: range) -> Dict[str, dict]:
    """Day of year of each past onset (same rule as the farm tool), and the years with no clear start."""
    rain = sorted(daily)
    out = {}
    for name, (m, d) in windows.items():
        doys, failed = [], []
        for y in years:
            start = dt.date(y, m, d)
            end = start + dt.timedelta(days=120)
            seg = [(dd, v) for dd, v in rain if start - dt.timedelta(days=5) <= dd <= end + dt.timedelta(days=40)]
            if not seg or seg[-1][0] < end:
                continue
            o = pm._onset(seg, start, end)
            (doys.append(o.timetuple().tm_yday) if o else failed.append(y))
        out[name] = {"doys": sorted(doys), "failed_years": failed}
    return out


def _month_means(series, first: int, last: int) -> List[Optional[float]]:
    acc: Dict[Tuple[int, int], List[float]] = {}
    for d, v in series:
        if first <= d.year <= last:
            acc.setdefault((d.year, d.month), []).append(v)
    out = []
    for m in range(1, 13):
        vals = [statistics.fmean(v) for (y, mm), v in acc.items() if mm == m and len(v) >= 28]
        out.append(round(statistics.fmean(vals), 1) if vals else None)
    return out


def last_30_vs_normal(daily, first: int, last: int) -> dict:
    """Rain in the 30 days to the last recorded day against the same window in the baseline years."""
    rain = dict(daily)
    end = max(rain)
    total = lambda e: sum(rain.get(e - dt.timedelta(days=i), 0.0) for i in range(30))
    past = []
    for y in range(first, last + 1):
        try:
            past.append(total(end.replace(year=y)))
        except ValueError:          # 29 February
            past.append(total(end.replace(year=y, day=28)))
    value, mean, sd = total(end), statistics.fmean(past), statistics.pstdev(past)
    return {"value": round(value, 1), "normal": round(mean, 1), "difference_pct": round((value - mean) / mean * 100) if mean else 0,
            "condition": uc.anomaly_label(value, mean, sd), "window_end": end.isoformat(), "years_compared": len(past)}


def point_forecast(lat: float, lon: float, daily, ens_rain, today: dt.date, temps: Optional[dict] = None) -> dict:
    """Everything the page needs for one grid point. ``daily``: [(date, mm)] since 1981;
    ``temps``: {"tmax": [(date, C)], "tmin": [...], "rh": [(date, %)]} from the same source."""
    last_full = today.year - 1
    first = last_full - BASELINE_YEARS + 1
    mon = monthly_totals(daily)
    base = {k: v for k, v in mon.items() if first <= k[0] <= last_full}
    normals = [round(statistics.fmean([v for (y, m), v in base.items() if m == mm]), 1) for mm in range(1, 13)]
    pattern = rain_pattern(normals)
    windows = SEASONS[pattern]
    obs_end = max(d for d, _ in daily)
    last30 = round(sum(v for d, v in daily if (obs_end - d).days < 30), 1)
    rec = {"lat": lat, "lon": lon, "pattern": pattern, "normals_mm": normals, "period": f"{first}-{last_full}",
           "records_from": min(d for d, _ in daily).year, "observed_to": obs_end.isoformat(), "rain_last_30_days_mm": last30,
           "onset_history": onset_history(daily, windows, range(first, last_full + 1)),
           "last_30_days": last_30_vs_normal(daily, first, last_full)}
    if temps:
        rec["tmax_normal_c"] = _month_means(temps.get("tmax", []), first, last_full)
        rec["tmin_normal_c"] = _month_means(temps.get("tmin", []), first, last_full)
        rec["rh_normal_pct"] = _month_means(temps.get("rh", []), first, last_full)
    uc_seasons = {n: ((m, d), (5, 31) if n == "long_rains" else (10, 31) if n == "short_rains" else (6, 30)) for n, (m, d) in windows.items()}
    try:
        ch = uc.climate_changes(daily, (temps or {}).get("tmax", []), uc_seasons, today)
        rec["changes"] = {"early_period": ch.get("early_period"), "recent_period": ch.get("recent_period"),
                          "years_analysed": ch.get("years_analysed"), "statements": ch.get("statements", [])}
    except Exception as exc:  # the change check must never lose the forecast
        rec["changes"] = {"statements": [], "error": str(exc)}
    if ens_rain:
        out = pm.season_outlook(ens_rain, base)
        s3 = out.get("next_3_months")
        keys = [tuple(int(x) for x in k.split("-")) for k in (s3["months"] if s3 else [])]
        local = pm.local_records_outlook(daily, keys) if keys else None
        name, start, end = pm.season_target(today, windows)
        hist = rec["onset_history"][name]["doys"]
        usual = int(statistics.median(hist)) if hist else None
        observed = [(d, v) for d, v in daily if d >= start - dt.timedelta(days=5)]
        onset = pm.onset_forecast(ens_rain, observed, start, end, usual)
        onset.update({"season": name, "season_window": [start.isoformat(), end.isoformat()]})
        rec["forecast"] = {"issued": today.isoformat(), "members": out["members"], "months": out["months"][:6], "next_3_months": s3,
                           "local_records_outlook": local, "combined_next_3_months": pm.combine_outlooks(s3["probabilities"] if s3 else None, local),
                           "onset": onset}
    return rec


# ---------------------------------------------------------------- fetching
def _get(url: str, params: dict, timeout: float = 90.0, tries: int = 4) -> dict:
    import httpx

    for k in range(tries):
        try:
            r = httpx.get(url, params=params, timeout=timeout, headers=UA)
            if r.status_code == 429:
                raise RuntimeError("rate limited")
            r.raise_for_status()
            return r.json()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)
    raise RuntimeError("unreachable")


def fetch_power(lat: float, lon: float, today: dt.date):
    """(daily rain, {"tmax", "tmin", "rh"}) for the point."""
    j = _get(POWER_URL, {"parameters": "PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M", "community": "AG", "latitude": lat, "longitude": lon,
                         "start": f"{CLIMATE_START}0101", "end": today.strftime("%Y%m%d"), "format": "JSON"})
    par = j["properties"]["parameter"]
    fill = (j.get("header") or {}).get("fill_value", -999)
    conv = lambda key, lo: [(dt.date(int(k[:4]), int(k[4:6]), int(k[6:])), float(v)) for k, v in par.get(key, {}).items()
                            if v is not None and v != fill and v > lo]
    return conv("PRECTOTCORR", -1), {"tmax": conv("T2M_MAX", -60), "tmin": conv("T2M_MIN", -60), "rh": conv("RH2M", -1)}


def fetch_ensemble(lat: float, lon: float) -> Dict[str, list]:
    j = _get(SEASONAL_URL, {"latitude": lat, "longitude": lon, "daily": "precipitation_sum", "forecast_days": 183, "timezone": "Africa/Kampala"})
    daily = j.get("daily", {})
    dates = [dt.date.fromisoformat(t) for t in daily.get("time", [])]
    rain = {}
    for key, values in daily.items():
        if key.startswith("precipitation_sum"):
            mid = key.split("_member")[1] if "_member" in key else "00"
            rain[mid] = [(d, float(v)) for d, v in zip(dates, values) if v is not None]
    if len(rain) < 10:
        raise ValueError(f"seasonal forecast returned {len(rain)} member(s)")
    return rain


def build(today: Optional[dt.date] = None, workers: int = 4, points=None) -> dict:
    today = today or dt.date.today()
    pts = points or grid_points()
    errors: List[str] = []

    def one(p):
        lat, lon = p
        try:
            daily, temps = fetch_power(lat, lon, today)
        except Exception as exc:
            errors.append(f"{lat},{lon} history: {exc}")
            return None
        try:
            ens = fetch_ensemble(lat, lon)
        except Exception as exc:
            errors.append(f"{lat},{lon} forecast: {exc}")
            ens = None
        try:
            return point_forecast(lat, lon, daily, ens, today, temps)
        except Exception as exc:
            errors.append(f"{lat},{lon} model: {exc}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        recs = [r for r in pool.map(one, pts) if r]
    return {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "grid_deg": GRID_DEG,
            "points": recs, "errors": errors[:50], "error_count": len(errors),
            "sources": {"history": "NASA POWER daily precipitation, temperature and humidity (MERRA-2; rain corrected with GPCP), 1981 to date, about 50 km",
                        "forecast": "ECMWF seasonal ensemble via Open-Meteo, about 6 months ahead"}}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="forecasts.json")
    ap.add_argument("--min-age-hours", type=float, default=20.0)
    a = ap.parse_args(argv)
    try:
        with open(a.out, encoding="utf-8") as f:
            prev = json.load(f)
        age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(prev["generated_at"])
        if age < dt.timedelta(hours=a.min_age_hours) and prev.get("points"):
            print(f"skipped: last run {age} ago")
            return 0
    except (OSError, ValueError, KeyError):
        prev = None
    data = build()
    if not data["points"]:
        print(f"no point succeeded ({data['error_count']} errors), keeping the previous file: {data['errors'][:5]}")
        return 0
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    with_fc = sum(1 for p in data["points"] if p.get("forecast"))
    print(f"{len(data['points'])} points ({with_fc} with a forecast), {data['error_count']} errors {data['errors'][:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
