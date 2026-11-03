"""Pure calculations for the Uganda data layers (no Earth Engine, fully testable).

Earth Engine returns raw series (daily rainfall, daily temperature, 16-day
NDVI, soil sand/clay %). These functions turn them into what farmers need:
monthly normals over the most recent 30 years, how the weather has changed since 1981, this season versus normal, the usual start of
the rains, pasture condition, warming trend and soil texture class.
"""

from __future__ import annotations

import datetime
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------------------
# Soil texture (USDA texture triangle, from sand and clay %)
# ---------------------------------------------------------------------------


def usda_texture_class(sand: float, clay: float) -> str:
    """USDA soil texture class from sand and clay percentages (silt = remainder)."""
    if not (0 <= sand <= 100 and 0 <= clay <= 100 and sand + clay <= 100.5):
        raise ValueError("sand and clay must be percentages summing to at most 100")
    silt = max(0.0, 100.0 - sand - clay)
    if silt + 1.5 * clay < 15:
        return "Sand"
    if silt + 2 * clay < 30:
        return "Loamy sand"
    if (7 <= clay < 20 and sand > 52) or (clay < 7 and silt < 50):
        return "Sandy loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "Loam"
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12):
        return "Silt loam"
    if silt >= 80 and clay < 12:
        return "Silt"
    if 20 <= clay < 35 and silt < 28 and sand > 45:
        return "Sandy clay loam"
    if 27 <= clay < 40 and 20 < sand <= 45:
        return "Clay loam"
    if 27 <= clay < 40 and sand <= 20:
        return "Silty clay loam"
    if clay >= 35 and sand > 45:
        return "Sandy clay"
    if clay >= 40 and silt >= 40:
        return "Silty clay"
    return "Clay"


SOIL_ADVICE = {
    "Sand": "Drains fast and holds little water or nutrients: drought-tolerant grasses (Rhodes, Buffel), manure and mulch help most.",
    "Loamy sand": "Light soil that dries quickly: choose drought-tolerant grasses and add manure.",
    "Sandy loam": "Easy to work and drains well; good for Brachiaria and Rhodes grass with manure.",
    "Loam": "Balanced, fertile soil: suits Napier, Brachiaria and legumes well.",
    "Silt loam": "Fertile but can crust and erode on slopes: keep grass cover.",
    "Silt": "Fertile but erodes easily: keep the ground covered.",
    "Sandy clay loam": "Holds water reasonably; suits Brachiaria and Rhodes grass.",
    "Clay loam": "Good water holding; suits Napier and Brachiaria; avoid grazing when waterlogged.",
    "Silty clay loam": "Holds water well; can waterlog: graze lightly after heavy rain.",
    "Sandy clay": "Hard when dry and sticky when wet; plant at the start of the rains.",
    "Silty clay": "Heavy, can waterlog; good for Napier in drained areas.",
    "Clay": "Heavy soil that holds water and can waterlog or crack: avoid trampling when wet; good for Napier on raised beds.",
}

# ---------------------------------------------------------------------------
# Climatology and anomalies
# ---------------------------------------------------------------------------


def monthly_totals(daily: Iterable[Tuple[datetime.date, float]]) -> Dict[Tuple[int, int], float]:
    """Sum daily values into (year, month) totals (rainfall)."""
    out: Dict[Tuple[int, int], float] = {}
    for d, v in daily:
        if v is None:
            continue
        out[(d.year, d.month)] = out.get((d.year, d.month), 0.0) + float(v)
    return out


def monthly_means(daily: Iterable[Tuple[datetime.date, float]]) -> Dict[Tuple[int, int], float]:
    """Average daily values into (year, month) means (temperature)."""
    sums: Dict[Tuple[int, int], List[float]] = {}
    for d, v in daily:
        if v is None:
            continue
        sums.setdefault((d.year, d.month), []).append(float(v))
    return {k: sum(v) / len(v) for k, v in sums.items()}


def climatology(by_year_month: Dict[Tuple[int, int], float], complete_years: Optional[Sequence[int]] = None) -> List[dict]:
    """Per calendar month: mean, lowest and highest over the years given."""
    out = []
    for m in range(1, 13):
        vals = [v for (y, mm), v in by_year_month.items() if mm == m and (complete_years is None or y in complete_years)]
        if vals:
            out.append({"month": MONTHS[m - 1], "mean": round(statistics.fmean(vals), 1),
                        "lowest": round(min(vals), 1), "highest": round(max(vals), 1), "years": len(vals)})
    return out


def anomaly_label(value: float, mean: float, std: float) -> str:
    """Plain-language comparison with normal, using standard deviations."""
    if std <= 0:
        return "normal"
    z = (value - mean) / std
    if z <= -1.5:
        return "much below normal"
    if z <= -0.5:
        return "below normal"
    if z < 0.5:
        return "normal"
    if z < 1.5:
        return "above normal"
    return "much above normal"


def compare_to_normal(value: float, history: Sequence[float]) -> dict:
    """This period versus the same period in past years."""
    if len(history) < 3:
        raise ValueError("need at least 3 past years")
    mean = statistics.fmean(history)
    std = statistics.pstdev(history)
    pct = (value - mean) / mean * 100 if mean else 0.0
    return {"value": round(value, 3), "normal": round(mean, 3), "difference_pct": round(pct, 1),
            "condition": anomaly_label(value, mean, std), "years_compared": len(history)}


def trend_per_decade(by_year: Dict[int, float]) -> Optional[float]:
    """Least-squares slope per 10 years (e.g. warming in °C per decade)."""
    pts = sorted(by_year.items())
    if len(pts) < 5:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den * 10, 2)


# ---------------------------------------------------------------------------
# Onset of the rains (agronomic definition)
# ---------------------------------------------------------------------------


def rain_onset(daily: Sequence[Tuple[datetime.date, float]], start: datetime.date, end: datetime.date,
               wet_mm: float = 20.0, wet_days: int = 3, dry_spell_days: int = 10, check_days: int = 30,
               dry_day_mm: float = 1.0) -> Optional[datetime.date]:
    """First day from ``start`` when ``wet_days`` days bring at least ``wet_mm``,
    not followed by a dry spell of ``dry_spell_days`` days within ``check_days``
    (a common agronomic onset rule; avoids false starts)."""
    series = sorted((d, float(v or 0.0)) for d, v in daily)
    idx = {d: i for i, (d, _) in enumerate(series)}
    rain = [v for _, v in series]
    for i, (d, _) in enumerate(series):
        if d < start or d > end or i + wet_days > len(series):
            continue
        if sum(rain[i:i + wet_days]) < wet_mm:
            continue
        window = rain[i + wet_days:i + wet_days + check_days]
        run = longest = 0
        for v in window:
            run = run + 1 if v < dry_day_mm else 0
            longest = max(longest, run)
        if longest < dry_spell_days:
            return d
    return None


def onset_statistics(onsets: Dict[int, Optional[datetime.date]]) -> dict:
    """Typical onset date (median), usual range (middle half of years) and failed seasons."""
    days = sorted((d.timetuple().tm_yday, y) for y, d in onsets.items() if d)
    failed = sorted(y for y, d in onsets.items() if d is None)
    if not days:
        return {"typical": None, "usual_range": None, "years": len(onsets), "no_clear_onset_years": failed}
    doys = [d for d, _ in days]

    def as_date(doy: float) -> str:
        return (datetime.date(2001, 1, 1) + datetime.timedelta(days=int(round(doy)) - 1)).strftime("%d %b")

    q = statistics.quantiles(doys, n=4) if len(doys) >= 4 else [min(doys), statistics.median(doys), max(doys)]
    return {"typical": as_date(statistics.median(doys)), "usual_range": [as_date(q[0]), as_date(q[-1])],
            "earliest": as_date(min(doys)), "latest": as_date(max(doys)),
            "years": len(onsets), "no_clear_onset_years": failed}


# ---------------------------------------------------------------------------
# Pasture condition from NDVI
# ---------------------------------------------------------------------------


def pasture_condition(ndvi_series: Sequence[Tuple[datetime.date, float]], window_days: int = 16) -> dict:
    """Latest NDVI versus the same time of year in all earlier years."""
    pts = sorted((d, v) for d, v in ndvi_series if v is not None)
    if len(pts) < 10:
        raise ValueError("not enough NDVI observations")
    last_date, last_val = pts[-1]
    doy = last_date.timetuple().tm_yday
    past = {}
    for d, v in pts[:-1]:
        if d.year == last_date.year:
            continue
        gap = abs(d.timetuple().tm_yday - doy)
        if min(gap, 365 - gap) <= window_days:
            past.setdefault(d.year, []).append(v)
    history = [statistics.fmean(v) for v in past.values()]
    result = compare_to_normal(last_val, history)
    result.update({"date": last_date.isoformat(), "ndvi": round(last_val, 3),
                   "meaning": {"much below normal": "Grass is far thinner and drier than usual for this time of year: reduce stocking, use stored feed.",
                               "below normal": "Grass is thinner than usual: watch grazing pressure and plan feed.",
                               "normal": "Grass cover is normal for this time of year.",
                               "above normal": "Grass is better than usual: a chance to cut and store fodder.",
                               "much above normal": "Grass is much better than usual: cut and store hay or silage."}[result["condition"]]})
    return result


# ---------------------------------------------------------------------------
# Live season status (recent daily weather versus normal)
# ---------------------------------------------------------------------------

BIMODAL_RAIN_MONTHS = {3, 4, 5, 9, 10, 11}
UNIMODAL_RAIN_MONTHS = {4, 5, 6, 7, 8, 9}
SEASON_SEARCH = {  # (start month, day) of each rainy season's onset search window
    "bimodal": {"long_rains": (2, 15), "short_rains": (8, 1)},
    "unimodal": {"rains": (3, 1)},
}


def season_status(recent: Sequence[Tuple[datetime.date, Optional[float], Optional[float], Optional[float]]],
                  today: datetime.date, rain_pattern: str = "bimodal",
                  rain_normal_monthly_mm: Optional[Dict[str, float]] = None,
                  tmax_normal_monthly_c: Optional[Dict[str, float]] = None) -> dict:
    """How this season is going, from recent daily weather (date, tmax, tmin, rain mm) up to ``today``.

    Works without normals (dry spell, onset, false start, heavy rain); with
    20-year normals it also says how the last 30 days and the last 7 days'
    heat compare with normal.
    """
    past = sorted((d, tx, tn, p) for d, tx, tn, p in recent if d <= today)
    if len(past) < 10:
        raise ValueError("need at least 10 days of recent weather")
    rain = [(d, float(p or 0.0)) for d, _, _, p in past]
    rain_months = UNIMODAL_RAIN_MONTHS if rain_pattern == "unimodal" else BIMODAL_RAIN_MONTHS

    def total(days: int) -> float:
        return round(sum(p for d, p in rain if (today - d).days < days), 1)

    wet_days = [d for d, p in rain if p >= 1.0]
    days_since_rain = (today - wet_days[-1]).days if wet_days else len(rain)
    dry_run = 0
    for _, p in reversed(rain):
        if p < 1.0:
            dry_run += 1
        else:
            break

    # Onset of the current rainy season, if its search window has opened
    onset = None
    season_name = None
    for name, (m, dd) in SEASON_SEARCH["unimodal" if rain_pattern == "unimodal" else "bimodal"].items():
        start = datetime.date(today.year, m, dd)
        if start <= today <= start + datetime.timedelta(days=150):
            season_name = name
            onset = rain_onset(rain, start, today - datetime.timedelta(days=3), check_days=min(30, max(0, (today - start).days)))
    # A "false start": a 20 mm burst in the last 45 days followed by 10+ dry days now
    burst = any(sum(p for d, p in rain[i:i + 3]) >= 20 and (today - rain[i][0]).days <= 45 for i in range(len(rain) - 2))
    false_start = bool(burst and dry_run >= 10 and today.month in rain_months)

    out = {
        "as_of": today.isoformat(),
        "rain_last_7_days_mm": total(7), "rain_last_30_days_mm": total(30),
        "days_since_last_rain": days_since_rain, "current_dry_spell_days": dry_run,
        "rainy_season_month": today.month in rain_months,
        "season": season_name, "season_onset_date": onset.isoformat() if onset else None,
        "possible_false_start": false_start,
        "heaviest_day_last_7": max(((d.isoformat(), p) for d, p in rain if (today - d).days < 7), key=lambda x: x[1], default=None),
    }
    if rain_normal_monthly_mm:
        month_key = MONTHS[today.month - 1]
        normal_30 = rain_normal_monthly_mm.get(month_key)
        if normal_30:
            out["rain_30_vs_normal_pct"] = round((out["rain_last_30_days_mm"] - normal_30) / normal_30 * 100)
    tmax7 = [tx for d, tx, _, _ in past if tx is not None and (today - d).days < 7]
    if tmax7:
        out["avg_max_last_7_c"] = round(statistics.fmean(tmax7), 1)
        if tmax_normal_monthly_c and MONTHS[today.month - 1] in tmax_normal_monthly_c:
            out["heat_vs_normal_c"] = round(out["avg_max_last_7_c"] - tmax_normal_monthly_c[MONTHS[today.month - 1]], 1)
    out["alerts"] = weather_alerts(out)
    return out


def weather_alerts(status: dict) -> List[dict]:
    """Plain alerts a farmer should hear about, most urgent first."""
    alerts = []
    heavy = status.get("heaviest_day_last_7")
    if heavy and heavy[1] >= 50:
        alerts.append({"key": f"heavy:{heavy[0]}", "level": "warning",
                       "text": f"Heavy rain: {round(heavy[1])} mm fell on {heavy[0]}. Check for flooding and move animals off low ground."})
    if status.get("possible_false_start"):
        alerts.append({"key": f"falsestart:{status['as_of'][:7]}", "level": "warning",
                       "text": f"False start: rain came, but it has now been dry for {status['current_dry_spell_days']} days. Hold off planting pasture or crops until the rains settle."})
    elif status.get("rainy_season_month") and status.get("current_dry_spell_days", 0) >= 10:
        alerts.append({"key": f"dryspell:{status['as_of'][:7]}", "level": "warning",
                       "text": f"Dry spell: no useful rain for {status['current_dry_spell_days']} days during the rainy season. Save water and stored feed."})
    if status.get("heat_vs_normal_c") is not None and status["heat_vs_normal_c"] >= 2:
        alerts.append({"key": f"heat:{status['as_of'][:10]}", "level": "advice",
                       "text": f"Hot spell: afternoons {status['heat_vs_normal_c']} °C above normal this week. Give shade and extra water; milk and move animals early."})
    if status.get("season_onset_date"):
        alerts.append({"key": f"onset:{status['season']}:{status['season_onset_date'][:4]}", "level": "info",
                       "text": f"The rains have started (from {status['season_onset_date']}). A good time to plant pasture and fodder."})
    pct = status.get("rain_30_vs_normal_pct")
    if pct is not None and pct <= -40 and status.get("rainy_season_month"):
        alerts.append({"key": f"deficit:{status['as_of'][:7]}", "level": "warning",
                       "text": f"Much drier than normal: the last 30 days brought {abs(pct)}% less rain than usual. Plan feed and water."})
    return alerts


# ---------------------------------------------------------------------------
# How the weather at THIS farm has changed over the decades
# ---------------------------------------------------------------------------

CLIMATE_START_YEAR = 1981      # CHIRPS daily rainfall starts in 1981
BASELINE_YEARS = 30            # "current climate" = the most recent 30 complete years (WMO normal length)
WET_MONTH_MM = 60.0            # a month whose normal rain is >= 60 mm counts as wet (the tropical wet-month line used by Koppen)
RAINY_DAY_MM = 1.0


def wet_months(rain_monthly_normal: Sequence[dict], threshold_mm: float = WET_MONTH_MM) -> List[int]:
    """Calendar months (1-12) whose normal rainfall at this farm is at least ``threshold_mm``."""
    return [MONTHS.index(r["month"]) + 1 for r in rain_monthly_normal if r.get("mean", 0) >= threshold_mm]


def _longest_dry(values: Sequence[float], dry_day_mm: float = RAINY_DAY_MM) -> int:
    run = longest = 0
    for v in values:
        run = run + 1 if v < dry_day_mm else 0
        longest = max(longest, run)
    return longest


def _false_starts(series: Sequence[Tuple[datetime.date, float]], start: datetime.date, end: datetime.date) -> int:
    """Wet bursts (>= 20 mm in 3 days) inside the season window followed by >= 10 dry days within 30 days."""
    rain = [v for _, v in series]
    days = [d for d, _ in series]
    count, i = 0, 0
    while i < len(series) - 3:
        if start <= days[i] <= end and sum(rain[i:i + 3]) >= 20 and _longest_dry(rain[i + 3:i + 33]) >= 10:
            count += 1
            i += 33
        else:
            i += 1
    return count


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _cv(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if len(xs) < 3 or not statistics.fmean(xs):
        return None
    return statistics.pstdev(xs) / statistics.fmean(xs) * 100


def climate_changes(rain_daily: Sequence[Tuple[datetime.date, float]],
                    tmax_daily: Sequence[Tuple[datetime.date, float]],
                    seasons: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]],
                    today: Optional[datetime.date] = None,
                    early: Tuple[int, int] = (1981, 2000), recent_years: int = 20) -> dict:
    """Per-year rain, rainy days, season start, dry spells, false starts and heat at one location,
    and how the most recent ``recent_years`` complete years differ from ``early``.

    ``seasons`` = {name: ((start_month, start_day), (end_month, end_day))}, the window in which each
    season's rains are looked for (months 1-12). Returns the numbers and plain sentences that only
    report changes large enough to matter on a farm.
    """
    today = today or datetime.date.today()
    last = today.year - 1
    recent = (last - recent_years + 1, last)
    series = sorted((d, float(v or 0.0)) for d, v in rain_daily if d.year <= last)
    by_year: Dict[int, List[Tuple[datetime.date, float]]] = {}
    for d, v in series:
        by_year.setdefault(d.year, []).append((d, v))
    tmax_year: Dict[int, List[float]] = {}
    for d, v in tmax_daily:
        if v is not None and d.year <= last:
            tmax_year.setdefault(d.year, []).append(float(v))

    years = {}
    for y, days in by_year.items():
        if len(days) < 360:            # incomplete year
            continue
        row = {"rain_mm": sum(v for _, v in days), "rainy_days": sum(1 for _, v in days if v >= RAINY_DAY_MM),
               "tmax_c": _mean(tmax_year.get(y, [])) if len(tmax_year.get(y, [])) >= 360 else None}
        for name, ((sm, sd), (em, ed)) in seasons.items():
            s, e = datetime.date(y, sm, sd), datetime.date(y, em, min(ed, 28 if em == 2 else 30 if em in (4, 6, 9, 11) else 31))
            win = [(d, v) for d, v in series if s <= d <= e + datetime.timedelta(days=40)]
            onset = rain_onset(win, s, e)
            row[f"{name}_onset_doy"] = onset.timetuple().tm_yday if onset else None
            season_days = [v for d, v in series if onset <= d <= onset + datetime.timedelta(days=90)] if onset else []
            row[f"{name}_longest_dry_spell"] = _longest_dry(season_days) if season_days else None
            row[f"{name}_false_starts"] = _false_starts(win, s, e)
            row[f"{name}_rain_mm"] = sum(season_days) if season_days else None
        years[y] = row

    def period(lo, hi, key):
        return [r[key] for y, r in years.items() if lo <= y <= hi and r.get(key) is not None]

    def compare(key, kind="mean"):
        a, b = period(*early, key), period(*recent, key)
        if len(a) < 8 or len(b) < 8:
            return None
        f = {"mean": _mean, "cv": _cv, "sd": lambda xs: statistics.pstdev(xs) if len(xs) >= 3 else None}[kind]
        ea, re_ = f(a), f(b)
        out = {"early": round(ea, 1) if ea is not None else None, "recent": round(re_, 1) if re_ is not None else None,
               "change": round(re_ - ea, 1) if None not in (ea, re_) else None}
        out["significant"] = _significant(a, b, kind)
        return out

    out = {"early_period": f"{early[0]}-{early[1]}", "recent_period": f"{recent[0]}-{recent[1]}",
           "years_analysed": len(years),
           "annual_rain_mm": compare("rain_mm"), "rainy_days": compare("rainy_days"),
           "annual_rain_variability_pct": compare("rain_mm", "cv"), "tmax_c": compare("tmax_c"),
           "trend_per_decade": {k: trend_per_decade({y: r[k] for y, r in years.items() if r.get(k) is not None})
                                for k in ("rain_mm", "rainy_days", "tmax_c")},
           "seasons": {}}
    for name in seasons:
        out["seasons"][name] = {
            "onset_doy": compare(f"{name}_onset_doy"),
            "onset_spread_days": compare(f"{name}_onset_doy", "sd"),
            "longest_dry_spell_days": compare(f"{name}_longest_dry_spell"),
            "false_starts_per_year": compare(f"{name}_false_starts"),
            "season_rain_mm": compare(f"{name}_rain_mm"),
            "season_rain_variability_pct": compare(f"{name}_rain_mm", "cv"),
        }
    out["statements"] = change_statements(out)
    return out


T_CRIT = 2.7     # Welch t, about the 1% level for 20 v 20 years (several checks per farm, so stricter than 5%)


def _significant(a: Sequence[float], b: Sequence[float], kind: str) -> bool:
    """Is the difference between the early and recent years more than year-to-year noise?"""
    if kind == "mean":
        va, vb = statistics.variance(a), statistics.variance(b)
        se = (va / len(a) + vb / len(b)) ** 0.5
        if se == 0:
            return statistics.fmean(a) != statistics.fmean(b)
        return abs(statistics.fmean(b) - statistics.fmean(a)) / se >= T_CRIT
    if kind == "cv":
        a = [x / statistics.fmean(a) for x in a] if statistics.fmean(a) else a
        b = [x / statistics.fmean(b) for x in b] if statistics.fmean(b) else b
    # Brown-Forsythe: compare distances from each period's median (robust when years are skewed,
    # as season start dates are), with the same t threshold, one-sided (more spread recently).
    da = [abs(x - statistics.median(a)) for x in a]
    db = [abs(x - statistics.median(b)) for x in b]
    se = (statistics.variance(da) / len(da) + statistics.variance(db) / len(db)) ** 0.5
    return se > 0 and (statistics.fmean(db) - statistics.fmean(da)) / se >= T_CRIT


SEASON_WORDS = {"long_rains": "long rains", "short_rains": "short rains", "rains": "rains"}


def change_statements(c: dict) -> List[str]:
    """Plain sentences for changes that are both statistically clear (``_significant``) and big enough to matter (rain +-10%, start +-7 days, dry spells +-3 days,
    variability +20% relative, start-date spread +4 days and +20%, heat +-0.3 C); nothing is said about smaller differences."""
    s: List[str] = []
    span = f"{c['recent_period']} compared with {c['early_period']}"
    r = c.get("annual_rain_mm")
    if r and r["early"] and abs(r["change"]) / r["early"] >= 0.10 and r["significant"]:
        s.append(f"Yearly rain is {abs(round(r['change'] / r['early'] * 100))}% {'higher' if r['change'] > 0 else 'lower'} ({span}).")
    rd = c.get("rainy_days")
    if rd and rd["early"] and abs(rd["change"]) / rd["early"] >= 0.10 and rd["significant"]:
        s.append(f"There are about {abs(round(rd['change']))} {'more' if rd['change'] > 0 else 'fewer'} rainy days a year, "
                 f"so rain comes in {'more' if rd['change'] > 0 else 'fewer, heavier'} showers.")
    v = c.get("annual_rain_variability_pct")
    if v and v["early"] and (v["recent"] - v["early"]) / v["early"] >= 0.20 and v["significant"]:
        s.append("Rain has become more irregular from year to year: good and bad years are further apart than before.")
    for name, sc in c.get("seasons", {}).items():
        word = SEASON_WORDS.get(name, name)
        o = sc.get("onset_doy")
        if o and o["change"] is not None and abs(o["change"]) >= 7 and o["significant"]:
            s.append(f"The {word} now start about {abs(round(o['change']))} days {'later' if o['change'] > 0 else 'earlier'} than they used to.")
        sp = sc.get("onset_spread_days")
        if sp and sp["early"] and sp["recent"] - sp["early"] >= max(4.0, 0.20 * sp["early"]) and sp["significant"]:
            s.append(f"The start of the {word} has become harder to predict from year to year.")
        dsp = sc.get("longest_dry_spell_days")
        if dsp and dsp["change"] is not None and abs(dsp["change"]) >= 3 and dsp["significant"]:
            s.append(f"Dry spells inside the {word} are about {abs(round(dsp['change']))} days {'longer' if dsp['change'] > 0 else 'shorter'}.")
        fs = sc.get("false_starts_per_year")
        if fs and fs["change"] is not None and fs["change"] >= 0.3 and fs["significant"]:
            s.append(f"False starts of the {word} (rain, then 10+ dry days) happen more often than before.")
    t = c.get("tmax_c")
    if t and t["change"] is not None and abs(t["change"]) >= 0.3 and t["significant"]:
        s.append(f"Afternoons are about {abs(t['change']):.1f} °C {'hotter' if t['change'] > 0 else 'cooler'} on average.")
    return s
