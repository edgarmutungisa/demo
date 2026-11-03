"""DST Uganda prediction engine: pure, tested calculations.

Every prediction product turns inputs (20-year history, the latest
observations, and a 51-member ECMWF seasonal ensemble) into PROBABILITIES,
a confidence level and a recommended action, and can later be verified.

Products (see docs/PREDICTION_DESIGN.md):
  P1 season rainfall outlook   - tercile probabilities per month and season
  P2 onset of the rains        - probability the rains start by each date
  P3 pasture outlook           - grass condition 1-3 months ahead (per-farm model)
  P4 feed balance              - month-by-month feed surplus / gap for the herd
  P5 disease risk outlook      - risk index for disease groups, next weeks
  P6 market price outlook      - seasonal price index from reporter history
  +  verification              - Brier score, RPSS, hit rates, bias learning
"""

from __future__ import annotations

import datetime
import math
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_ORDER = ["below_normal", "near_normal", "above_normal"]


def terciles(values: Sequence[float]) -> Tuple[float, float]:
    """Lower and upper tercile boundaries of a climatology."""
    if len(values) < 6:
        raise ValueError("need at least 6 years to define terciles")
    q = statistics.quantiles(sorted(values), n=3, method="inclusive")
    return q[0], q[1]


def tercile_probabilities(members: Sequence[float], lower: float, upper: float) -> Dict[str, float]:
    """Share of ensemble members below / within / above the climatological terciles."""
    if not members:
        raise ValueError("no ensemble members")
    n = len(members)
    below = sum(1 for m in members if m < lower) / n
    above = sum(1 for m in members if m > upper) / n
    return {"below_normal": round(below, 2), "near_normal": round(1 - below - above, 2), "above_normal": round(above, 2)}


def headline(probs: Dict[str, float]) -> str:
    """Plain-language headline for tercile probabilities (equal odds = 33% each)."""
    cat, p = max(probs.items(), key=lambda kv: kv[1])
    words = {"below_normal": "drier than normal", "near_normal": "about normal", "above_normal": "wetter than normal"}
    if p < 0.40:
        return "no clear signal (similar chances of dry, normal or wet)"
    strength = "likely" if p < 0.55 else "very likely" if p >= 0.70 else "probably"
    return f"{strength} {words[cat]} ({round(p * 100)}% chance)"


def confidence(spread_ratio: float, verified_skill: Optional[float] = None) -> str:
    """'high' / 'medium' / 'low' from ensemble agreement, capped at 'medium' until local skill is verified."""
    level = 2 if spread_ratio < 0.35 else 1 if spread_ratio < 0.7 else 0
    if verified_skill is None:          # members agreeing is not accuracy: "high" needs verified local skill
        level = min(level, 1)
    else:
        level = min(level, 2 if verified_skill >= 0.3 else 1 if verified_skill >= 0.1 else 0)
    return ["low", "medium", "high"][level]


def apply_bias(members: Sequence[float], factor: Optional[float]) -> List[float]:
    """Multiplicative rainfall bias correction learned from verification (1.0 = none)."""
    if not factor or factor <= 0:
        return list(members)
    return [m * factor for m in members]


# ---------------------------------------------------------------------------
# P1 Season rainfall outlook
# ---------------------------------------------------------------------------


def season_outlook(member_daily: Dict[str, List[Tuple[datetime.date, float]]],
                   history_monthly: Dict[Tuple[int, int], float],
                   bias_factor: Optional[float] = None,
                   verified_skill: Optional[float] = None) -> dict:
    """Monthly and 3-month tercile probabilities from ensemble members vs the farm's own history.

    member_daily: {member_id: [(date, mm), ...]} from the seasonal ensemble.
    history_monthly: {(year, month): total mm} observed over 20+ years.
    """
    if len(member_daily) < 10:
        raise ValueError("need at least 10 ensemble members")
    # monthly totals per member (only complete months)
    per_member: Dict[str, Dict[Tuple[int, int], float]] = {}
    for mid, series in member_daily.items():
        tot: Dict[Tuple[int, int], List[float]] = {}
        for d, v in series:
            tot.setdefault((d.year, d.month), []).append(float(v or 0.0))
        per_member[mid] = {k: sum(v) for k, v in tot.items() if len(v) >= 28}
    months = sorted(set.intersection(*[set(m) for m in per_member.values()]))
    out_months = []
    for (y, m) in months[:6]:
        hist = [v for (yy, mm), v in history_monthly.items() if mm == m]
        if len(hist) < 6:
            continue
        lo, hi = terciles(hist)
        members = apply_bias([per_member[k][(y, m)] for k in per_member], bias_factor)
        probs = tercile_probabilities(members, lo, hi)
        med = statistics.median(members)
        normal = statistics.fmean(hist)
        spread = (statistics.quantiles(members, n=4)[2] - statistics.quantiles(members, n=4)[0]) / normal if normal else 1.0
        out_months.append({"month": f"{y}-{m:02d}", "probabilities": probs, "headline": headline(probs),
                           "forecast_median_mm": round(med), "normal_mm": round(normal),
                           "likely_range_mm": [round(statistics.quantiles(members, n=10)[0]), round(statistics.quantiles(members, n=10)[-1])],
                           "confidence": confidence(spread, verified_skill)})
    season = None
    if len(out_months) >= 3:
        keys = months[:3]
        hist3 = []
        years = sorted({yy for (yy, _) in history_monthly})
        for yy in years:
            vals = []
            for (y, m) in keys:
                yr = yy + (y - keys[0][0])
                if (yr, m) in history_monthly:
                    vals.append(history_monthly[(yr, m)])
            if len(vals) == 3:
                hist3.append(sum(vals))
        if len(hist3) >= 6:
            lo, hi = terciles(hist3)
            members3 = apply_bias([sum(per_member[k][km] for km in keys) for k in per_member], bias_factor)
            probs = tercile_probabilities(members3, lo, hi)
            season = {"months": [f"{y}-{m:02d}" for (y, m) in keys], "probabilities": probs, "headline": headline(probs),
                      "forecast_median_mm": round(statistics.median(members3)), "normal_mm": round(statistics.fmean(hist3))}
    return {"months": out_months, "next_3_months": season, "members": len(per_member),
            "calibration": "bias-corrected from verified forecasts" if bias_factor else "raw ECMWF ensemble vs local 20-year terciles (not yet calibrated locally)"}


def _month_days(y: int, m: int) -> int:
    return ((datetime.date(y + m // 12, m % 12 + 1, 1)) - datetime.date(y, m, 1)).days


def _same_day(d: datetime.date, years_back: int) -> datetime.date:
    y = d.year - years_back
    return d.replace(year=y, day=min(d.day, 28)) if (d.month == 2 and d.day == 29) else d.replace(year=y)


LOCAL_SKILL_MIN = 0.10   # on 40 rain series with no signal, the leave-one-year-out score never reached 0.10


def local_records_outlook(daily: Sequence[Tuple[datetime.date, float]], target_months: Sequence[Tuple[int, int]],
                          k: int = 8, min_cases: int = 15) -> Optional[dict]:
    """Forecast of the target months' rain from THIS farm's own daily records (analogue years).

    Predictors: rain in the last 30 days and in the 30 days before that, up to the last recorded day.
    The ``k`` past years whose same two windows looked most alike (z-scored distance) give the
    tercile probabilities (with one pseudo-count per tercile so 8 years never give 0% or 100%).
    The method is tested on every past year with that year left out (cross-validation), and the
    ranked probability skill score against equal odds (at least LOCAL_SKILL_MIN) says whether it can be trusted here, at this
    time of year. Returns None when the records are too short.
    """
    rain = {d: float(v or 0.0) for d, v in daily}
    if not rain or len(target_months) < 1:
        return None
    anchor = max(rain)
    first_year = min(rain).year

    def feats(back: int):
        a = _same_day(anchor, back)
        w = [rain.get(a - datetime.timedelta(days=i)) for i in range(60)]
        if any(v is None for v in w):
            return None
        return (sum(w[:30]), sum(w[30:]))

    def target(back: int):
        tot = 0.0
        for (y, m) in target_months:
            yy = y - back
            vals = [rain.get(datetime.date(yy, m, dd)) for dd in range(1, _month_days(yy, m) + 1)]
            if any(v is None for v in vals) or datetime.date(yy, m, 1) > anchor:
                return None
            tot += sum(vals)
        return tot

    cases = []
    for back in range(1, anchor.year - first_year + 1):
        f, t = feats(back), target(back)
        if f is not None and t is not None:
            cases.append({"year": anchor.year - back, "f": f, "t": t})
    now = feats(0)
    if now is None or len(cases) < min_cases:
        return None

    def predict(x, pool):
        n = len(pool)
        mu = [sum(c["f"][j] for c in pool) / n for j in (0, 1)]
        sd = [statistics.pstdev([c["f"][j] for c in pool]) or 1.0 for j in (0, 1)]
        dist = sorted(pool, key=lambda c: (sum(((c["f"][j] - x[j]) / sd[j]) ** 2 for j in (0, 1)), -c["year"]))
        near = dist[:k]
        lo, hi = terciles([c["t"] for c in pool])
        cnt = {"below_normal": 1.0, "near_normal": 1.0, "above_normal": 1.0}
        for c in near:
            cnt["below_normal" if c["t"] < lo else "above_normal" if c["t"] > hi else "near_normal"] += 1
        tot = sum(cnt.values())
        return {c: round(v / tot, 2) for c, v in cnt.items()}, near, (lo, hi)

    tested, hits = [], 0
    for i, c in enumerate(cases):
        pool = cases[:i] + cases[i + 1:]
        probs, _, (lo, hi) = predict(c["f"], pool)
        obs = "below_normal" if c["t"] < lo else "above_normal" if c["t"] > hi else "near_normal"
        tested.append((probs, obs))
        hits += max(probs, key=probs.get) == obs
    skill = rpss(tested)
    probs, near, _ = predict(now, cases)
    return {"probabilities": probs, "headline": headline(probs),
            "rain_last_30_days_mm": round(now[0]), "rain_30_to_60_days_ago_mm": round(now[1]), "records_to": anchor.isoformat(),
            "analogue_years": [{"year": c["year"], "then_last_60_days_mm": round(sum(c["f"])), "target_months_mm": round(c["t"])} for c in near],
            "skill": {"years_tested": len(tested), "hits": hits, "rpss": skill, "hit_rate_pct": round(100 * hits / len(tested)), "chance_hit_rate_pct": 33},
            "usable": skill is not None and skill >= LOCAL_SKILL_MIN,
            "method": f"{k} past years at this farm whose previous 60 days of rain were most alike; tested on {len(tested)} past years with each year left out"}


def combine_outlooks(ecmwf: Optional[Dict[str, float]], local: Optional[dict]) -> Optional[dict]:
    """One answer from the ECMWF ensemble and the farm's own records.

    The farm's records are only used when they clearly beat equal odds in the leave-one-year-out
    test (rpss >= LOCAL_SKILL_MIN); then both get equal weight, because neither has yet been verified here against the
    other. Otherwise the ECMWF probabilities stand alone.
    """
    if ecmwf is None and not (local and local["usable"]):
        return None
    if ecmwf is None:
        probs, basis = dict(local["probabilities"]), "the farm's own records only (the ECMWF forecast was unavailable)"
    elif local and local["usable"]:
        probs = {c: round((ecmwf[c] + local["probabilities"][c]) / 2, 2) for c in _ORDER}
        basis = "ECMWF ensemble and the farm's own records, equal weight"
    else:
        probs, basis = dict(ecmwf), ("ECMWF ensemble only: the farm's own records had no skill for these months in the past-year test"
                                     if local else "ECMWF ensemble only: the farm's records are too short to test")
    agree = None
    if ecmwf is not None and local:
        agree = max(ecmwf, key=ecmwf.get) == max(local["probabilities"], key=local["probabilities"].get)
    return {"probabilities": probs, "headline": headline(probs), "basis": basis, "sources_agree": agree}


# ---------------------------------------------------------------------------
# P2 Onset of the rains
# ---------------------------------------------------------------------------


def season_target(today: datetime.date, windows: Dict[str, Tuple[int, int]], length_days: int = 120):
    """(name, start, end) of the rainy season worth forecasting: one whose onset window opened
    in the last 45 days (its start may still be pending), otherwise the next one to open."""
    candidates = []
    for year in (today.year - 1, today.year, today.year + 1):
        for name, (m, d) in windows.items():
            start = datetime.date(year, m, d)
            end = start + datetime.timedelta(days=length_days)
            if end >= today and (start >= today or (today - start).days <= 45):
                candidates.append((start, end, name))
    start, end, name = min(candidates)
    return name, start, end


def _onset(series: Sequence[Tuple[datetime.date, float]], start: datetime.date, end: datetime.date) -> Optional[datetime.date]:
    s = sorted(series)
    rain = [float(v or 0) for _, v in s]
    for i, (d, _) in enumerate(s):
        if d < start or d > end or i + 3 > len(s):
            continue
        if sum(rain[i:i + 3]) < 20:
            continue
        window = rain[i + 3:i + 33]
        run = longest = 0
        for v in window:
            run = run + 1 if v < 1 else 0
            longest = max(longest, run)
        if longest < 10:
            return d
    return None


def onset_forecast(member_daily: Dict[str, List[Tuple[datetime.date, float]]],
                   observed: Sequence[Tuple[datetime.date, float]],
                   season_start: datetime.date, season_end: datetime.date,
                   usual_onset_doy: Optional[int] = None, bias_factor: Optional[float] = None) -> dict:
    """Probability distribution of the onset date, joining what has already fallen with each member's future."""
    today = max(d for d, _ in observed) if observed else season_start
    obs = [(d, v) for d, v in observed if d >= season_start - datetime.timedelta(days=5)]
    already = _onset(obs, season_start, today - datetime.timedelta(days=3)) if obs else None
    if already:
        return {"status": "started", "onset_date": already.isoformat()}
    dates = []
    for series in member_daily.values():
        fut = [(d, v * (bias_factor or 1.0)) for d, v in series if d > today]
        d = _onset(obs + fut, max(season_start, today - datetime.timedelta(days=2)), season_end)
        dates.append(d)
    n = len(dates)
    started = sorted(d for d in dates if d)
    p_fail = 1 - len(started) / n

    def p_by(day: datetime.date) -> float:
        return round(sum(1 for d in started if d <= day) / n, 2)

    milestones = []
    first = max(today, season_start)
    step = datetime.timedelta(days=7 if (season_end - first).days <= 56 else 14)
    probe = first + step
    while probe <= season_end and len(milestones) < 8:
        milestones.append({"by": probe.isoformat(), "probability": p_by(probe)})
        probe += step
    out = {"status": "not_started", "members": n, "probability_by_date": milestones,
           "probability_no_clear_start": round(p_fail, 2)}
    if started:
        doys = sorted(d.timetuple().tm_yday for d in started)
        q = statistics.quantiles(doys, n=4) if len(doys) >= 4 else [doys[0], statistics.median(doys), doys[-1]]
        base = datetime.date(today.year, 1, 1)
        to_date = lambda doy: (base + datetime.timedelta(days=int(round(doy)) - 1)).isoformat()
        out.update({"most_likely": to_date(statistics.median(doys)), "likely_window": [to_date(q[0]), to_date(q[-1])]})
        if usual_onset_doy:
            shift = statistics.median(doys) - usual_onset_doy
            out["versus_usual_days"] = round(shift)
            out["timing"] = "later than usual" if shift > 7 else "earlier than usual" if shift < -7 else "about the usual time"
    return out


# ---------------------------------------------------------------------------
# P3 Pasture outlook (per-farm statistical model, backtested)
# ---------------------------------------------------------------------------


def _ols(X: List[List[float]], y: List[float]) -> List[float]:
    """Least squares via normal equations (small k), with a tiny ridge for stability."""
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) + (1e-6 if i == j else 0) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(k)]
    for c in range(k):  # Gauss-Jordan
        piv = max(range(c, k), key=lambda r: abs(A[r][c]))
        A[c], A[piv], b[c], b[piv] = A[piv], A[c], b[piv], b[c]
        for r in range(k):
            if r != c and A[c][c]:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * cc for a, cc in zip(A[r], A[c])]
                b[r] -= f * b[c]
    return [b[i] / A[i][i] if A[i][i] else 0.0 for i in range(k)]


def pasture_model(monthly_ndvi_anom: Dict[Tuple[int, int], float], monthly_rain_anom: Dict[Tuple[int, int], float],
                  lead: int = 1) -> dict:
    """Fit NDVI anomaly(m+lead) = a + b*NDVI anom(m) + c*rain anom(m, already observed)
    + d*mean rain anom over the lead months (the forecast part), with a
    leave-one-year-out backtest. Anomalies are standardized (z-scores)."""
    def shift(y: int, m: int, k: int) -> Tuple[int, int]:
        t = y * 12 + (m - 1) + k
        return t // 12, t % 12 + 1

    rows = []
    for (y, m), nd in monthly_ndvi_anom.items():
        target = shift(y, m, lead)
        rains = [monthly_rain_anom.get(shift(y, m, k)) for k in range(1, lead + 1)]
        now_rain = monthly_rain_anom.get((y, m))
        if target in monthly_ndvi_anom and now_rain is not None and all(r is not None for r in rains):
            rows.append((y, [1.0, nd, now_rain, sum(rains) / lead], monthly_ndvi_anom[target]))
    if len(rows) < 36:
        raise ValueError("need at least 3 years of matched monthly data")
    coef = _ols([r[1] for r in rows], [r[2] for r in rows])
    hits = n = 0
    preds, obs = [], []
    for year in sorted({r[0] for r in rows}):
        train = [r for r in rows if r[0] != year]
        test = [r for r in rows if r[0] == year]
        c = _ols([r[1] for r in train], [r[2] for r in train])
        for _, x, yy in test:
            p = sum(ci * xi for ci, xi in zip(c, x))
            preds.append(p); obs.append(yy)
            n += 1
            hits += (p < -0.43) == (yy < -0.43)  # same call on "below normal" (lower tercile of a normal)
    corr = statistics.correlation(preds, obs) if len(preds) > 2 else 0.0
    return {"coefficients": [round(c, 3) for c in coef], "lead_months": lead, "samples": len(rows),
            "backtest_correlation": round(corr, 2), "backtest_hit_rate": round(hits / n, 2)}


def predict_pasture(model: dict, ndvi_anom_now: float, rain_anom_members: Sequence[float],
                    rain_anom_now: float = 0.0) -> dict:
    """Probability the grass will be below / near / above normal `lead` months ahead.

    rain_anom_members: each ensemble member's mean rain anomaly (z) over the lead months.
    rain_anom_now: this month's observed rain anomaly (z) so far."""
    a, b, c, d = model["coefficients"]
    preds = [a + b * ndvi_anom_now + c * rain_anom_now + d * r for r in rain_anom_members]
    # residual uncertainty from backtest skill (std of a z-score not explained)
    resid = math.sqrt(max(0.05, 1 - model["backtest_correlation"] ** 2))
    def cdf(x):  # normal CDF
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    pb = statistics.fmean(cdf((-0.43 - p) / resid) for p in preds)
    pa = statistics.fmean(1 - cdf((0.43 - p) / resid) for p in preds)
    probs = {"below_normal": round(pb, 2), "near_normal": round(1 - pb - pa, 2), "above_normal": round(pa, 2)}
    words = {"below_normal": "thinner than normal", "near_normal": "about normal", "above_normal": "better than normal"}
    cat = max(probs, key=probs.get)
    skill = model["backtest_correlation"]
    return {"lead_months": model["lead_months"], "probabilities": probs, "expected_anomaly_z": round(statistics.fmean(preds), 2),
            "headline": f"grass likely {words[cat]} ({round(probs[cat] * 100)}% chance)" if probs[cat] >= 0.4 else "no clear signal for the grass",
            "confidence": "high" if skill >= 0.6 else "medium" if skill >= 0.35 else "low",
            "model_skill": {"correlation": skill, "hit_rate": model["backtest_hit_rate"]}}


# ---------------------------------------------------------------------------
# P4 Feed balance
# ---------------------------------------------------------------------------


def feed_balance(months: Sequence[str], forage_t_dm: Sequence[Tuple[float, float, float]], cattle: int, avg_weight_kg: float,
                 grazing_loss: float = 0.5, carryover: float = 0.7, opening_stock_t: float = 0.0) -> dict:
    """Month-by-month feed balance for the herd.

    forage_t_dm: per month (low, likely, high) forage production in t dry matter.
    Demand: 2.5% of live weight per day; half of the grass is assumed lost to
    trampling (the LAPIG rule), so usable forage = production * (1 - loss).
    Standing grass not eaten carries over to the next month, losing
    (1 - carryover) of its value each month (drying and quality loss).
    """
    if not (cattle > 0 and avg_weight_kg > 50):
        raise ValueError("cattle and weight are required")
    demand = cattle * avg_weight_kg * 0.025 * 30.4 / 1000  # t DM per month
    rows, gap_months, gap_total = [], [], 0.0
    stock = opening_stock_t
    for m, (lo, mid, hi) in zip(months, forage_t_dm):
        carried = stock * carryover
        usable = mid * (1 - grazing_loss) + carried
        bal = usable - demand
        stock = max(0.0, bal)
        lo, mid, hi = lo + carried / (1 - grazing_loss), mid + carried / (1 - grazing_loss), hi + carried / (1 - grazing_loss)
        p_gap = 0.9 if hi * (1 - grazing_loss) < demand else 0.6 if mid * (1 - grazing_loss) < demand else 0.25 if lo * (1 - grazing_loss) < demand else 0.05
        rows.append({"month": m, "usable_forage_t": round(usable, 2), "herd_demand_t": round(demand, 2),
                     "balance_t": round(bal, 2), "probability_of_gap": p_gap})
        if bal < 0:
            gap_months.append(m)
            gap_total += -bal
    first_gap = gap_months[0] if gap_months else None
    hay_bales = math.ceil(gap_total * 1000 / 0.85 / 20) if gap_total else 0  # ~20 kg small bales, 85% DM
    sustainable_head = max(0, int(min((mid * (1 - grazing_loss)) for _, mid, _ in forage_t_dm) / (avg_weight_kg * 0.025 * 30.4 / 1000))) if forage_t_dm else 0
    return {"months": rows, "first_gap_month": first_gap, "total_gap_t_dm": round(gap_total, 2),
            "hay_needed_small_bales": hay_bales, "head_the_worst_month_can_feed": sustainable_head,
            "advice": (f"Feed gap expected from {first_gap}: store about {round(gap_total / 0.85, 1)} t of hay (≈{hay_bales} small bales) "
                       f"or reduce to about {sustainable_head} animal{'s' if sustainable_head != 1 else ''} before then." if first_gap else "No feed gap expected over these months.")}


# ---------------------------------------------------------------------------
# P5 Disease risk outlook (index, not a diagnosis)
# ---------------------------------------------------------------------------


def disease_risk(weekly: Sequence[dict], calendar_peak: Dict[str, str]) -> List[dict]:
    """Risk level per disease group for each coming week.

    weekly: [{"week_start": iso, "rain_mm": x, "tmean_c": t, "rain_days": n}] from the forecast.
    calendar_peak: {"tick_borne": "wet", "lsd": "wet", "fmd_cbpp": "dry", ...} from the livestock guide.
    """
    out = []
    for w in weekly:
        wet = w["rain_mm"] >= 15 or w.get("rain_days", 0) >= 3
        warm = 18 <= w["tmean_c"] <= 30
        scores = {
            "tick_borne": 3 if (wet and warm) else 2 if (wet or warm) and calendar_peak.get("tick_borne") == "wet" else 1,
            "lumpy_skin_and_tryps": 3 if wet and w["tmean_c"] >= 20 else 2 if wet else 1,
            "worms_and_fluke": 3 if w["rain_mm"] >= 30 else 2 if wet else 1,
            "fmd_cbpp_movement": 3 if (not wet and calendar_peak.get("fmd_cbpp") == "dry") else 1,
        }
        label = {1: "low", 2: "medium", 3: "high"}
        out.append({"week_start": w["week_start"], **{k: label[v] for k, v in scores.items()}})
    return out


# ---------------------------------------------------------------------------
# P6 Market price outlook
# ---------------------------------------------------------------------------


def price_outlook(monthly_median_ugx: Dict[Tuple[int, int], float], horizon: int = 3,
                  drought_probability: Optional[float] = None, today: Optional[datetime.date] = None) -> dict:
    """Seasonal price index from at least 12 months of reports; never guesses without history."""
    if len({(y, m) for (y, m) in monthly_median_ugx}) < 12:
        return {"status": "insufficient_history", "months_available": len(monthly_median_ugx),
                "note": "Price predictions start after 12 months of market reports."}
    overall = statistics.fmean(monthly_median_ugx.values())
    index = {}
    for m in range(1, 13):
        vals = [v for (y, mm), v in monthly_median_ugx.items() if mm == m]
        if vals:
            index[m] = statistics.fmean(vals) / overall
    today = today or datetime.date.today()
    latest_key = max(monthly_median_ugx)
    latest = monthly_median_ugx[latest_key]
    base_idx = index.get(latest_key[1], 1.0)
    out = []
    for k in range(1, horizon + 1):
        t = today.year * 12 + today.month - 1 + k
        y, m = t // 12, t % 12 + 1
        factor = index.get(m, 1.0) / base_idx
        if drought_probability and drought_probability >= 0.5:
            factor *= 1 - 0.1 * drought_probability  # distress sales push prices down in drought years
        out.append({"month": f"{y}-{m:02d}", "expected_ugx": int(round(latest * factor, -4)), "change_pct": round((factor - 1) * 100)})
    best = max(out, key=lambda r: r["expected_ugx"])
    return {"status": "ok", "months": out, "best_month_to_sell": best["month"],
            "note": "Seasonal pattern from past reports; prices on the day depend on weight and condition."}


# ---------------------------------------------------------------------------
# Verification and learning
# ---------------------------------------------------------------------------

def rps(probs: Dict[str, float], observed: str) -> float:
    """Ranked probability score (0 = perfect) for a tercile forecast."""
    cf = co = s = 0.0
    for cat in _ORDER:
        cf += probs.get(cat, 0.0)
        co += 1.0 if cat == observed else 0.0
        s += (cf - co) ** 2
    return s / (len(_ORDER) - 1)


def rpss(forecasts: Iterable[Tuple[Dict[str, float], str]]) -> Optional[float]:
    """Skill versus always forecasting 1/3-1/3-1/3 (climatology). >0 means better than climatology."""
    pairs = list(forecasts)
    if len(pairs) < 3:
        return None
    clim = {c: 1 / 3 for c in _ORDER}
    rf = statistics.fmean(rps(p, o) for p, o in pairs)
    rc = statistics.fmean(rps(clim, o) for _, o in pairs)
    return round(1 - rf / rc, 2) if rc else None


def brier(prob: float, happened: bool) -> float:
    return (prob - (1.0 if happened else 0.0)) ** 2


def onset_error_days(predicted_iso: str, observed_iso: str) -> int:
    return (datetime.date.fromisoformat(observed_iso) - datetime.date.fromisoformat(predicted_iso)).days


def learn_bias(pairs: Iterable[Tuple[float, float]]) -> Optional[float]:
    """Multiplicative rainfall bias from (forecast median, observed) pairs; needs 3+ and is clamped to 0.5-2."""
    pts = [(f, o) for f, o in pairs if f and f > 0]
    if len(pts) < 3:
        return None
    return round(min(2.0, max(0.5, sum(o for _, o in pts) / sum(f for f, _ in pts))), 2)
