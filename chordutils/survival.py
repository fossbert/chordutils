"""Survival analysis helpers that respect the MSK-CHORD design (left truncation, immortal time).

- ``km_estimate`` / ``km_median`` / ``summarize_endpoint``: Kaplan-Meier with optional delayed
  entry (numpy only; agrees with lifelines, see tests).
- ``exposure_days`` / ``to_counting_process`` / ``landmark``: effect of an exposure that starts
  during follow-up (e.g. radiation) without immortal time bias. Comparing "ever exposed" vs.
  "never exposed" from the origin credits the exposed group with the time they had to
  survive to be exposed; the counting-process format lets the exposure switch on at the day
  it happens, a landmark analysis fixes the exposure status at a chosen day.
- ``association_screen``: univariable (optionally adjusted) Cox models for many covariates,
  fitted with lifelines.

Adapted from earlier PDAC analysis modules (chordendpoints, chordcovariates).
"""

from statistics import NormalDist
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .endpoints import DAYS_PER_MONTH

ID = "PATIENT_ID"

Z95 = NormalDist().inv_cdf(0.975)  # 1.959963..., as in lifelines and R


def km_estimate(time, event, entry=None) -> pd.DataFrame:
    """Kaplan-Meier estimator with optional delayed entry (left truncation).

    A subject is at risk at time t if ``entry < t <= time``. Confidence limits are 95%
    log-log (Greenwood variance), as in lifelines and R ``survfit(conf.type='log-log')``;
    they are undefined (NaN) where the survival estimate is 0.

    Returns
    -------
    pd.DataFrame
        time, n_risk, n_event, surv, lower, upper at every event time.
    """

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    entry = np.zeros_like(time) if entry is None else np.asarray(entry, dtype=float)

    ts = np.unique(time[event == 1])
    n_risk = np.array([np.sum((entry < t) & (time >= t)) for t in ts])
    n_ev = np.array([np.sum((time == t) & (event == 1) & (entry < t)) for t in ts])
    ok = (n_risk > 0) & (n_ev > 0)
    ts, n_risk, n_ev = ts[ok], n_risk[ok], n_ev[ok]

    s = np.cumprod(1 - n_ev / n_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.cumsum(n_ev / (n_risk * (n_risk - n_ev)))
        se = np.sqrt(var) / np.abs(np.log(s))
        lower, upper = s ** np.exp(Z95 * se), s ** np.exp(-Z95 * se)

    return pd.DataFrame({"time": ts, "n_risk": n_risk, "n_event": n_ev, "surv": s, "lower": lower, "upper": upper})


def km_median(time, event, entry=None):
    """Median survival and its 95% CI (first time the curve / CI band reaches <= 0.5).

    Returns
    -------
    (median, lower, upper)
        NaN where not reached.
    """

    km = km_estimate(time, event, entry)

    def first_below(col):
        hit = km.loc[km[col] <= 0.5, "time"]
        return hit.iloc[0] if len(hit) else np.nan

    return first_below("surv"), first_below("lower"), first_below("upper")


def summarize_endpoint(d: pd.DataFrame, group_col: str, time_col: str, event_col: str,
                       entry_col: Optional[str] = None, min_n: int = 20, unit: str = "months") -> pd.DataFrame:
    """Median (95% CI) of an endpoint per group, e.g. TTD per regimen in the first line.

    Parameters
    ----------
    entry_col : str, optional
        Delayed-entry column (use 'ENTRY' for OS); None = no left truncation.
    min_n : int
        Groups with fewer rows are skipped.
    unit : {'months', 'days'}
        Months of 365/12 days.
    """

    f = 1 / DAYS_PER_MONTH if unit == "months" else 1
    rows = []
    for g, gd in d.groupby(group_col):
        if len(gd) < min_n:
            continue
        med, lo, hi = km_median(gd[time_col], gd[event_col], gd[entry_col] if entry_col else None)
        rows.append({group_col: g, "n": len(gd), "events": int(gd[event_col].sum()),
                     f"median_{unit}": med * f, "ci_lower": lo * f, "ci_upper": hi * f})
    return pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True) if rows else pd.DataFrame()


def exposure_days(events: pd.DataFrame, origins: pd.DataFrame, name: str, min_offset: int = 0,
                  time_col: str = "START_DATE", origin_col: str = "ORIGIN_DAY") -> pd.DataFrame:
    """First exposure event (e.g. radiation) per patient on or after origin + ``min_offset`` days.

    Returns
    -------
    pd.DataFrame
        PATIENT_ID and <name>_DAY (timeline day).
    """

    e = events[[ID, time_col]].merge(origins[[ID, origin_col]], on=ID)
    e = e[e[time_col] >= e[origin_col] + min_offset]
    return e.groupby(ID)[time_col].min().rename(f"{name}_DAY").reset_index()


def to_counting_process(base: pd.DataFrame, exposures: Dict[str, pd.DataFrame],
                        covariates: Sequence[str] = (), origin_col: str = "ORIGIN_DAY",
                        time_col: str = "OS_DAYS", event_col: str = "OS_EVENT",
                        entry_col: str = "ENTRY") -> pd.DataFrame:
    """Split each patient's follow-up at exposure times into (START, STOP] intervals.

    Parameters
    ----------
    base : pd.DataFrame
        One row per patient, e.g. ``endpoints.os_base`` output.
    exposures : dict of {name: DataFrame}
        Per exposure a table with PATIENT_ID and one timeline-day column (``exposure_days``
        output). The covariate <name> switches from 0 to 1 on that day and stays 1.
    covariates : sequence of str
        Baseline columns of ``base`` carried along unchanged.

    Returns
    -------
    pd.DataFrame
        PATIENT_ID, START, STOP, EVENT, one 0/1 column per exposure, covariates. The first
        interval starts at ENTRY (delayed entry); exposures before ENTRY count from the start.
        Zero-length intervals are removed. Use with lifelines ``CoxTimeVaryingFitter``
        (start_col='START', stop_col='STOP') or R ``coxph(Surv(START, STOP, EVENT) ~ ...)``.
    """

    d = base[[ID, origin_col, time_col, event_col, entry_col, *covariates]].copy()
    names = list(exposures)
    for n in names:
        ex = exposures[n]
        day_col = next(c for c in ex.columns if c != ID)
        d = d.merge(ex[[ID, day_col]].rename(columns={day_col: f"_x_{n}"}), on=ID, how="left")
        d[f"_x_{n}"] = d[f"_x_{n}"] - d[origin_col]

    rows = []
    for r in d.to_dict("records"):
        entry, stop, ev = r[entry_col], r[time_col], r[event_col]
        cuts = sorted({t for n in names if pd.notnull(t := r[f"_x_{n}"]) and entry < t < stop})
        bounds = [entry, *cuts, stop]
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            row = {ID: r[ID], "START": s, "STOP": e, "EVENT": int(ev == 1 and i == len(bounds) - 2)}
            row.update({n: int(pd.notnull(r[f"_x_{n}"]) and r[f"_x_{n}"] <= s) for n in names})
            row.update({c: r[c] for c in covariates})
            rows.append(row)

    out = pd.DataFrame(rows)
    return out[out["STOP"] > out["START"]].reset_index(drop=True)


def landmark(base: pd.DataFrame, exposure: pd.DataFrame, landmark_day: int, name: str = "EXPOSED",
             origin_col: str = "ORIGIN_DAY", time_col: str = "OS_DAYS", event_col: str = "OS_EVENT",
             entry_col: str = "ENTRY") -> pd.DataFrame:
    """Landmark data set: patients alive and under observation ``landmark_day`` days after origin.

    Exposure status is fixed at the landmark (1 if the exposure happened on or before it);
    time is counted from the landmark (LM_DAYS, LM_EVENT). Patients who entered the cohort
    (were sequenced) only after the landmark are excluded.
    """

    day_col = next(c for c in exposure.columns if c != ID)
    d = base.merge(exposure[[ID, day_col]], on=ID, how="left")
    d = d[(d[time_col] > landmark_day) & (d[entry_col] <= landmark_day)].copy()
    rel = d[day_col] - d[origin_col]
    d[name] = (rel.notna() & (rel <= landmark_day)).astype(int)
    d["LM_DAYS"] = d[time_col] - landmark_day
    d["LM_EVENT"] = d[event_col]
    return d.reset_index(drop=True)


def benjamini_hochberg(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values); NaN stays NaN."""

    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    n = ok.sum()
    if n == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    q_ok = np.empty(n)
    q_ok[order] = ranked
    q[ok] = q_ok
    return q


def association_screen(d: pd.DataFrame, endpoint: str, covariates: Sequence[str],
                       entry_col: Optional[str] = None, adjust: Sequence[str] = (),
                       min_events: int = 10, min_group: int = 10) -> pd.DataFrame:
    """Cox model per covariate (optionally adjusted) against one endpoint, with BH-FDR.

    Parameters
    ----------
    d : pd.DataFrame
        Analysis table with <endpoint>_DAYS and <endpoint>_EVENT (e.g. first-line table).
    endpoint : str
        Prefix, e.g. 'TTD', 'PFS', 'TTNT', 'OS'.
    covariates : sequence of str
        Numeric (continuous or 0/1) columns; dummy-code categorical variables first.
    entry_col : str, optional
        Delayed-entry column (OS only, see ``endpoints``).
    adjust : sequence of str
        Covariates added to every model (complete cases).
    min_events, min_group : int
        Covariates are skipped with fewer events, or for 0/1 covariates fewer patients in the
        smaller group.

    Returns
    -------
    pd.DataFrame
        covariate, n, events, HR, HR_lower, HR_upper, p, q_BH and, for 0/1 covariates, n and
        KM median (months) per group; sorted by p.
    """

    from lifelines import CoxPHFitter  # imported lazily: only needed here

    t_col, e_col = f"{endpoint}_DAYS", f"{endpoint}_EVENT"
    rows = []

    for cov in covariates:
        cols = [cov, *adjust]
        x = d.dropna(subset=cols + [t_col, e_col])
        x = x[x[t_col] > (x[entry_col] if entry_col else 0)]
        if x[e_col].sum() < min_events or x[cov].nunique() < 2:
            continue
        binary = set(pd.unique(x[cov])) <= {0, 1}
        if binary and min(x[cov].sum(), (1 - x[cov]).sum()) < min_group:
            continue

        fit_cols = cols + [t_col, e_col] + ([entry_col] if entry_col else [])
        try:
            cph = CoxPHFitter().fit(x[fit_cols].astype(float), duration_col=t_col, event_col=e_col,
                                    entry_col=entry_col)
        except Exception:  # convergence failure (e.g. complete separation)
            continue

        s = cph.summary.loc[cov]
        row = {"covariate": cov, "n": len(x), "events": int(x[e_col].sum()), "HR": s["exp(coef)"],
               "HR_lower": s["exp(coef) lower 95%"], "HR_upper": s["exp(coef) upper 95%"], "p": s["p"]}
        if binary:
            for level in (0, 1):
                g = x[x[cov] == level]
                med = km_median(g[t_col], g[e_col], g[entry_col] if entry_col else None)[0]
                row[f"n_{level}"], row[f"median_{level}_months"] = len(g), med / DAYS_PER_MONTH
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        out["q_BH"] = benjamini_hochberg(out["p"])
        out = out.sort_values("p", ignore_index=True)
    return out
