"""Joint Nelder-Mead fit to PFR/first-transition CRP and their variances.

Observations come either from a CSV with columns
condition,measure,x,value,variance
(measure pfr: x=1..12; crp: x=-11..-1,1..11), or from recall matrices via
observations_from_recalls. CRP fitting uses delayed/continuous only,
matching section 4.1.4.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from tcm import (CONDITIONS, Parameters, empirical_first_two, expected_lag_crp,
                 sample_first_two, study_list, transition_distribution)


def read_observations(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        data = list(csv.DictReader(stream))
    if not data:
        raise ValueError("empty observations file")
    observed, seen = [], set()
    for row in data:
        condition, measure = row["condition"], row["measure"]
        x = int(row["x"])
        value, variance = float(row["value"]), float(row["variance"])
        if condition not in CONDITIONS or measure not in ("pfr", "crp"):
            raise ValueError("unknown condition or measure")
        if measure == "pfr" and not 1 <= x <= 12:
            raise ValueError("PFR serial positions must be 1..12")
        if measure == "crp" and (condition == "immediate" or x == 0 or abs(x) > 11):
            raise ValueError("CRP must be nonzero lag in delayed/continuous")
        if not np.isfinite(value) or not 0 <= value <= 1 or not np.isfinite(variance) or variance <= 0:
            raise ValueError("value must be a probability; variance must be finite and positive")
        key = (condition, measure, x)
        if key in seen:
            raise ValueError(f"duplicate observation: {key}")
        seen.add(key)
        observed.append((key, value, variance))
    return observed


def observations_from_recalls(recalls, subjects, condition, *, list_length=12,
                              max_lag=5, min_variance=1e-4):
    """Subject-level PFR and first-output lag-CRP from a recalls matrix.

    recalls: (lists, outputs) serial positions 1..list_length; 0, negative or
    larger values mark no response or intrusions. subjects: one label per list.
    PFR uses lists whose first output is a list item. The CRP uses lists whose
    first two outputs are distinct list items. Each statistic is averaged over
    subjects; its variance is the squared standard error of that mean.
    """
    recalls = np.asarray(recalls)
    subjects = np.asarray(subjects)
    if recalls.ndim != 2 or len(subjects) != len(recalls):
        raise ValueError("recalls must be (lists, outputs) with one subject per list")
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    valid = (recalls >= 1) & (recalls <= list_length)
    lags = [x for x in range(-max_lag, max_lag+1) if x]
    pfr_rows, crp_rows = [], []
    for subject in np.unique(subjects):
        rows = recalls[subjects == subject]
        ok = valid[subjects == subject]
        first = rows[ok[:, 0], 0]
        if first.size:
            pfr_rows.append(np.bincount(first, minlength=list_length+1)[1:] / first.size)
        if condition == "immediate" or rows.shape[1] < 2:
            continue
        pair = ok[:, 0] & ok[:, 1] & (rows[:, 0] != rows[:, 1])
        a, b = rows[pair, 0], rows[pair, 1]
        if not a.size:
            continue
        numerator = np.array([np.sum(b - a == lag) for lag in lags], dtype=float)
        denominator = np.array([np.sum((a + lag >= 1) & (a + lag <= list_length))
                                for lag in lags], dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            crp_rows.append(numerator / denominator)
    observed = []

    def summarize(matrix, measure, xs):
        matrix = np.asarray(matrix, dtype=float)
        for j, x in enumerate(xs):
            column = matrix[:, j][np.isfinite(matrix[:, j])]
            if column.size < 2:
                continue
            variance = max(column.var(ddof=1) / column.size, min_variance)
            observed.append(((condition, measure, int(x)), float(column.mean()), float(variance)))

    if pfr_rows:
        summarize(pfr_rows, "pfr", range(1, list_length+1))
    if crp_rows:
        summarize(crp_rows, "crp", lags)
    return observed


def predictions(params, *, exact=False, trials=3000, seed=2002, suppress_repeats=True):
    result = {}
    for k, condition in enumerate(CONDITIONS):
        model, _ = study_list(condition, params)
        first, second = transition_distribution(model, suppress_repeats=suppress_repeats)
        if exact:
            lags, crp, _, _ = expected_lag_crp(first, second)
        else:
            responses = sample_first_two(first, second, trials, seed+k)
            first, lags, crp, _, _ = empirical_first_two(responses, 12)
        for i, value in enumerate(first.tolist(), start=1):
            result[(condition, "pfr", i)] = value
        for lag, value in zip(lags.tolist(), crp.tolist()):
            result[(condition, "crp", lag)] = value
    return result


def fit(observed, *, exact=False, trials=3000, seed=2002,
        suppress_repeats=True, maxiter=500):
    def objective(x):
        beta, tau, duration = x
        if not (0 < beta < 1 and tau > 0 and duration >= 0) or not np.isfinite(x).all():
            return 1e12
        pred = predictions(Parameters(beta=beta, tau=tau, distractor_duration=duration),
                           exact=exact, trials=trials, seed=seed,
                           suppress_repeats=suppress_repeats)
        errors = [(pred[key]-value)**2 / variance for key, value, variance in observed]
        return float(np.sqrt(np.mean(errors))) if np.isfinite(errors).all() else 1e12

    initial = np.array([.402, .247, 7.24])
    # Nonzero simplex scale is important for the Monte Carlo objective plateaus.
    simplex = np.stack([initial, initial+[.025, 0, 0], initial+[0, .025, 0], initial+[0, 0, .8]])
    result = minimize(objective, initial, method="Nelder-Mead",
                      options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6,
                               "initial_simplex": simplex})
    beta, tau, duration = result.x
    return {"parameters": asdict(Parameters(beta=beta, tau=tau, distractor_duration=duration)),
            "success": bool(result.success), "message": str(result.message),
            "weighted_rmsd": float(result.fun), "evaluations": int(result.nfev),
            "exact_expectations": exact, "trials_per_condition": None if exact else trials,
            "seed": seed, "suppress_repeats": suppress_repeats,
            "observations": len(observed),
            "caution": "fit applies to documented distractor/output conventions; not original-author code"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "outputs/tcm2002/fit.json")
    parser.add_argument("--exact", action="store_true", help="noise-free expectation instead of 3000 samples")
    parser.add_argument("--allow-repeats", action="store_true",
                        help="raw Eq. (14) for the second response, repeats allowed")
    parser.add_argument("--trials", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2002)
    parser.add_argument("--maxiter", type=int, default=500)
    args = parser.parse_args()
    torch.set_num_threads(1)
    observed = read_observations(args.data)
    result = fit(observed, exact=args.exact, trials=args.trials, seed=args.seed,
                 suppress_repeats=not args.allow_repeats, maxiter=args.maxiter)
    result["source_data"] = str(args.data.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
