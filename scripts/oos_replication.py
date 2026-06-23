#!/usr/bin/env python
"""Out-of-sample replication of the 'fade the acceleration' signal.

Re-runs the keyless backfill + conditional battery on several independent
windows and tabulates the accelerating-attention (and spike) forward returns in
liquid names. A real signal replicates with a consistent sign across windows; a
lucky in-sample draw does not.
"""
import _bootstrap  # noqa: F401

import pandas as pd

from reddit_hype.config import load_settings
from reddit_hype.diagnostics import conditional_battery
from reddit_hype.pipeline import backfill_keyless
from reddit_hype.utils import read_parquet

SUBS = ["wallstreetbets", "stocks", "smallstreetbets", "pennystocks"]
WINDOWS = [
    ("2022-09-01", "2022-11-30"),
    ("2023-03-01", "2023-05-31"),
    ("2024-03-01", "2024-05-31"),
]

if __name__ == "__main__":
    s = load_settings()
    rows = []
    for since, until in WINDOWS:
        print(f"\n######## WINDOW {since}..{until} ########", flush=True)
        summ = backfill_keyless(since=since, until=until, subreddits=SUBS,
                                max_records_per_kind=4000, include_comments=False)
        print("summary:", summ, flush=True)
        if not summ.get("panel_rows"):
            continue
        panel = read_parquet(s.path("panel"))
        cond = conditional_battery(panel, s, min_dollar_volume=1e7)
        for name in ["acceleration_top_quartile", "sudden_spike"]:
            for h in [1, 5]:
                r = cond[(cond["condition"] == name) & (cond["horizon_days"] == h)]
                if not r.empty:
                    r = r.iloc[0]
                    rows.append({"window": f"{since}..{until}", "condition": name,
                                 "horizon_days": h, "n_obs": int(r["n_obs"]),
                                 "cond_mean_mktadj": r["cond_mean_mktadj"],
                                 "cond_tstat_nw": r["cond_tstat_nw"]})

    out = pd.DataFrame(rows)
    if not out.empty:
        path = s.path("tables") / "oos_acceleration_fade.csv"
        out.to_csv(path, index=False)
        print("\n================ OOS REPLICATION SUMMARY ================")
        print(out.to_string(index=False))
        print(f"\nsaved -> {path}")
        acc1 = out[(out["condition"] == "acceleration_top_quartile") & (out["horizon_days"] == 1)]
        if not acc1.empty:
            signs = (acc1["cond_mean_mktadj"] < 0).sum()
            sig = (acc1["cond_tstat_nw"].abs() >= 2).sum()
            print(f"\nAcceleration 1d: {signs}/{len(acc1)} windows negative (fade), "
                  f"{sig}/{len(acc1)} individually significant.")
