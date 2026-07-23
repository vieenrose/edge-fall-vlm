"""Paired comparison of two models from bench_full.py per-clip records (McNemar exact).

Compares on BINARY DANGER-CORRECTNESS (was the danger/normal call right?), matched by clip
id — the discordant pairs (one model right, the other wrong on the same clip) carry all the
signal, which is why this resolves ~5-8pt deltas where unpaired comparison at n=150 needs
~13-21pts. See EVAL_PROTOCOL.md for the pre-registered promotion rule this implements.

    python scripts/paired_test.py --report bench_a.json --report bench_b.json \
        [--label MODEL_A --label MODEL_B]

Each --report is a bench_full.py output JSON containing a "predictions" list. If a report
holds several models, give --label to pick one (defaults to the first).
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.eval import DANGER


def load_preds(report_path: Path, label: str | None) -> tuple[str, dict[str, dict]]:
    rep = json.loads(report_path.read_text())
    key = label if label else next(iter(rep))
    if key not in rep:
        raise SystemExit(f"{report_path}: no model '{key}' (have: {list(rep)})")
    recs = rep[key].get("predictions")
    if not recs:
        raise SystemExit(f"{report_path}[{key}]: no per-clip 'predictions' — re-run "
                         "bench_full.py (per-clip logging landed 2026-07-23)")
    return key, {r["id"]: r for r in recs}


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant counts (b vs c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="append", required=True, type=Path,
                    help="bench_full.py report JSON (give exactly two)")
    ap.add_argument("--label", action="append", default=None,
                    help="model label inside each report (defaults to first key)")
    args = ap.parse_args()
    if len(args.report) != 2:
        raise SystemExit("need exactly two --report")
    labels = args.label or [None, None]

    (name_a, a), (name_b, b) = (load_preds(args.report[i], labels[i]) for i in (0, 1))
    common = sorted(set(a) & set(b))
    if len(common) != len(a) or len(common) != len(b):
        print(f"WARNING: clip sets differ (A={len(a)}, B={len(b)}, common={len(common)}) — "
              "pairing on the intersection only")
    if not common:
        raise SystemExit("no common clips")

    b_cnt = c_cnt = both_right = both_wrong = 0     # b: A right & B wrong; c: A wrong & B right
    rec_flips = {"a_only_catches": [], "b_only_catches": []}
    parse_fail = {"a": 0, "b": 0}
    for cid in common:
        ra, rb = a[cid], b[cid]
        gold_danger = ra["gold"] in DANGER
        ok_a = (ra["pred"] in DANGER) == gold_danger
        ok_b = (rb["pred"] in DANGER) == gold_danger
        parse_fail["a"] += ra.get("parse", "json") != "json"
        parse_fail["b"] += rb.get("parse", "json") != "json"
        if ok_a and ok_b: both_right += 1
        elif ok_a and not ok_b: b_cnt += 1
        elif ok_b and not ok_a: c_cnt += 1
        else: both_wrong += 1
        if gold_danger and ok_a and not ok_b: rec_flips["a_only_catches"].append(cid)
        if gold_danger and ok_b and not ok_a: rec_flips["b_only_catches"].append(cid)

    n = len(common)
    p = mcnemar_exact_p(b_cnt, c_cnt)
    net = b_cnt - c_cnt   # positive = A better
    print(f"paired on {n} clips: {name_a} vs {name_b}")
    print(f"  both right {both_right}, both wrong {both_wrong}, "
          f"{name_a}-only-right {b_cnt}, {name_b}-only-right {c_cnt}")
    print(f"  net delta (danger-correct clips): {net:+d}  ({name_a} minus {name_b})")
    print(f"  McNemar exact p = {p:.4f}")
    print(f"  positives only {name_a} catches: {len(rec_flips['a_only_catches'])} "
          f"{rec_flips['a_only_catches'][:8]}")
    print(f"  positives only {name_b} catches: {len(rec_flips['b_only_catches'])} "
          f"{rec_flips['b_only_catches'][:8]}")
    print(f"  parse failures: {name_a} {parse_fail['a']}/{n}, {name_b} {parse_fail['b']}/{n}"
          + ("  << >5% — fix harness before judging" if max(parse_fail.values()) > 0.05 * n else ""))
    verdict = ("SIGNIFICANT" if p < 0.05 else "within noise")
    better = name_a if net > 0 else (name_b if net < 0 else "neither")
    print(f"  => {verdict}; point-estimate favors {better}. Apply EVAL_PROTOCOL.md rule.")


if __name__ == "__main__":
    main()
