r"""
run_backtest.py - Time-travel data generator for the Quant Backtesting Lab.

Loops through the last N days and runs every prediction engine with
--date YYYY-MM-DD. Each engine's BACKTEST injection block then writes a
uniquely named file into D:\backtest:  predictions_<YYYY-MM-DD>_<engine>.json

Usage (from D:\backtest):
    python run_backtest.py                 # last 30 days, all engines
    python run_backtest.py --days 60       # last 60 days
    python run_backtest.py --start 2024-09-01 --end 2024-10-31
    python run_backtest.py --engines v31 v32          # subset
    python run_backtest.py --dry-run       # print commands only
    python run_backtest.py --workers 2     # parallel engines (default 1)

Notes:
  * v31.py and prediction_engine_v31.py share the 'v31' label -> pick ONE
    (default list uses prediction_engine_v31.py; switch if you prefer v31.py).
  * Engines that fail are logged to backtest_errors.log and the loop continues.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET_DIR = HERE  # run this script from inside D:\backtest
ERROR_LOG = HERE / "backtest_errors.log"

# Engine scripts present in D:\backtest. Edit this list to add/remove engines.
ENGINES = [
    "v10.py",
    "v20.py",
    "v30.py",
    "prediction_engine_v31.py",   # or swap to "v31.py" (same 'v31' label!)
    "prediction_engine_v32.py",
    "prediction_engine_v33.py",
    "prediction_engine_ml_panna_first.py",
    "AbsoluteEngine.py",
    "prediction_engine_v50_unified.py",
    "prediction_engine_v51_unified.py",
    "prediction_engine_v52_adaptive.py",
    "prediction_engine_v53_unified.py",
    "prediction_engine_v4.py",
    "matka_engine_v14_3.py",
]


def _history_dates() -> list[str]:
    """Distinct dates present in all_markets_history.json, normalized to
    YYYY-MM-DD, sorted ascending. Empty list if file missing."""
    hist = HERE / "all_markets_history.json"
    if not hist.exists():
        return []
    try:
        data = json.loads(hist.read_text(encoding="utf-8"))
    except Exception:
        return []
    seen = set()
    for recs in data.values():
        if isinstance(recs, dict):                     # {"records": [...]} shape
            recs = recs.get("records") or recs.get("history") or []
        if not isinstance(recs, list):
            continue
        for r in recs:
            if not isinstance(r, dict):
                continue
            d = str(r.get("Date") or r.get("date") or "").strip()
            if not d:
                continue
            parts = d.replace("/", "-").split("-")
            if len(parts) == 3:
                a, b, c = parts
                if len(a) == 4:            # already YYYY-MM-DD
                    iso = f"{a}-{int(b):02d}-{int(c):02d}"
                elif len(c) == 4:          # DD/MM/YYYY or MM/DD/YYYY
                    mth, dy = (b, a) if int(b) <= 12 else (a, b)
                    iso = f"{c}-{int(mth):02d}-{int(dy):02d}"
                else:
                    continue
                seen.add(iso)
    return sorted(seen)


def build_dates(start_s: str | None, end_s: str | None, days: int) -> list[str]:
    hd = _history_dates()
    if start_s and end_s:
        s = datetime.strptime(start_s, "%Y-%m-%d").date().isoformat()
        e = datetime.strptime(end_s, "%Y-%m-%d").date().isoformat()
        if hd:
            return [d for d in hd if s <= d <= e]   # only real market days
        d0, d1 = date.fromisoformat(s), date.fromisoformat(e)
        out, d = [], d0
        while d <= d1:
            out.append(d.isoformat())
            d += timedelta(days=1)
        return out
    if hd:
        # Only dates up to today (future-dated rows in history have no actuals yet)
        cap = date.today().isoformat()
        past = [d for d in hd if d <= cap] or hd
        return past[-days:]                          # last N days WITH actual results
    today = date.today()
    return [(today - timedelta(days=i)).isoformat() for i in range(days, 0, -1)]


ENGINE_LABELS = {
    "v10.py": "v10", "v20.py": "v20", "v30.py": "v30",
    "v31.py": "v31", "prediction_engine_v31.py": "v31",
    "prediction_engine_v32.py": "v32", "prediction_engine_v33.py": "v33",
    "prediction_engine_ml_panna_first.py": "ml_panna_first",
    "AbsoluteEngine.py": "absolute",
    "prediction_engine_v50_unified.py": "v50_unified",
    "prediction_engine_v51_unified.py": "v51_unified",
    "prediction_engine_v52_adaptive.py": "v52_adaptive",
    "prediction_engine_v53_unified.py": "v53_unified",
    "prediction_engine_v4.py": "v4",
    "matka_engine_v14_3.py": "v14_3",
}


def run_one(engine: str, dt: str, dry: bool, skip_existing: bool = False) -> tuple[str, str, float, str]:
    if skip_existing and not dry:
        label = ENGINE_LABELS.get(engine, engine.replace(".py", ""))
        if (TARGET_DIR / f"predictions_{dt}_{label}.json").exists():
            return engine, dt, 0.0, "SKIP (file exists)"
    cmd = [sys.executable, str(HERE / engine), "--date", dt]
    if dry:
        return engine, dt, 0.0, "DRY " + " ".join(cmd)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, timeout=600)
        el = time.time() - t0
        if p.returncode != 0:
            msg = (p.stderr or p.stdout or "").strip().splitlines()[-3:]
            return engine, dt, el, f"FAIL rc={p.returncode}: {' | '.join(msg)}"
        return engine, dt, el, "OK"
    except subprocess.TimeoutExpired:
        return engine, dt, time.time() - t0, "FAIL timeout(600s)"
    except Exception as e:  # noqa: BLE001
        return engine, dt, time.time() - t0, f"FAIL {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=30, help="Lookback window (default 30)")
    ap.add_argument("--full", action="store_true", help="EVERY date in all_markets_history.json (~8 years)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip engines whose dated prediction file already exists")
    ap.add_argument("--start", type=str, default=None, help="Explicit start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="Explicit end date YYYY-MM-DD")
    ap.add_argument("--engines", nargs="*", default=None, help="Subset of engine filenames")
    ap.add_argument("--dry-run", action="store_true", help="Print commands, execute nothing")
    ap.add_argument("--workers", type=int, default=1, help="Parallel engines per date (default 1)")
    args = ap.parse_args()

    engines = args.engines or ENGINES
    missing = [e for e in engines if not (HERE / e).exists()]
    if missing:
        print("ERROR: engine scripts not found in", HERE, "->", ", ".join(missing))
        return 2

    if args.full:
        hd = _history_dates()
        if not hd:
            print("ERROR: --full requested but no dates found in all_markets_history.json")
            return 2
        dates = hd                                   # EVERY date in history (2018-2025)
    else:
        dates = build_dates(args.start, args.end, args.days)
    if not dates:
        print("ERROR: no dates to run for the requested range")
        return 2
    total = len(dates) * len(engines)
    print(f"Backtest plan: {len(dates)} dates x {len(engines)} engines = {total} runs")
    print(f"Output folder: {TARGET_DIR}")
    print(f"Date range: {dates[0]} .. {dates[-1]}")
    if not args.dry_run:
        TARGET_DIR.mkdir(parents=True, exist_ok=True)

    done = fails = 0
    t_start = time.time()
    for i, dt in enumerate(dates, 1):
        print(f"\n[{i}/{len(dates)}] {dt}")
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(lambda e: run_one(e, dt, args.dry_run, args.skip_existing), engines))
        else:
            results = [run_one(e, dt, args.dry_run, args.skip_existing) for e in engines]
        for eng, _dt, el, status in results:
            done += 1
            if status.startswith("FAIL"):
                fails += 1
                with open(ERROR_LOG, "a", encoding="utf-8") as fh:
                    fh.write(f"{_dt}\t{eng}\t{status}\n")
            print(f"   {eng:<40} {status}{'' if el == 0 else f'  ({el:.1f}s)'}")

    mins = (time.time() - t_start) / 60
    produced = sorted(TARGET_DIR.glob("predictions_*.json")) if not args.dry_run else []
    print("\n===== SUMMARY =====")
    print(f"Runs attempted: {done} | failures: {fails} | elapsed: {mins:.1f} min")
    print(f"prediction_*.json files in {TARGET_DIR}: {len(produced)}")
    if fails:
        print(f"Failure details: {ERROR_LOG}")
    print("Next step: python engine_analyzer.py --aggregate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
