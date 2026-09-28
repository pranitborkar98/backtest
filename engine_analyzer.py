"""
engine_analyzer.py — Internal Multi-Engine Prediction Analyzer
- Reads 10+ engine predictions from D:\backtest
- Reads actual results from C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\
- Supports --aggregate for weekly/monthly/all-time analysis.
"""
from __future__ import annotations

import os
import sys
import re
import json
import argparse
import logging
import csv
import hashlib
from pathlib import Path
from datetime import date, datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple, Iterable, Set
from itertools import combinations_with_replacement

# Hardcoded paths based on your architecture
TARGET_DIR = Path(r"D:\backtest")
HISTORY_FILE_DEFAULT = Path(r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\sattaboss-data\data\all_markets_history.json")
YESTERDAY_RESULTS_FILE = Path(r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\yesterday_results.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])
log = logging.getLogger("engine_analyzer")

CATEGORIES = ["od", "cd", "op", "cp", "j"]
CATEGORY_LABELS = {"od": "Open Digits", "cd": "Close Digits", "op": "Open Pannas", "cp": "Close Pannas", "j":  "Jodis"}
CATEGORY_ALIASES = {
    "od": ("opendigit", "od", "opendigits", "open_digit", "open_digits"),
    "cd": ("closedigit", "cd", "closedigits", "close_digit", "close_digits"),
    "op": ("openpanna", "op", "openpannas", "open_panna", "open_pannas"),
    "cp": ("closepanna", "cp", "closepannas", "close_panna", "close_pannas"),
    "j":  ("jodi", "j", "jodis"),
}
CATEGORY_WIDTH = {"od": 1, "cd": 1, "op": 3, "cp": 3, "j": 2}

PANNAS_UNIVERSE = tuple(sorted("".join(p) for p in combinations_with_replacement("0123456789", 3)))
JODI_UNIVERSE = tuple(f"{i:02d}" for i in range(100))
DIGIT_UNIVERSE = tuple(str(i) for i in range(10))
UNIVERSES = {"od": DIGIT_UNIVERSE, "cd": DIGIT_UNIVERSE, "op": PANNAS_UNIVERSE, "cp": PANNAS_UNIVERSE, "j": JODI_UNIVERSE}
INDEX = {k: {v: i for i, v in enumerate(vals)} for k, vals in UNIVERSES.items()}

def key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

def market_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

def parse_date(value: Any) -> Optional[date]:
    if not value: return None
    value = str(value).strip()
    for fmt, part in (("%Y-%m-%d", value[:10]), ("%d/%m/%Y", value), ("%Y%m%d", value)):
        try: return datetime.strptime(part, fmt).date()
        except ValueError: pass
    return None

def normalize_value(value: Any, width: int) -> Optional[str]:
    if isinstance(value, bool) or value is None: return None
    s = str(value).strip()
    if not re.fullmatch(r"[0-9]{1," + str(width) + r"}", s): return None
    return s.zfill(width)

def safe_load_json(path: Path) -> Optional[Any]:
    try: return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception: return None

def detect_engine_label(path: Path, meta: Dict[str, Any]) -> str:
    eng = (meta or {}).get("engine") or (meta or {}).get("engine_version")
    if eng: return str(eng)
    
    name = path.stem.lower()
    # Matches v4, v10, v31, v52, v14_3, v50_unified, ml_panna_first, AbsoluteEngine, etc.
    patterns = [
        re.compile(r"_(v\d+[\w_]*)"),        # matches _v31, _v14_3, _v50_unified
        re.compile(r"_(ml_panna_first)"),    # matches _ml_panna_first
        re.compile(r"_(absoluteengine)")     # matches _absoluteengine
    ]
    for pat in patterns:
        m = pat.search(name)
        if m:
            return m.group(1)
    return path.stem

def load_history(path: Path) -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
    raw = safe_load_json(path)
    if not isinstance(raw, dict): return {}
    out: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
    for market, rows in raw.items():
        if not isinstance(rows, list): continue
        m = market_key(market)
        if not m: continue
        by_date: Dict[str, Dict[str, Optional[str]]] = {}
        for row in rows:
            if not isinstance(row, dict): continue
            dt = parse_date(row.get("Date") or row.get("date"))
            if dt is None: continue
            iso = dt.isoformat()
            norm = {key(k): v for k, v in row.items()}
            entry: Dict[str, Optional[str]] = {}
            for cat, aliases in CATEGORY_ALIASES.items():
                val = next((norm[a] for a in aliases if a in norm and norm[a] is not None), None)
                entry[cat] = normalize_value(val, CATEGORY_WIDTH[cat])
            if entry.get("op") and not entry.get("od"): entry["od"] = str(sum(map(int, entry["op"])) % 10)
            if entry.get("cp") and not entry.get("cd"): entry["cd"] = str(sum(map(int, entry["cp"])) % 10)
            if entry.get("od") and entry.get("cd") and not entry.get("j"): entry["j"] = entry["od"] + entry["cd"]
            by_date[iso] = entry
        out[m] = by_date
    return out

def load_yesterday_results(path: Path, target_date: date) -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
    raw = safe_load_json(path)
    if not raw: return {}
    if isinstance(raw, list):
        data_dict = {}
        for item in raw:
            m = item.get("Market") or item.get("market") or "UNKNOWN"
            data_dict[m] = item
        raw = data_dict
    if not isinstance(raw, dict): return {}
    iso = target_date.isoformat()
    out: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
    for market, fields in raw.items():
        m = market_key(market)
        if not m: continue
        norm = {key(k): v for k, v in fields.items()}
        entry: Dict[str, Optional[str]] = {}
        for cat, aliases in CATEGORY_ALIASES.items():
            val = next((norm[a] for a in aliases if a in norm and norm[a] is not None), None)
            entry[cat] = normalize_value(val, CATEGORY_WIDTH[cat])
        if entry.get("op") and not entry.get("od"): entry["od"] = str(sum(map(int, entry["op"])) % 10)
        if entry.get("cp") and not entry.get("cd"): entry["cd"] = str(sum(map(int, entry["cp"])) % 10)
        if entry.get("od") and entry.get("cd") and not entry.get("j"): entry["j"] = entry["od"] + entry["cd"]
        if m not in out: out[m] = {}
        out[m][iso] = entry
    return out

def iter_prediction_entries(obj: Any) -> Iterable[Tuple[str, Any, Dict[str, Any]]]:
    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
    items = obj.get("markets", []) if isinstance(obj, dict) else obj
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict): yield item.get("market", item.get("Market", "")), item, meta
    elif isinstance(items, dict):
        for market, entries in items.items():
            if isinstance(entries, list):
                for item in entries:
                    if isinstance(item, dict): yield market, item, meta
            elif isinstance(entries, dict):
                yield market, entries, meta

def extract_picks(item: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[Optional[date], Dict[str, List[str]]]:
    pred = item.get("predicted") or item.get("predictions") or item
    if not isinstance(pred, dict): return None, {}
    dt = parse_date(item.get("date") or pred.get("Date") or pred.get("date") or meta.get("prediction_date") or meta.get("date"))
    norm = {key(k): v for k, v in pred.items()}
    picks: Dict[str, List[str]] = {}
    for cat, label in CATEGORY_LABELS.items():
        raw = norm.get(key(label))
        if isinstance(raw, list) and len(raw) == 2 and isinstance(raw[0], list) and isinstance(raw[1], dict):
            raw = raw[0]
        if not isinstance(raw, list): raw = []
        width = CATEGORY_WIDTH[cat]
        cleaned: List[str] = []
        seen: Set[str] = set()
        for val in raw:
            v = normalize_value(val, width)
            if v in INDEX[cat] and v not in seen:
                seen.add(v)
                cleaned.append(v)
        picks[cat] = cleaned
    return dt, picks

def score_one(picks: Dict[str, List[str]], actual: Dict[str, Optional[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for cat in CATEGORIES:
        actual_val = actual.get(cat)
        picks_list = picks.get(cat, [])
        if actual_val is None or actual_val == "": out[cat] = "no_actual"
        elif not picks_list: out[cat] = "no_picks"
        elif actual_val in picks_list: out[cat] = "hit"
        else: out[cat] = "miss"
    return out

def score_records(records: List[Dict[str, Any]], history: Dict[str, Dict[str, Dict[str, Optional[str]]]]) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for rec in records:
        hist_actual = history.get(rec["market"], {}).get(rec["target_date"], {})
        merged_actual: Dict[str, Optional[str]] = {}
        for cat in CATEGORIES:
            merged_actual[cat] = hist_actual.get(cat)
        status = score_one(rec["picks"], merged_actual)
        scored.append({**rec, "actual": merged_actual, "status": status})
    return scored

def aggregate_scores(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_engine: Dict[str, Counter] = defaultdict(Counter)
    by_engine_cat: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    by_engine_market_cat: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)

    for rec in scored:
        eng, mkt = rec["engine"], rec["market"]
        for cat in CATEGORIES:
            st = rec["status"][cat]
            by_engine[eng][st] += 1
            by_engine_cat[(eng, cat)][st] += 1
            by_engine_market_cat[(eng, mkt, cat)][st] += 1

    def rates(c: Counter) -> Dict[str, Any]:
        hits, misses = c.get("hit", 0), c.get("miss", 0)
        scored_count = hits + misses
        return {
            "hits": hits, "misses": misses,
            "no_actual": c.get("no_actual", 0), "no_picks": c.get("no_picks", 0),
            "scored": scored_count,
            "hit_rate": (hits / scored_count) if scored_count else None,
        }

    return {
        "by_engine": {eng: rates(c) for eng, c in by_engine.items()},
        "by_engine_category": {f"{eng}|{cat}": rates(c) for (eng, cat), c in by_engine_cat.items()},
        "by_engine_market_category": {f"{eng}|{mkt}|{cat}": rates(c) for (eng, mkt, cat), c in by_engine_market_cat.items()},
    }

def print_console_report(agg: Dict[str, Any], target_date: Optional[str]) -> None:
    print("\n" + "=" * 80)
    if target_date: print(f"  ANALYSIS FOR DATE: {target_date}")
    else: print("  AGGREGATE ANALYSIS (all available history in D:\\backtest)")
    print("=" * 80)
    
    print("\nENGINE OVERVIEW (sorted by overall hit rate, scored only)")
    print("-" * 80)
    rows = []
    for eng, r in agg["by_engine"].items():
        hr = r["hit_rate"]
        rows.append((eng, r, hr if hr is not None else -1))
    rows.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Engine':<25} {'Hits':>6} {'Misses':>7} {'NoAct':>6} {'NoPick':>7} {'Scored':>7} {'HitRate':>8}")
    for eng, r, _ in rows:
        hr = f"{r['hit_rate']:.3f}" if r['hit_rate'] is not None else "  n/a "
        print(f"{eng:<25} {r['hits']:>6} {r['misses']:>7} {r['no_actual']:>6} {r['no_picks']:>7} {r['scored']:>7} {hr:>8}")

    print("\nPER-ENGINE PER-CATEGORY HIT RATES")
    print("-" * 80)
    engines = sorted({k.split("|")[0] for k in agg["by_engine_category"]})
    header = f"{'Engine':<25}" + "".join(f"{CATEGORY_LABELS[c]:>14}" for c in CATEGORIES)
    print(header)
    for eng in engines:
        line = f"{eng:<25}"
        for cat in CATEGORIES:
            r = agg["by_engine_category"].get(f"{eng}|{cat}", {})
            hr = r.get("hit_rate")
            cell = f"{hr:.2f} ({r.get('scored', 0)})" if hr is not None else "n/a"
            line += f"{cell:>14}"
        print(line)
    print("=" * 80 + "\n")

def write_json_report(path: Path, agg: Dict[str, Any], scored: List[Dict[str, Any]], target_date: Optional[str], engines_seen: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "target_date": target_date,
        "engines_seen": engines_seen,
        "summary": agg,
        "records": scored,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def write_csv_report(path: Path, scored: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["engine", "market", "target_date", "category", "status", "actual", "picks", "source"])
        for rec in scored:
            for cat in CATEGORIES:
                w.writerow([
                    rec["engine"], rec["market"], rec["target_date"], cat, rec["status"][cat],
                    rec["actual"].get(cat, "") or "",
                    "|".join(rec["picks"].get(cat, [])),
                    rec.get("source", ""),
                ])

def analyze(target_date: Optional[date], history_path: Path, aggregate: bool, out_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    log.info(f"Loading history from {history_path}")
    history = load_history(history_path)
    
    if not aggregate and target_date and YESTERDAY_RESULTS_FILE.exists():
        log.info(f"Loading actuals from {YESTERDAY_RESULTS_FILE}")
        yday_data = load_yesterday_results(YESTERDAY_RESULTS_FILE, target_date)
        for mkt, dt_dict in yday_data.items():
            if mkt not in history: history[mkt] = {}
            for dt, actuals in dt_dict.items(): history[mkt][dt] = actuals
    
    if not TARGET_DIR.exists():
        log.error(f"Target directory does not exist: {TARGET_DIR}")
        return {}, []

    log.info(f"Scanning for JSON files in: {TARGET_DIR}")
    # STRICTLY look for files starting with 'predictions_' to ignore raw .py and history files
    files = list(TARGET_DIR.glob("predictions_*.json"))
    
    records: List[Dict[str, Any]] = []
    for p in files:
        obj = safe_load_json(p)
        if obj is None: continue
        engine_label = detect_engine_label(p, obj.get("meta", {}) if isinstance(obj, dict) else {})
        for market, item, meta in iter_prediction_entries(obj):
            dt, picks = extract_picks(item, meta)
            if dt is None or not any(picks.values()): continue
            records.append({
                "engine": engine_label,
                "market": market_key(market),
                "target_date": dt.isoformat(),
                "picks": picks,
                "source": str(p),
            })

    if not aggregate and target_date is not None:
        iso = target_date.isoformat()
        records = [r for r in records if r["target_date"] == iso]

    scored = score_records(records, history)
    agg = aggregate_scores(scored)
    engines_seen = sorted({r["engine"] for r in scored})

    target_str = target_date.isoformat() if target_date else None
    print_console_report(agg, target_str)
    write_json_report(out_dir / "analysis_report.json", agg, scored, target_str, engines_seen)
    write_csv_report(out_dir / "analysis_detail.csv", scored)

    print("BEST ENGINE PER CATEGORY (by hit rate, min 5 scored)")
    print("-" * 60)
    for cat in CATEGORIES:
        best_eng, best_hr, best_n = None, -1, 0
        for eng in engines_seen:
            r = agg["by_engine_category"].get(f"{eng}|{cat}", {})
            n = r.get("scored", 0)
            if n < 5: continue
            hr = r.get("hit_rate") or 0
            if hr > best_hr:
                best_hr, best_eng, best_n = hr, eng, n
        if best_eng:
            print(f"  {CATEGORY_LABELS[cat]:<14} -> {best_eng}  (hr={best_hr:.3f}, n={best_n})")
        else:
            print(f"  {CATEGORY_LABELS[cat]:<14} -> insufficient data")
    print()

    return agg, scored

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD. Default: yesterday.")
    ap.add_argument("--aggregate", action="store_true", help="Aggregate ALL files in D:\\backtest for long-term analysis.")
    ap.add_argument("--history", type=Path, default=HISTORY_FILE_DEFAULT, help="Path to all_markets_history.json")
    ap.add_argument("--out-dir", type=Path, default=TARGET_DIR, help="Where to write reports")
    args = ap.parse_args()

    target_date: Optional[date] = None
    if not args.aggregate:
        if args.date:
            target_date = parse_date(args.date)
            if target_date is None:
                log.error(f"Could not parse --date {args.date}")
                return 1
        else:
            target_date = date.today() - timedelta(days=1)
            log.info(f"No --date given; defaulting to yesterday ({target_date.isoformat()})")

    if not args.history.exists():
        log.error(f"History file does not exist: {args.history}")
        return 1

    analyze(target_date, args.history, args.aggregate, args.out_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())