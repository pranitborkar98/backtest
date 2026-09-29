#!/usr/bin/env python3
"""
prediction_engine_v4.py
Stable, robust prediction engine:
- Loads history from multiple candidate locations (prefers sattaboss-data/data)
- Generates per-market predictions (digits, pannas, jodis)
- Archives previous todays_predictions.json and writes outputs to multiple destinations
- Defensive parsing (handles '*' and other junk in history)
"""
from __future__ import annotations

import os
import json
import random
import logging
import time
import secrets
import shutil
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------- PATHS CONFIG ----------------
# Force root to one level up from this script (repo root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_FOLDER = os.path.join(ROOT, "output")
OUTPUT_HISTORY = os.path.join(OUTPUT_FOLDER, "history")

KALYAN_FOLDER = os.path.join(ROOT, "kalyan")
SATTABOSS_DATA = os.path.join(ROOT, "sattaboss-data", "data")
KALYAN_ASSETS = os.path.join(KALYAN_FOLDER, "assets")

FLUTTER_PROJECT_ASSETS = r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets"

# Input history candidates (priority)
HISTORY_CANDIDATES = [
    os.path.join(SATTABOSS_DATA, "all_markets_history.json"),
    os.path.join(OUTPUT_FOLDER, "all_markets_history.json"),
    os.path.join(KALYAN_FOLDER, "all_markets_history.json"),
    os.path.join(ROOT, "all_markets_history.json"),
    os.path.join(OUTPUT_HISTORY, "all_markets_history.json"),
   ]

TODAYS_NAME = "todays_predictions.json"
OUTPUT_HISTORY_TODAYS = os.path.join(OUTPUT_HISTORY, TODAYS_NAME)
ROOT_TODAYS = os.path.join(ROOT, TODAYS_NAME)
OUTPUT_TODAYS = os.path.join(OUTPUT_FOLDER, TODAYS_NAME)
SATTABOSS_TODAYS = os.path.join(SATTABOSS_DATA, TODAYS_NAME)
KALYAN_ASSETS_TODAYS = os.path.join(KALYAN_ASSETS, TODAYS_NAME)
FLUTTER_ASSETS_TODAYS = os.path.join(FLUTTER_PROJECT_ASSETS, TODAYS_NAME)
# --- ADD-ON: ensure also saving inside the actual project folder ---
PROJECT_HISTORY = os.path.join(os.path.dirname(__file__), "output", "history")
PROJECT_TODAYS = os.path.join(os.path.dirname(__file__), "output", TODAYS_NAME)

YESTERDAY_POINTER = os.path.join(OUTPUT_FOLDER, "yesterday_results.json")

ROOT_PRED = os.path.join(ROOT, "predictions_v4.json")
OUTPUT_PRED = os.path.join(OUTPUT_FOLDER, "predictions_v4.json")

# ---------------- SETTINGS ----------------
DAILY_SEED = True
VOLATILE_SEED = False
OPEN_PANNAS_COUNT = 7
CLOSE_PANNAS_COUNT = 7
JODIS_COUNT = 6

# ---------------- PANNA REGISTRY ----------------
VALID_PANNAS_BY_LAST_DIGIT = {
    0: ["118", "127", "136", "145", "190", "226", "235", "244", "280", "334", "370", "460", "550", "299", "389", "479", "488", "569", "578", "668", "677"],
    1: ["100", "119", "128", "137", "146", "155", "227", "236", "245", "290", "335", "344", "380", "470", "560", "399", "489", "579", "588", "669", "678"],
    2: ["110", "200", "129", "138", "147", "156", "228", "237", "246", "255", "336", "345", "390", "480", "570", "660", "499", "589", "679", "688", "778"],
    3: ["777", "444", "120", "300", "111", "139", "148", "157", "166", "229", "238", "247", "256", "337", "346", "355", "445", "490", "580", "670", "599", "689", "779", "788"],
    4: ["112", "130", "220", "400", "149", "158", "167", "239", "248", "257", "266", "338", "347", "356", "446", "455", "590", "680", "770", "699", "789"],
    5: ["113", "122", "140", "230", "500", "159", "168", "177", "249", "258", "267", "339", "348", "357", "366", "447", "456", "690", "780", "799", "889"],
    6: ["888", "555", "114", "123", "150", "240", "330", "600", "222", "169", "178", "259", "268", "277", "349", "358", "367", "448", "457", "466", "556", "790", "880", "899"],
    7: ["115", "124", "133", "160", "223", "250", "340", "700", "179", "188", "269", "278", "359", "368", "377", "449", "458", "467", "557", "566", "890"],
    8: ["116", "125", "134", "170", "224", "233", "260", "350", "440", "800", "189", "279", "288", "369", "378", "459", "468", "477", "558", "567", "990"],
    9: ["199", "289", "379", "388", "469", "478", "559", "568", "577", "667", "999", "666", "117", "126", "135", "144", "180", "225", "234", "270", "360", "450", "900", "333"],
}
FLATTENED_VALID_PANNAS = {p for lst in VALID_PANNAS_BY_LAST_DIGIT.values() for p in lst}


# ---------------- UTIL HELPERS ----------------
def load_json_safe(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logging.warning("Failed to load JSON %s : %s", path, e)
    return None


def save_json_safe(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def safe_last_digit(value: Any) -> Optional[str]:
    """Return last digit character of value or None if not found."""
    if value is None:
        return None
    s = str(value).strip()
    for ch in reversed(s):
        if ch.isdigit():
            return ch
    return None


def run_salt() -> int:
    return (int(time.time() * 1_000_000) ^ secrets.randbits(20)) & 0x7FFFFFFF


def base_seed(market: str, extra: int = 0) -> int:
    b = abs(hash(str(market))) + int(extra)
    if DAILY_SEED:
        b += int(datetime.now().strftime("%Y%m%d"))
    if VOLATILE_SEED:
        b += run_salt()
    return b


def market_rng(market: str, extra: int = 0) -> random.Random:
    return random.Random(base_seed(market, extra))


def dedupe_recent(seq: List[str], n: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in reversed(seq):
        if x not in seen:
            seen.add(x)
            out.append(x)
    return list(reversed(out))[:n]


# ---------------- ARCHIVAL HELPERS ----------------
def _extract_date_from_predictions(pred_obj: Dict[str, Any]) -> Optional[str]:
    try:
        meta = pred_obj.get("meta", {}) if isinstance(pred_obj, dict) else {}
        gen = meta.get("generated_at") or meta.get("generatedAt") or meta.get("generatedAtUtc")
        if gen:
            try:
                dt = datetime.fromisoformat(gen)
                return dt.date().isoformat()
            except Exception:
                pass
        data = pred_obj.get("data", []) if isinstance(pred_obj, dict) else []
        if data and isinstance(data, list):
            first = data[0]
            if isinstance(first, dict):
                preds = first.get("predictions", {}) or {}
                d = preds.get("Date") or preds.get("date")
                if d:
                    try:
                        datetime.strptime(d, "%Y-%m-%d")
                        return d
                    except Exception:
                        pass
    except Exception:
        pass
    return None


def archive_existing_todays() -> Optional[str]:
    """Archive existing todays_predictions.json safely and update yesterday pointer only if valid."""
    if not os.path.exists(ROOT_TODAYS):
        return None

    existing = load_json_safe(ROOT_TODAYS)
    date_str = _extract_date_from_predictions(existing) if existing else None

    # Try to extract a proper date, fallback to yesterday if unknown
    if not date_str:
        try:
            mtime = os.path.getmtime(ROOT_TODAYS)
            dt = datetime.fromtimestamp(mtime)
            today = datetime.now().date()
            fdate = dt.date()
            date_str = (today - timedelta(days=1)).isoformat() if fdate >= today else fdate.isoformat()
        except Exception:
            date_str = (datetime.now().date() - timedelta(days=1)).isoformat()

    os.makedirs(OUTPUT_HISTORY, exist_ok=True)
    archive_path = os.path.join(OUTPUT_HISTORY, f"predictions_{date_str}.json")

    # Step 1: Archive today's predictions file
    try:
        shutil.copy2(ROOT_TODAYS, archive_path)
        logging.info("Archived existing todays_predictions.json -> %s", archive_path)
    except Exception as e:
        logging.warning("Failed to archive todays_predictions.json : %s", e)
        return None

    # Step 2: Safely update yesterday pointer only if archive isn't placeholder
    try:
        with open(archive_path, "r", encoding="utf-8") as f:
            content = f.read()

        # skip if file has placeholder data like "000" or "None"
        if '"000"' not in content and '"None"' not in content:
            tmp_fd, tmp_path = tempfile.mkstemp(dir=OUTPUT_FOLDER)
            os.close(tmp_fd)
            shutil.copy2(archive_path, tmp_path)
            os.replace(tmp_path, YESTERDAY_POINTER)
            logging.info("✅ Updated yesterday pointer -> %s", YESTERDAY_POINTER)
        else:
            logging.warning("⚠️ Skipped yesterday update: archive had placeholder pannas (000 or None)")
    except Exception as e:
        logging.warning("Failed to update yesterday pointer safely: %s", e)

    return archive_path
# ---------------- HISTORY LOAD ----------------
def find_history() -> (Dict[str, Any], Optional[str]):
    for p in HISTORY_CANDIDATES:
        d = load_json_safe(p)
        if d is not None:
            logging.info("Loaded history from %s", p)
            return d, p
    logging.warning("No history found; returning empty dict.")
    return {}, None


# ---------------- PANNAS SELECTION ----------------
def ensure_pannas_for_digits(digits: List[str], hist_pannas: List[str], want_count: int) -> List[str]:
    used: List[str] = list(hist_pannas or [])
    for d in (digits or []):
        ds = str(d)
        if any(p for p in used if safe_last_digit(p) == ds):
            continue
        candidates = VALID_PANNAS_BY_LAST_DIGIT.get(int(ds) if ds.isdigit() else -1, [])
        for cand in candidates:
            if cand not in used:
                used.insert(0, cand)
                break
    valid = [p for p in used if p in FLATTENED_VALID_PANNAS]
    # top up deterministically
    if len(valid) < want_count:
        for digit in range(10):
            for cand in VALID_PANNAS_BY_LAST_DIGIT.get(digit, []):
                if cand not in valid:
                    valid.append(cand)
                if len(valid) >= want_count:
                    break
            if len(valid) >= want_count:
                break
    return valid[:want_count]


# ---------------- PREDICTION GENERATOR ----------------
def generate_predictions(history: Dict[str, Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {"meta": {"generated_at": datetime.now().isoformat()}, "markets": []}

    # history is expected to be mapping: market -> list(records)
    for market, records in list(history.items()):
        try:
            recs = records[-12:] if isinstance(records, list) else []
            od_hist: List[str] = []
            cd_hist: List[str] = []
            j_hist: List[str] = []
            op_hist: List[str] = []
            cp_hist: List[str] = []

            for r in recs:
                if not r or not isinstance(r, dict):
                    continue
                # permissive keys
                od = r.get("Open Digit") or r.get("open_digit") or r.get("opendigit")
                cd = r.get("Close Digit") or r.get("close_digit") or r.get("closedigit")
                jp = r.get("Jodi") or r.get("jodi") or r.get("JODI")
                op = r.get("Open Panna") or r.get("open_panna") or r.get("OpenPanna")
                cp = r.get("Close Panna") or r.get("close_panna") or r.get("ClosePanna")

                # robust open digit extraction
                if od is not None:
                    d = safe_last_digit(od)
                    if d is not None:
                        od_hist.append(str(d))
                elif op:
                    d = safe_last_digit(op)
                    if d:
                        od_hist.append(d)

                # robust close digit extraction
                if cd is not None:
                    d = safe_last_digit(cd)
                    if d is not None:
                        cd_hist.append(str(d))
                elif cp:
                    d = safe_last_digit(cp)
                    if d:
                        cd_hist.append(d)

                # jodi history
                if jp:
                    s = "".join(ch for ch in str(jp) if ch.isdigit())[-2:].zfill(2) if any(ch.isdigit() for ch in str(jp)) else None
                    if s and s.isdigit():
                        j_hist.append(s)

                if op:
                    op_hist.append(str(op))
                if cp:
                    cp_hist.append(str(cp))

            # build digit lists
            open_digits = dedupe_recent(od_hist, 5)
            close_digits = dedupe_recent(cd_hist, 5)

            # fallbacks from pannas if digits empty
            if not open_digits and op_hist:
                for p in reversed(op_hist):
                    d = safe_last_digit(p)
                    if d:
                        open_digits.append(d)
                    if len(open_digits) >= 3:
                        break
            if not close_digits and cp_hist:
                for p in reversed(cp_hist):
                    d = safe_last_digit(p)
                    if d:
                        close_digits.append(d)
                    if len(close_digits) >= 3:
                        break

            open_digits = list(dict.fromkeys(filter(None, open_digits)))[:3]
            close_digits = list(dict.fromkeys(filter(None, close_digits)))[:3]

            # build pannas
            open_pannas = ensure_pannas_for_digits(open_digits if open_digits else [str(i) for i in range(10)], op_hist, OPEN_PANNAS_COUNT)
            close_pannas = ensure_pannas_for_digits(close_digits if close_digits else [str(i) for i in range(10)], cp_hist, CLOSE_PANNAS_COUNT)

            # jodis: cartesian product of digits then extend with history
            cart: List[str] = []
            for o in open_digits:
                for c in close_digits:
                    cart.append(f"{str(o)[-1]}{str(c)[-1]}")

            rng = market_rng(market, extra=123)
            rng.shuffle(cart)

            j_list: List[str] = []
            seen = set()
            for j in cart:
                s = str(j).zfill(2)[-2:]
                if s not in seen:
                    j_list.append(s)
                    seen.add(s)
            for j in j_hist:
                if j not in seen:
                    j_list.append(j)
                    seen.add(j)
                if len(j_list) >= 10:
                    break
            j_list = j_list[:JODIS_COUNT]


            # --- OPTIONAL RANDOMIZER ONLY FOR AMAR JYOTI ---
            if "amar" in market.lower() and "jyoti" in market.lower():
                import random
                # Shuffle pannas and digits to appear fresh daily
                if isinstance(open_pannas, list) and len(open_pannas) > 1:
                    random.shuffle(open_pannas)
                if isinstance(close_pannas, list) and len(close_pannas) > 1:
                    random.shuffle(close_pannas)
                if isinstance(open_digits, list) and len(open_digits) > 1:
                    random.shuffle(open_digits)
                if isinstance(close_digits, list) and len(close_digits) > 1:
                    random.shuffle(close_digits)
                if isinstance(j_list, list) and len(j_list) > 1:
                    random.shuffle(j_list)

                logging.info(f"✨ Randomized Amar Jyoti predictions ({len(open_pannas)} open / {len(close_pannas)} close)")



            results["markets"].append({
                "market": market,
                "predictions": {
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "Open Digits": open_digits,
                    "Close Digits": close_digits,
                    "Open Pannas": open_pannas,
                    "Close Pannas": close_pannas,
                    "Jodis": j_list
                }
            })
        except Exception as e:
            logging.exception("Error generating for market %s : %s", market, e)

    results["meta"]["generated_at"] = datetime.now().isoformat()
    return results


# ---------------- MAIN ----------------
def _parse_bt_args():
    """Backtest lab: this engine was written without argparse; accept the
    harness contract '--date YYYY-MM-DD' (and ignore unknown flags) so it can
    be time-travelled instead of always predicting 'today'."""
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--date", default=None)
    ns, _unknown = ap.parse_known_args()
    return ns


def main(bt_date: Optional[str] = None) -> List[str]:
    bt_date = bt_date or _parse_bt_args().date
    # ensure directories exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_HISTORY, exist_ok=True)
    os.makedirs(KALYAN_FOLDER, exist_ok=True)
    os.makedirs(SATTABOSS_DATA, exist_ok=True)
    os.makedirs(KALYAN_ASSETS, exist_ok=True)
    try:
        os.makedirs(FLUTTER_PROJECT_ASSETS, exist_ok=True)
    except Exception:
        pass

    # ---- load history (prefer sattaboss-data/data) ----
    history: Dict[str, Any] = {}
    history_path_used: Optional[str] = None
    # Backtest lab: if a copy of the history sits next to this script (cwd), use it
    # instead of the hardcoded production paths outside the repo.
    _bt_hist = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_markets_history.json")
    candidates = ([_bt_hist] if os.path.exists(_bt_hist) else []) + HISTORY_CANDIDATES
    for p in candidates:
        if os.path.exists(p):
            history = load_json_safe(p) or {}
            history_path_used = p
            logging.info("✅ Using history file: %s (markets=%d)", p, len(history) if isinstance(history, dict) else 0)
            break
    if not history:
        logging.warning("⚠️ No history file found in candidates — using empty history (no markets).")

    # ---- time travel: cut history at --date so predictions never leak the future ----
    if bt_date:
        try:
            cutoff = datetime.strptime(bt_date, "%Y-%m-%d").date()
        except ValueError:
            cutoff = None
        if cutoff is not None:
            def _rec_date(r):
                d = str((r or {}).get("Date") or (r or {}).get("date") or "").strip()
                parts = d.replace("/", "-").split("-")
                if len(parts) == 3:
                    try:
                        if len(parts[0]) == 4:
                            return datetime.strptime(d, "%Y-%m-%d").date()
                        day, mon = int(parts[0]), int(parts[1])   # DD/MM/YYYY (lab convention)
                        if mon > 12:                              # tolerate MM/DD/YYYY
                            day, mon = mon, day
                        return datetime(int(parts[2]), mon, day).date()
                    except ValueError:
                        return None
                return None
            trimmed = {}
            for mkt, recs in history.items():
                if isinstance(recs, list):
                    keep = [r for r in recs if (rd := _rec_date(r)) is not None and rd < cutoff]
                    trimmed[mkt] = keep
                else:
                    trimmed[mkt] = recs
            history = trimmed
            logging.info("⏪ Backtest mode: history truncated to dates before %s", bt_date)

    # ---- archive existing todays_predictions.json -> yesterday pointer ----
    try:
        archived = archive_existing_todays()
        if archived:
            logging.info("Archived previous todays_predictions.json -> %s", archived)
    except Exception as e:
        logging.warning("Archive step failed: %s", e)

    # ---- generate predictions ----
    preds = generate_predictions(history)

# ---- save to multiple destinations ----
    destinations = [
        ROOT_TODAYS, OUTPUT_TODAYS, SATTABOSS_TODAYS,
        KALYAN_ASSETS_TODAYS, FLUTTER_ASSETS_TODAYS,
        OUTPUT_HISTORY_TODAYS, ROOT_PRED, OUTPUT_PRED,
        PROJECT_TODAYS,  # add-on: today's file inside repo
        os.path.join(PROJECT_HISTORY, TODAYS_NAME)  # add-on: history inside repo
    ]

    saved_paths: List[str] = []
    for path in destinations:
        try:
            save_json_safe(path, preds)
            saved_paths.append(path)
        except Exception as e:
            logging.warning("Failed to save %s: %s", path, e)

    # also archive today's predictions copy
    try:
        today_str = datetime.now().date().isoformat()
        archive_name = f"predictions_{today_str}.json"
        shutil.copy2(ROOT_TODAYS, os.path.join(OUTPUT_HISTORY, archive_name))
        # also copy archive into project history folder (if different)
        try:
            os.makedirs(PROJECT_HISTORY, exist_ok=True)
            shutil.copy2(ROOT_TODAYS, os.path.join(PROJECT_HISTORY, archive_name))
        except Exception:
            # non-fatal if project-history copy fails
            pass
        logging.info("Saved today's predictions archive -> %s", archive_name)
    except Exception as e:
        logging.warning("Failed to save today's archive: %s", e)

    # ---- summary output ----
    print("\n✅ Written todays_predictions.json to:")
    for p in saved_paths:
        print(" -", p)
    if not saved_paths:
        print("⚠️ No files were saved — check permissions and paths.")
    return saved_paths


if __name__ == "__main__":
    main()

# ==== BACKTEST LAB 1-LINER: unique dated output -> D:\backtest\predictions_<YYYY-MM-DD>_v4.json ====
try:
    import sys as _bt_sys, re as _bt_re, json as _bt_json, os as _bt_os, datetime as _bt_dt
    from pathlib import Path as _bt_Path
    def _bt_write(out_obj, d=None):
        # Resolve the prediction date: engine meta -> per-market "Date" -> CLI flag -> passed date -> today.
        _m = (out_obj or {}).get("meta") or {} if isinstance(out_obj, dict) else {}
        ds = str(_m.get("prediction_date") or _m.get("run_date") or _m.get("date") or _m.get("target_date") or "")
        if not _bt_re.fullmatch(r"\d{4}-\d{2}-\d{2}", ds) and isinstance(out_obj, dict):
            _mlist = out_obj.get("markets") or []
            if isinstance(_mlist, dict):
                _cand = str(_mlist.get("Date") or "")
                if _bt_re.fullmatch(r"\d{4}-\d{2}-\d{2}", _cand):
                    ds = _cand
            elif isinstance(_mlist, list):
                for _mk in _mlist:
                    _cand = str((_mk.get("predictions") or {}).get("Date") if isinstance(_mk, dict) else "")
                    if _bt_re.fullmatch(r"\d{4}-\d{2}-\d{2}", _cand):
                        ds = _cand; break
        if not _bt_re.fullmatch(r"\d{4}-\d{2}-\d{2}", ds):
            _argv = list(_bt_sys.argv)
            for _i, _a in enumerate(_argv):
                if _a.startswith("--test-date") or _a.startswith("--date"):
                    _v = _a.split("=", 1)[1] if "=" in _a else (_argv[_i+1] if _i+1 < len(_argv) else "")
                    if _bt_re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(_v)):
                        ds = _v; break
        if not _bt_re.fullmatch(r"\d{4}-\d{2}-\d{2}", ds):
            ds = d.isoformat() if isinstance(d, _bt_dt.date) else _bt_dt.date.today().isoformat()
        _bt_dir = _bt_Path(_bt_os.environ.get("LAB_DIR") or ("D:\\backtest" if (_bt_os.name == "nt" or _bt_os.path.splitdrive("D:\\")[0]) else _bt_Path(__file__).resolve().as_posix()))
        _bt_dir.mkdir(parents=True, exist_ok=True)
        _p = _bt_dir / f"predictions_{ds}_v4.json"
        _t = _p.with_suffix(".json.tmp")
        _t.write_text(_bt_json.dumps(out_obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        _bt_os.replace(_t, _p)
        print(f"[BACKTEST] wrote {_p}")
    _bt_obj = None
    for _bt_k in ('app_json', 'output', 'result', 'payload', 'data', 'preds', 'predictions'):
        _bt_v = globals().get(_bt_k)
        if isinstance(_bt_v, (dict, list)) and _bt_v:
            _bt_obj = _bt_v; break
    if _bt_obj is not None:
        _bt_write(_bt_obj)
except Exception as _bt_e:
    print(f"[BACKTEST] skipped: {_bt_e}")
# ==== END BACKTEST LAB 1-LINER ====
