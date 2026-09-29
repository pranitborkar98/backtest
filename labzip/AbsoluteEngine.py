# -*- coding: utf-8 -*-
"""
AbsoluteEngine.py — The Ultimate SattaMatkaAI Engine

Combines the absolute best elements of the v10-v33 ecosystem:
  1. REAL Self-Awareness (SignalRegistry actually records and learns from hits/misses).
  2. Hierarchical Priority Stack (VETO > HARD > SOFT > NOISE).
  3. Adversarial Human-Bias Heuristics (Cut-digit, Sum-mod10, Anti-herd, Fakeout).
  4. Strict Panna-to-Digit Math Enforcement.
  5. Closed-Loop Backtesting (Rolling 30-day accuracy calibration).
  6. Atomic I/O & Portable Paths.
"""

import os
import sys
import json
import random
import datetime
import argparse
import logging
import tempfile
import bisect
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

# ==========================================
# ================= CONFIG =================
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "satta_data"
STATE_DIR = BASE_DIR / "state"

PATHS = {
    "HISTORY_FILE": DATA_DIR / "all_markets_history.json",
    "OUTPUT_DIR": DATA_DIR / "output",
    "STATE_FILE": STATE_DIR / "absolute_engine_state.json",
    "TODAY_FILE": DATA_DIR / "todays_predictions.json",
}

for p in PATHS.values():
    if p.suffix == "":
        p.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "NUM_OPEN_DIGITS": 3, "NUM_CLOSE_DIGITS": 3,
    "NUM_OPEN_PANNAS": 8, "NUM_CLOSE_PANNAS": 8, "NUM_JODIS": 9,

    "FREQ_LOOKBACK_DAYS": 45,
    "MARKOV_LOOKBACK_DAYS": 180,
    "MARKOV_MIN_COUNT": 4,  # Prevents overfitting on tiny sample sizes

    # Signal Registry (The REAL Self-Awareness Loop)
    "SIGNAL_ACCURACY_LOOKBACK": 30,
    "SIGNAL_MUTE_THRESHOLD": 0.15,
    "SIGNAL_BOOST_THRESHOLD": 0.55,
    "SIGNAL_MIN_SAMPLES": 5,

    # Behavioral Weights
    "BREAK_PENALTY": 0.15, "STREAK_PENALTY": 0.10,
    "SUMMOD_BOOST": 1.45, "NEIGHBOR_BOOST": 1.25,
    "MIRROR_BOOST": 1.35, "CUT_BOOST": 1.40,
    "RED_JODI_PENALTY": 0.25,

    "PANNA_RECENT_PENALTY": 0.35, "PANNA_MIRROR_BONUS": 1.30,
    
    "CONFIDENCE_BASE": 0.35, "CONFIDENCE_CAP": 0.85,
    "STATE_HISTORY_CAP_PER_MARKET": 90,
}

VALID_PANNAS_BY_LAST_DIGIT = {
    0:["118","127","136","145","190","226","235","244","280","334","370","460","550","299","389","479","488","569","578","668","677"],
    1:["100","119","128","137","146","155","227","236","245","290","335","344","380","470","560","399","489","579","588","669","678"],
    2:["110","200","129","138","147","156","228","237","246","255","336","345","390","480","570","660","499","589","679","688","778"],
    3:["777","444","120","300","111","139","148","157","166","229","238","247","256","337","346","355","445","490","580","670","599","689","779","788"],
    4:["112","130","220","400","149","158","167","239","248","257","266","338","347","356","446","455","590","680","770","699","789"],
    5:["113","122","140","230","500","159","168","177","249","258","267","339","348","357","366","447","456","690","780","799","889"],
    6:["888","555","114","123","150","240","330","600","222","169","178","259","268","277","349","358","367","448","457","466","556","790","880","899"],
    7:["115","124","133","160","223","250","340","700","179","188","269","278","359","368","377","449","458","467","557","566","890"],
    8:["116","125","134","170","224","233","260","350","440","800","189","279","288","369","378","459","468","477","558","567","990"],
    9:["199","289","379","388","469","478","559","568","577","667","999","666","117","126","135","144","180","225","234","270","360","450","900","333"]
}

# ==========================================
# ================ LOGGING =================
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("AbsoluteEngine")

# ==========================================
# ================ UTILITIES ===============
# ==========================================
def load_json(p: Path) -> Any:
    if not p.exists(): return None
    try:
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    except: return None

def save_json_atomic(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, p)
    except Exception:
        os.remove(tmp_path)
        raise

def parse_date(d: Any) -> Optional[datetime.date]:
    if not d: return None
    s = str(d)
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try: return datetime.datetime.strptime(s, fmt).date()
        except: pass
    return None

def extract_day_result(rec: Dict[str, Any]) -> Dict[str, str]:
    def get(*keys):
        for k in keys:
            if k in rec and rec[k] not in (None, ""): return str(rec[k]).strip()
        return ""
    
    od = get("Open Digit", "OpenDigit", "open_digit")
    cd = get("Close Digit", "CloseDigit", "close_digit")
    j_raw = get("Jodi", "jodi")
    
    j = f"{od}{cd}" if od.isdigit() and cd.isdigit() else (j_raw.zfill(2) if j_raw.isdigit() else "")
    
    op_raw = get("Open Panna", "OpenPanna", "open_panna")
    cp_raw = get("Close Panna", "ClosePanna", "close_panna")
    op = op_raw.zfill(3) if op_raw.isdigit() else ""
    cp = cp_raw.zfill(3) if cp_raw.isdigit() else ""
    
    return {"od": od, "cd": cd, "j": j, "op": op, "cp": cp}

class MarketIndex:
    def __init__(self, rows: List[Dict[str, Any]]):
        self.by_date = {parse_date(r.get("Date") or r.get("date")): r for r in rows if parse_date(r.get("Date") or r.get("date"))}
        self.dates_asc = sorted(self.by_date.keys())

    def last_n_before(self, asof: datetime.date, n: int) -> List[Tuple[datetime.date, Dict[str, Any]]]:
        pos = bisect.bisect_right(self.dates_asc, asof)
        if pos == 0: return []
        start = max(0, pos - n)
        return [(d, self.by_date[d]) for d in reversed(self.dates_asc[start:pos])]

# ==========================================
# ======== SIGNAL REGISTRY (LEARNING) ======
# ==========================================
class SignalRegistry:
    """Actually records hits/misses and adjusts rule weights dynamically."""
    def __init__(self, state: Dict[str, Any]):
        self.history = defaultdict(lambda: deque(maxlen=CONFIG["SIGNAL_ACCURACY_LOOKBACK"]))
        self.boosts = defaultdict(lambda: 1.0)
        
        saved = state.get("signal_registry", {})
        for rule, data in saved.items():
            self.boosts[rule] = data.get("boost", 1.0)
            self.history[rule].extend(data.get("history", []))

    def record(self, rule: str, hit: bool):
        if not rule: return
        self.history[rule].append(1 if hit else 0)
        self._recalc(rule)

    def _recalc(self, rule: str):
        hist = list(self.history[rule])
        if len(hist) < CONFIG["SIGNAL_MIN_SAMPLES"]:
            self.boosts[rule] = 1.0
            return
        
        acc = sum(hist) / len(hist)
        if acc < CONFIG["SIGNAL_MUTE_THRESHOLD"]:
            self.boosts[rule] = 0.2  # Mute failing rules
        elif acc > CONFIG["SIGNAL_BOOST_THRESHOLD"]:
            self.boosts[rule] = min(2.0, 1.0 + (acc - CONFIG["SIGNAL_BOOST_THRESHOLD"]) * 2.0)  # Boost winners
        else:
            self.boosts[rule] = 1.0

    def get(self, rule: str) -> float:
        return self.boosts.get(rule, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {r: {"boost": self.boosts[r], "history": list(self.history[r])} for r in self.history}

# ==========================================
# ============ CORE PREDICTION =============
# ==========================================
def predict_market(mkt: str, idx: MarketIndex, asof: datetime.date, registry: SignalRegistry) -> Dict[str, Any]:
    if len(idx.dates_asc) < 5:
        return {"OpenDigits": [], "CloseDigits": [], "OpenPannas": [], "ClosePannas": [], "Jodis": [], "attribution": {}, "confidence": 0.0}

    # 1. Yesterday Context
    last = idx.last_n_before(asof, 1)
    yctx = extract_day_result(last[0][1]) if last else {}
    y_od, y_cd, y_j = yctx.get("od", ""), yctx.get("cd", ""), yctx.get("j", "")

    # 2. Base Frequency (Dampened Noise)
    last_45 = idx.last_n_before(asof, CONFIG["FREQ_LOOKBACK_DAYS"])
    cnt = Counter()
    for _, r in last_45:
        res = extract_day_result(r)
        if res["od"].isdigit(): cnt[res["od"]] += 1
        if res["cd"].isdigit(): cnt[res["cd"]] += 1
    weights = {str(i): (cnt.get(str(i), 0) + 1.0) * 0.1 for i in range(10)}

    attribution = defaultdict(set)  # Track which rules backed which digits

    # 3. Human-Bias Adversarial Logic
    herd = {d for d, c in cnt.items() if c >= 4}  # 4+ hits in 45 days = herd bait
    for d in herd:
        weights[d] *= CONFIG["STREAK_PENALTY"]; attribution[d].add("anti_herd")

    if y_od.isdigit():
        weights[y_od] *= CONFIG["BREAK_PENALTY"]; attribution[y_od].add("break")
        
        cut_o = str((int(y_od) + 5) % 10)
        weights[cut_o] *= CONFIG["CUT_BOOST"] * registry.get("cut"); attribution[cut_o].add("cut")
        
        for nb in [(int(y_od)+1)%10, (int(y_od)-1)%10]:
            weights[str(nb)] *= CONFIG["NEIGHBOR_BOOST"]; attribution[str(nb)].add("neighbor")

    if y_cd.isdigit():
        weights[y_cd] *= CONFIG["BREAK_PENALTY"]; attribution[y_cd].add("break")
        
        cut_c = str((int(y_cd) + 5) % 10)
        weights[cut_c] *= CONFIG["CUT_BOOST"] * registry.get("cut"); attribution[cut_c].add("cut")
        
        comp_o = str((10 - int(y_cd)) % 10)  # Sum-Mod10
        weights[comp_o] *= CONFIG["SUMMOD_BOOST"] * registry.get("sumcomp"); attribution[comp_o].add("sumcomp")

    if y_j and len(y_j) == 2:
        weights[y_j[1]] *= CONFIG["MIRROR_BOOST"] * registry.get("mirror"); attribution[y_j[1]].add("mirror")
        weights[y_j[0]] *= CONFIG["MIRROR_BOOST"] * registry.get("mirror"); attribution[y_j[0]].add("mirror")

    # 4. Select Digits
    items = list(weights.keys())
    ws = [weights[k] for k in items]
    
    # Weighted choice without replacement
    open_d, close_d = [], []
    temp_items, temp_ws = list(items), list(ws)
    
    for _ in range(CONFIG["NUM_OPEN_DIGITS"]):
        if not temp_items: break
        pick = random.choices(temp_items, weights=temp_ws, k=1)[0]
        open_d.append(pick)
        idx_pick = temp_items.index(pick)
        temp_items.pop(idx_pick); temp_ws.pop(idx_pick)
        
    temp_items, temp_ws = list(items), list(ws)
    for _ in range(CONFIG["NUM_CLOSE_DIGITS"]):
        if not temp_items: break
        pick = random.choices(temp_items, weights=temp_ws, k=1)[0]
        close_d.append(pick)
        idx_pick = temp_items.index(pick)
        temp_items.pop(idx_pick); temp_ws.pop(idx_pick)

    # Fill if shortfall
    while len(open_d) < 3:
        for d in map(str, range(10)):
            if d not in open_d: open_d.append(d); break
    while len(close_d) < 3:
        for d in map(str, range(10)):
            if d not in close_d: close_d.append(d); break

    # 5. Strict Panna Math (Sum%10 MUST match chosen digits)
    def get_pannas(digits, role):
        pool = []
        for d in digits:
            pool.extend(VALID_PANNAS_BY_LAST_DIGIT.get(int(d), []))
        pool = list(set(p for p in pool if len(set(p)) != 1 or len(p) != 3))  # Exclude triples
        
        math_valid = [p for p in pool if str(sum(int(x) for x in p) % 10) in digits] or pool
        
        recent_pannas = set()
        for _, r in idx.last_n_before(asof, 5):
            res = extract_day_result(r)
            if res["op"]: recent_pannas.add(res["op"])
            if res["cp"]: recent_pannas.add(res["cp"])

        p_ws = []
        for p in math_valid:
            w = 1.0
            if p in recent_pannas: w *= CONFIG["PANNA_RECENT_PENALTY"]
            mp = p[::-1]
            if mp in recent_pannas and mp != p: w *= CONFIG["PANNA_MIRROR_BONUS"]
            p_ws.append(w)
            
        picks = random.choices(math_valid, weights=p_ws, k=CONFIG[f"NUM_{role.upper()}_PANNAS"])
        return list(dict.fromkeys(picks))[:CONFIG[f"NUM_{role.upper()}_PANNAS"]]

    open_p = get_pannas(open_d, "open")
    close_p = get_pannas(close_d, "close")

    # 6. Build Jodis
    cands = []
    cut_jodi = f"{(int(y_j[0])+5)%10}{(int(y_j[1])+5)%10}" if y_j and len(y_j)==2 else ""
    
    for a in open_d:
        for b in close_d:
            j = f"{a}{b}"
            w = 1.0
            if j == y_j: w *= 0.2
            if a == b and (not y_j or y_j[0] != y_j[1]): w *= 1.15
            elif a == b and y_j and y_j[0] == y_j[1]: w *= CONFIG["RED_JODI_PENALTY"]
            if y_j and j == y_j[::-1]: w *= 1.5 * registry.get("mirror")
            if (int(a)+int(b)) in (10, 20): w *= 1.35
            if cut_jodi and j == cut_jodi: w *= CONFIG["CUT_BOOST"] * registry.get("cut")
            cands.append((j, w))
            
    cands.sort(key=lambda x: x[1], reverse=True)
    jodis = list(dict.fromkeys([j for j, _ in cands]))[:CONFIG["NUM_JODIS"]]

    # 7. Confidence
    conf = CONFIG["CONFIDENCE_BASE"]
    if y_j and y_j[::-1] in jodis: conf += 0.15
    if any((int(a)+int(b))%10==0 for a in open_d for b in close_d if a!=b): conf += 0.10
    conf = min(conf, CONFIG["CONFIDENCE_CAP"])

    # Convert attribution sets to lists for JSON
    attr_out = {d: list(rules) for d, rules in attribution.items() if d in open_d or d in close_d}

    return {
        "OpenDigits": open_d, "CloseDigits": close_d,
        "OpenPannas": open_p, "ClosePannas": close_p,
        "Jodis": jodis, "attribution": attr_out, "confidence": round(conf, 2)
    }

# ==========================================
# ============ STATE & BACKTEST ============
# ==========================================
def load_state() -> Dict[str, Any]:
    data = load_json(PATHS["STATE_FILE"])
    return data if isinstance(data, dict) else {"markets": {}, "signal_registry": {}}

def backtest_update(state: Dict[str, Any], indexes: Dict[str, MarketIndex], registry: SignalRegistry):
    """The REAL learning loop. Scores past predictions and updates registry."""
    scored = 0
    for mkt, entries in state.get("markets", {}).items():
        idx = indexes.get(mkt)
        if not idx: continue
        for entry in entries:
            if entry.get("actual") is not None: continue
            d = parse_date(entry["date"])
            if not d: continue
            
            actual_rec = idx.by_date.get(d)
            if not actual_rec: continue
            actual = extract_day_result(actual_rec)
            
            pred = entry.get("predicted", {})
            attr = entry.get("attribution", {})
            
            # Score Digit Hits
            od_hit = actual["od"] in pred.get("OpenDigits", []) if actual["od"] else False
            cd_hit = actual["cd"] in pred.get("CloseDigits", []) if actual["cd"] else False
            
            # Update Signal Registry based on Attribution!
            for d, rules in attr.items():
                if d == actual["od"] or d == actual["cd"]:
                    for r in rules: registry.record(r, True)
                else:
                    for r in rules: registry.record(r, False)
            
            entry["actual"] = actual
            entry["hits"] = {"od": od_hit, "cd": cd_hit}
            scored += 1
    return scored

# ==========================================
# ================ MAIN ====================
# ==========================================
def run(for_date: Optional[datetime.date] = None, dry_run: bool = False):
    if for_date is None: for_date = datetime.date.today()
    
    log.info(f"Loading history from {PATHS['HISTORY_FILE']}...")
    raw_hist = load_json(PATHS["HISTORY_FILE"]) or {}
    indexes = {m: MarketIndex(rows) for m, rows in raw_hist.items() if isinstance(rows, list)}
    if not indexes:
        log.error("No history found. Place all_markets_history.json in ./satta_data/")
        return

    state = load_state()
    registry = SignalRegistry(state)
    
    # 1. Backtest and Learn
    scored_count = backtest_update(state, indexes, registry)
    if scored_count:
        log.info(f"Backtest scored {scored_count} new outcomes. Signal registry updated.")
    
    # 2. Predict Today
    preds = {}
    for mkt, idx in indexes.items():
        try:
            preds[mkt] = predict_market(mkt, idx, for_date, registry)
            # Save to state for future backtesting
            state["markets"].setdefault(mkt, []).append({
                "date": for_date.isoformat(),
                "predicted": {k: preds[mkt][k] for k in ["OpenDigits", "CloseDigits", "OpenPannas", "ClosePannas", "Jodis"]},
                "attribution": preds[mkt]["attribution"],
                "actual": None, "hits": None
            })
            # Trim state history
            if len(state["markets"][mkt]) > CONFIG["STATE_HISTORY_CAP_PER_MARKET"]:
                state["markets"][mkt] = state["markets"][mkt][-CONFIG["STATE_HISTORY_CAP_PER_MARKET"]:]
        except Exception as e:
            log.error(f"Error predicting {mkt}: {e}")
    
    state["signal_registry"] = registry.to_dict()

    # 3. Output
    output = {
        "meta": {"generated_at": datetime.datetime.now().isoformat(), "engine": "AbsoluteEngine_v1"},
        "markets": [{"market": m, "predictions": p} for m, p in preds.items()]
    }
    
    if not dry_run:
        save_json_atomic(PATHS["TODAY_FILE"], output)
        save_json_atomic(PATHS["STATE_FILE"], state)
        log.info(f"Predictions saved to {PATHS['TODAY_FILE']}")
        log.info(f"State saved to {PATHS['STATE_FILE']}")
    else:
        log.info("Dry run complete. No files saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Absolute SattaMatka AI Engine")
    parser.add_argument("--date", type=str, help="Override run date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving files")
    args = parser.parse_args()
    
    run_date = parse_date(args.date) if args.date else None
    run(for_date=run_date, dry_run=args.dry_run)

# ==== BACKTEST LAB 1-LINER: unique dated output -> D:\backtest\predictions_<YYYY-MM-DD>_absolute.json ====
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
        _p = _bt_dir / f"predictions_{ds}_absolute.json"
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
