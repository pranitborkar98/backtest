# -*- coding: utf-8 -*-
"""
prediction_engine_v31.py — SattaMatkaAI (Human-Bias & Adversarial Engine — ROBUST)

Kalyan-Optimized Adversarial Engine.
Updates vs old v31:
  1. 14-Day Lookback Window (FAKEOUT_LOOKBACK = 14, HERD_WINDOW_DAYS = 14).
     Scans the last two weeks for herd digits and repeated fakeout loops instead of just 2 days.
  2. Strict Panna-to-Digit Mathematics. Panna selection is now mathematically forced to match
     the predicted Open/Close digits (sum of panna % 10 == digit).
  3. Cut-Jodi (±5) Boost. If yesterday was 13, today has a high probability of 68. Explicitly boosted.
  4. Retains all v31 robustness: atomic I/O, leading-zero fixes, closed feedback loop, self-test.
"""

import os
import sys
import json
import random
import datetime
import argparse
import logging
import logging.handlers
import tempfile
import bisect
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple, Iterable, Optional, Set

# ==========================================
# ================= CONFIG =================
# ==========================================
_DEFAULTS = {
    "HISTORY_FILE": r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\sattaboss-data\data\all_markets_history.json",
    "OUTPUT_DIR":   r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output",
    "STATE_FILE":   r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\state\engine_v31_state.json",
    "TODAY_TEST_FILE": r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\todays_predictions_v31_TEST.json",
    "TODAY_PROD_FILE": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\todays_predictions.json",
    "TODAYS_PREDICTIONS_JSON": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\todays_predictions.json",
    "LOG_FILE":   r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\engine_v31.log",
    "ASSETS_DIR": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets",
}

_ENV_PREFIX = "SATTA_"
_LOCAL_FALLBACK_DIR = Path(__file__).resolve().parent / "satta_data"


def _resolve_path(key: str, cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value)
    env_val = os.environ.get(_ENV_PREFIX + key)
    if env_val:
        return Path(env_val)
    default = Path(_DEFAULTS[key])
    if default.parent.exists():
        return default
    return _LOCAL_FALLBACK_DIR / default.name


CONFIG = {
    "NUM_OPEN_DIGITS": 3,
    "NUM_CLOSE_DIGITS": 3,
    "NUM_OPEN_PANNAS": 8,
    "NUM_CLOSE_PANNAS": 8,
    "NUM_JODIS": 9,

    "EXCLUDE_TRIPLE_PANNAS": True,

    "FREQ_LOOKBACK_DAYS": 45,
    "HERD_WINDOW_DAYS": 14,      # Increased to 14 days (two weeks)
    "HERD_HOT_THRESHOLD": 4,     # 4+ hits in 14 days = bait
    "FAKEOUT_LOOKBACK": 14,      # Scan 14 days for loop patterns

    "BREAK_PENALTY": 0.15,
    "STREAK_PENALTY": 0.10,
    "SUMMOD_BOOST": 1.45,
    "NEIGHBOR_BOOST": 1.25,
    "MIRROR_BOOST": 1.35,
    "CUT_BOOST": 1.40,
    "RED_JODI_PENALTY": 0.25,

    "PANNA_RECENT_PENALTY": 0.35,
    "PANNA_MIRROR_BONUS": 1.30,
    "PANNA_CUT_BONUS": 1.25,
    "PANNA_SUMMOD_BONUS": 1.20,

    "CONFIDENCE_BASE": 0.35,
    "CONFIDENCE_CAP": 0.80,
    "CONFIDENCE_MIN": 0.05,
    "MIN_BACKTEST_SAMPLES_FOR_BLEND": 5,
    "ACCURACY_LOOKBACK_DAYS": 30,
    "RULE_WEIGHT_WHEN_BLENDED": 0.45,
    "STATE_HISTORY_CAP_PER_MARKET": 90,

    "ENABLE_LOGGING": True,
    "LOG_TO_CONSOLE": False,
    "LOG_MAX_MB": 5,
    "LOG_BACKUPS": 3,

    "TEST_MODE": False,
    "TEST_SEED": 20251109,
}

# ==========================================
# ============ VALID PANEL MAP =============
# ==========================================
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
logger = logging.getLogger("pred_engine_v31")
logger.setLevel(logging.INFO)

def setup_logging(log_file: Path, enable_file: bool, enable_console: bool) -> None:
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if enable_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                str(log_file), maxBytes=CONFIG["LOG_MAX_MB"] * 1024 * 1024,
                backupCount=CONFIG["LOG_BACKUPS"], encoding="utf-8"
            )
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception as e:
            logger.warning(f"Logging file handler failed: {e}")
    if enable_console or not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

# ==========================================
# ================ UTILITIES ===============
# ==========================================
def today_str(dt: Optional[datetime.date] = None) -> str:
    if dt is None: dt = datetime.date.today()
    return dt.strftime("%Y%m%d")

def iso_timestamp() -> str:
    return datetime.datetime.now().isoformat()

def safe_mkdir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def load_json(p: Path) -> Any:
    p = Path(p)
    if not p.exists(): return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to parse JSON at {p}: {e}")
        return None

def save_json_atomic(p: Path, obj: Any) -> None:
    p = Path(p)
    safe_mkdir(p.parent)
    fd, tmp_path = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, p)
    except Exception:
        try: os.remove(tmp_path)
        except OSError: pass
        raise

def parse_date(d: Any) -> Optional[datetime.date]:
    if not d: return None
    s = str(d)
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try: return datetime.datetime.strptime(s, fmt).date()
        except Exception: pass
    return None

def is_triple_panna(p: str) -> bool:
    return isinstance(p, str) and len(p) == 3 and p.isdigit() and len(set(p)) == 1

def mirror_jodi(j: str) -> str:
    return f"{j[1]}{j[0]}" if isinstance(j, str) and len(j) == 2 else j

def mirror_panna(p: str) -> str:
    return p[::-1] if isinstance(p, str) and len(p) == 3 else p

def get_field(rec: Dict[str, Any], *keys: str) -> str:
    if not isinstance(rec, dict): return ""
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return str(rec[k]).strip()
    norm_keys = {k.lower().replace(" ", "").replace("_", ""): k for k in keys}
    for rk, rv in rec.items():
        nk = str(rk).lower().replace(" ", "").replace("_", "")
        if nk in norm_keys and rv not in (None, ""):
            return str(rv).strip()
    return ""

def normalize_jodi(raw_jodi: str, od: str, cd: str) -> str:
    if od.isdigit() and cd.isdigit() and len(od) == 1 and len(cd) == 1:
        return f"{od}{cd}"
    if raw_jodi.isdigit() and len(raw_jodi) <= 2:
        return raw_jodi.zfill(2)
    return ""

def normalize_panna(raw_panna: str) -> str:
    if raw_panna.isdigit() and 1 <= len(raw_panna) <= 3:
        return raw_panna.zfill(3)
    return ""

def extract_day_result(rec: Dict[str, Any]) -> Dict[str, str]:
    y_od = get_field(rec, "Open Digit", "OpenDigit", "open_digit")
    y_cd = get_field(rec, "Close Digit", "CloseDigit", "close_digit")
    y_j_raw = get_field(rec, "Jodi", "jodi")
    y_op = normalize_panna(get_field(rec, "Open Panna", "OpenPanna", "open_panna"))
    y_cp = normalize_panna(get_field(rec, "Close Panna", "ClosePanna", "close_panna"))
    y_j = normalize_jodi(y_j_raw, y_od, y_cd)
    return {"od": y_od, "cd": y_cd, "j": y_j, "op": y_op, "cp": y_cp}

class MarketIndex:
    __slots__ = ("dates_asc", "by_date")

    def __init__(self, rows: List[Dict[str, Any]]):
        by_date: Dict[datetime.date, Dict[str, Any]] = {}
        for r in rows:
            d = parse_date(r.get("Date") or r.get("date"))
            if d: by_date[d] = r
        self.by_date = by_date
        self.dates_asc = sorted(by_date.keys())

    def last_n_on_or_before(self, asof: datetime.date, n: int) -> List[Tuple[datetime.date, Dict[str, Any]]]:
        pos = bisect.bisect_right(self.dates_asc, asof)
        if pos == 0: return []
        start = max(0, pos - n)
        window = self.dates_asc[start:pos]
        return [(d, self.by_date[d]) for d in reversed(window)]

    def row_for_date(self, d: datetime.date) -> Optional[Dict[str, Any]]:
        return self.by_date.get(d)

    def __len__(self) -> int:
        return len(self.by_date)

def last_n_days(rows: List[Dict[str, Any]], asof: datetime.date, n: int) -> List[Tuple[datetime.date, Dict[str, Any]]]:
    dated = []
    for r in rows:
        d = parse_date(r.get("Date") or r.get("date"))
        if d and d <= asof: dated.append((d, r))
    dated.sort(key=lambda x: x[0], reverse=True)
    return dated[:n]

# ==========================================
# ============== DATA INGEST ===============
# ==========================================
def load_market_history(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    data = load_json(path)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, list)}
    if isinstance(data, list):
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in data:
            m = rec.get("Market") or rec.get("market") or "UNKNOWN"
            grouped[str(m)].append(rec)
        return dict(grouped)
    return {}

def validate_history_schema(all_hist: Dict[str, List[Dict[str, Any]]]) -> None:
    if not isinstance(all_hist, dict) or not all_hist:
        raise RuntimeError("Historical data structure unreadable/empty.")
    check_limit = 0
    for mkt, rows in all_hist.items():
        if not isinstance(rows, list): raise TypeError(f"{mkt}: rows must be a list.")
        for r in rows:
            if not isinstance(r, dict): raise TypeError(f"{mkt}: each row must be a dict.")
            if not any(k in r for k in ("OpenPanna", "open_panna", "ClosePanna", "close_panna", "Jodi", "jodi", "Date", "date")):
                raise ValueError(f"{mkt}: row missing all expected keys.")
        check_limit += 1
        if check_limit >= 3: break

# ==========================================
# ========== HUMAN BIAS CORE ===============
# ==========================================
def build_base_freq(idx: "MarketIndex", asof: datetime.date, n_days: int) -> Dict[str, float]:
    last = idx.last_n_on_or_before(asof, n_days)
    cnt = Counter()
    for _, r in last:
        res = extract_day_result(r)
        if res["od"].isdigit(): cnt[res["od"]] += 1
        if res["cd"].isdigit(): cnt[res["cd"]] += 1
    return {str(i): (cnt.get(str(i), 0) + 1.0) * 0.1 for i in range(10)}

def get_day_context(idx: "MarketIndex", asof: datetime.date, offset_days: int = 1) -> Dict[str, str]:
    target_rows = idx.last_n_on_or_before(asof, offset_days)
    if len(target_rows) < offset_days: return {}
    _, r = target_rows[offset_days - 1]
    return extract_day_result(r)

def find_actual_for_date(idx: "MarketIndex", target_date: datetime.date) -> Optional[Dict[str, str]]:
    r = idx.row_for_date(target_date)
    if r is None: return None
    res = extract_day_result(r)
    if not (res["od"] or res["cd"] or res["j"] or res["op"] or res["cp"]): return None
    return res

def detect_14day_fakeout(idx: "MarketIndex", asof: datetime.date) -> Dict[str, Any]:
    """Scans the last 14 days for immediate 2-day loops AND 14-day repeated baiting."""
    last_n = idx.last_n_on_or_before(asof, CONFIG["FAKEOUT_LOOKBACK"])
    if len(last_n) < 2:
        return {"active": False, "jodi_to_break": None}
    
    # Immediate 2-day check (highest priority)
    (_, r0), (_, r1) = last_n[0], last_n[1]
    res0, res1 = extract_day_result(r0), extract_day_result(r1)
    j0, j1 = res0["j"], res1["j"]
    
    identical = (j0 == j1 and j0 != "")
    mirror = bool(j0 and j1 and j0 == mirror_jodi(j1))
    
    jodi_to_break = None
    if identical:
        jodi_to_break = j0
    elif mirror:
        jodi_to_break = mirror_jodi(j1)
    else:
        # 14-day trend check: has a jodi appeared 3+ times in the last 14 days?
        jodi_counts = Counter(extract_day_result(r[1])["j"] for r in last_n if extract_day_result(r[1])["j"])
        for j, count in jodi_counts.most_common(1):
            if count >= 3:
                jodi_to_break = j
                break
                
    return {
        "active": identical or mirror or (jodi_to_break is not None),
        "jodi_to_break": jodi_to_break
    }

def get_herd_digits(idx: "MarketIndex", asof: datetime.date, n_days: int, thr: int) -> Set[str]:
    last = idx.last_n_on_or_before(asof, n_days)
    cnt = Counter()
    for _, r in last:
        res = extract_day_result(r)
        if res["od"].isdigit(): cnt[res["od"]] += 1
        if res["cd"].isdigit(): cnt[res["cd"]] += 1
    return {d for d, c in cnt.items() if c >= thr}

def recent_pannas(idx: "MarketIndex", asof: datetime.date, n_days: int) -> Set[str]:
    last = idx.last_n_on_or_before(asof, n_days)
    out = set()
    for _, r in last:
        res = extract_day_result(r)
        if res["op"]: out.add(res["op"])
        if res["cp"]: out.add(res["cp"])
    return out

def weighted_choice(items: List[Any], weights: List[float], k: int) -> List[Any]:
    items = list(items)
    weights = [max(0.0, float(w)) for w in weights]
    picks = []
    for _ in range(min(k, len(items))):
        s = sum(weights)
        if s <= 0:
            if not items: break
            idx = random.randrange(len(items))
            picks.append(items.pop(idx))
            weights.pop(idx)
            continue
        r, acc = random.random() * s, 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                picks.append(items.pop(i))
                weights.pop(i)
                break
    return picks

# ==========================================
# ========= SHARED SIGNAL COMPUTATION =======
# ==========================================
def compute_signals(idx: "MarketIndex", asof: datetime.date,
                     yctx: Dict[str, str], fakeout: Dict[str, Any]) -> Dict[str, Any]:
    herd = get_herd_digits(idx, asof, CONFIG["HERD_WINDOW_DAYS"], CONFIG["HERD_HOT_THRESHOLD"])
    y_od, y_cd, y_j = yctx.get("od", ""), yctx.get("cd", ""), yctx.get("j", "")

    cut_open = str((int(y_od) + 5) % 10) if y_od.isdigit() else None
    cut_close = str((int(y_cd) + 5) % 10) if y_cd.isdigit() else None
    sum_comp_open = str((10 - int(y_cd)) % 10) if y_cd.isdigit() else None
    sum_comp_close = str((10 - int(y_od)) % 10) if y_od.isdigit() else None

    return {
        "herd": herd,
        "y_od": y_od, "y_cd": y_cd, "y_j": y_j,
        "fakeout": fakeout,
        "cut_open": cut_open, "cut_close": cut_close,
        "sum_comp_open": sum_comp_open, "sum_comp_close": sum_comp_close,
    }

def compose_digits_bias(base_freq: Dict[str, float], role: str, sig: Dict[str, Any]) -> List[str]:
    weights = dict(base_freq)
    target_y = sig["y_od"] if role == "open" else sig["y_cd"]
    other_y = sig["y_cd"] if role == "open" else sig["y_od"]
    cut_digit = sig["cut_open"] if role == "open" else sig["cut_close"]
    comp_digit = sig["sum_comp_open"] if role == "open" else sig["sum_comp_close"]

    for d in sig["herd"]:
        weights[d] *= CONFIG["STREAK_PENALTY"]

    if target_y and target_y.isdigit():
        weights[target_y] *= CONFIG["BREAK_PENALTY"]

    if cut_digit:
        weights[cut_digit] *= CONFIG["CUT_BOOST"]

    if comp_digit:
        weights[comp_digit] *= CONFIG["SUMMOD_BOOST"]

    if other_y and other_y.isdigit():
        for nb in [(int(other_y) + 1) % 10, (int(other_y) - 1) % 10]:
            weights[str(nb)] *= CONFIG["NEIGHBOR_BOOST"]

    y_j = sig["y_j"]
    if y_j and len(y_j) == 2:
        weights[y_j[1] if role == "open" else y_j[0]] *= CONFIG["MIRROR_BOOST"]

    if sig["fakeout"].get("active") and sig["fakeout"].get("jodi_to_break"):
        for ch in sig["fakeout"]["jodi_to_break"]:
            if ch.isdigit():
                weights[ch] *= 0.3

    items = list(weights.keys())
    ws = [weights[k] for k in items]
    want = CONFIG["NUM_OPEN_DIGITS"] if role == "open" else CONFIG["NUM_CLOSE_DIGITS"]
    picks = weighted_choice(items, ws, want)

    out, seen = [], set()
    for p in picks:
        if p not in seen:
            seen.add(p); out.append(p)
        if len(out) >= want: break
    while len(out) < want:
        for k in items:
            if k not in seen:
                seen.add(k); out.append(k); break
        else: break
    return out[:want]

def pannas_pool_for_digits(digs: List[str]) -> List[str]:
    raw: List[str] = []
    for d in digs:
        try: raw += VALID_PANNAS_BY_LAST_DIGIT.get(int(d), [])
        except Exception: continue
    seen, out = set(), []
    for p in raw:
        if p not in seen:
            seen.add(p); out.append(p)
    if CONFIG["EXCLUDE_TRIPLE_PANNAS"]:
        out = [p for p in out if not is_triple_panna(p)]
    return out

def sample_pannas_bias(candidate_pool: List[str], need_k: int, idx: "MarketIndex",
                        asof: datetime.date, role: str, sig: Dict[str, Any], chosen_digits: List[str]) -> List[str]:
    if not candidate_pool: return []
    
    # MATHEMATICAL ENFORCEMENT: Filter candidate pool to ONLY pannas that sum to the chosen digits
    math_valid_pool = []
    for p in candidate_pool:
        p_sum_mod = sum(int(x) for x in p) % 10
        if str(p_sum_mod) in chosen_digits:
            math_valid_pool.append(p)
            
    if not math_valid_pool: return []

    recent = recent_pannas(idx, asof, 5)
    cut_digit = sig["cut_open"] if role == "open" else sig["cut_close"]
    comp_digit = sig["sum_comp_open"] if role == "open" else sig["sum_comp_close"]

    items = list(math_valid_pool)
    ws = []
    for p in items:
        w = 1.0
        if p in recent:
            w *= CONFIG["PANNA_RECENT_PENALTY"]
        mp = mirror_panna(p)
        if mp in recent and mp != p:
            w *= CONFIG["PANNA_MIRROR_BONUS"]
        if cut_digit and cut_digit in p:
            w *= CONFIG["PANNA_CUT_BONUS"]
        if comp_digit and comp_digit in p:
            w *= CONFIG["PANNA_SUMMOD_BONUS"]
        ws.append(max(w, 1e-6))

    picks = weighted_choice(items, ws, need_k)
    out, seen = [], set()
    for p in picks:
        if p not in seen:
            seen.add(p); out.append(p)
        if len(out) >= need_k: break
    while len(out) < need_k:
        progressed = False
        for p in items:
            if p not in seen:
                seen.add(p); out.append(p); progressed = True; break
        if not progressed: break
    return out[:need_k]

def build_jodis_bias(open_d: List[str], close_d: List[str], sig: Dict[str, Any]) -> List[str]:
    cands = []
    y_j = sig["y_j"]
    fakeout = sig["fakeout"]
    
    # Cut-Jodi calculation (±5 shift of yesterday's jodi)
    cut_jodi = ""
    if y_j and len(y_j) == 2:
        cut_jodi = f"{(int(y_j[0]) + 5) % 10}{(int(y_j[1]) + 5) % 10}"

    for a in open_d:
        for b in close_d:
            j = f"{a}{b}"
            w = 1.0
            if j == y_j: w *= 0.2
            if a == b:
                if y_j and len(y_j) == 2 and y_j[0] == y_j[1]:
                    w *= CONFIG["RED_JODI_PENALTY"]
                else:
                    w *= 1.15
            if fakeout.get("active") and j == fakeout.get("jodi_to_break"):
                w *= 0.1
            if y_j and j == mirror_jodi(y_j):
                w *= 1.5
            if (int(a) + int(b)) in (10, 20):
                w *= 1.35
            
            # NEW: Explicit Cut-Jodi Boost
            if cut_jodi and j == cut_jodi:
                w *= CONFIG["CUT_BOOST"]
                
            cands.append((j, w))

    cands.sort(key=lambda x: x[1], reverse=True)
    out, seen = [], set()
    for j, _ in cands:
        if j not in seen:
            seen.add(j); out.append(j)
        if len(out) >= CONFIG["NUM_JODIS"]: break
    return out

# ==========================================
# ============ FEEDBACK / BACKTEST ==========
# ==========================================
def load_state(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    if isinstance(data, dict) and "markets" in data: return data
    return {"markets": {}, "schema_version": 1}

def backtest_update(state: Dict[str, Any], indexes: Dict[str, "MarketIndex"]) -> Dict[str, int]:
    scored_counts: Dict[str, int] = {}
    markets = state.get("markets", {})
    for mkt, entries in markets.items():
        idx = indexes.get(mkt)
        if idx is None: continue
        n_scored = 0
        for entry in entries:
            if entry.get("actual") is not None: continue
            d = parse_date(entry.get("date"))
            if not d: continue
            actual = find_actual_for_date(idx, d)
            if actual is None: continue
            pred = entry.get("predicted", {})
            hits = {
                "open_digit": actual["od"] in pred.get("OpenDigits", []) if actual.get("od") else None,
                "close_digit": actual["cd"] in pred.get("CloseDigits", []) if actual.get("cd") else None,
                "jodi": actual["j"] in pred.get("Jodis", []) if actual.get("j") else None,
                "open_panna": actual["op"] in pred.get("OpenPannas", []) if actual.get("op") else None,
                "close_panna": actual["cp"] in pred.get("ClosePannas", []) if actual.get("cp") else None,
            }
            entry["actual"] = actual
            entry["hits"] = hits
            n_scored += 1
        if n_scored: scored_counts[mkt] = n_scored
    return scored_counts

def rolling_accuracy(entries: List[Dict[str, Any]], lookback_days: int) -> Optional[Dict[str, float]]:
    scored = [e for e in entries if e.get("hits") is not None]
    if len(scored) < CONFIG["MIN_BACKTEST_SAMPLES_FOR_BLEND"]: return None
    scored.sort(key=lambda e: e.get("date", ""), reverse=True)
    window = scored[:lookback_days]
    rates = {}
    for cat in ("open_digit", "close_digit", "jodi", "open_panna", "close_panna"):
        vals = [e["hits"][cat] for e in window if e["hits"].get(cat) is not None]
        rates[cat] = (sum(1 for v in vals if v) / len(vals)) if vals else 0.0
    rates["overall"] = sum(rates.values()) / max(1, len(rates))
    rates["sample_size"] = len(window)
    return rates

def record_prediction(state: Dict[str, Any], market: str, date_str: str, predicted: Dict[str, Any]) -> None:
    markets = state.setdefault("markets", {})
    entries = markets.setdefault(market, [])
    if any(e.get("date") == date_str for e in entries): return
    entries.append({"date": date_str, "predicted": predicted, "actual": None, "hits": None})
    if len(entries) > CONFIG["STATE_HISTORY_CAP_PER_MARKET"]:
        del entries[: len(entries) - CONFIG["STATE_HISTORY_CAP_PER_MARKET"]]

# ==========================================
# ============ PREDICTION LOGIC ============
# ==========================================
def predict_market(mkt: str, idx: "MarketIndex", asof: datetime.date,
                    accuracy: Optional[Dict[str, float]]) -> Dict[str, Any]:
    if len(idx) < 5:
        return {"OpenDigits": [], "CloseDigits": [], "OpenPannas": [], "ClosePannas": [], "Jodis": [],
                "confidence": 0.0, "confidence_basis": "insufficient_data"}

    base_freq = build_base_freq(idx, asof, CONFIG["FREQ_LOOKBACK_DAYS"])
    yctx = get_day_context(idx, asof, offset_days=1)
    fakeout = detect_14day_fakeout(idx, asof)
    sig = compute_signals(idx, asof, yctx, fakeout)

    open_d = compose_digits_bias(base_freq, "open", sig)
    close_d = compose_digits_bias(base_freq, "close", sig)

    open_pool = pannas_pool_for_digits(open_d)
    close_pool = pannas_pool_for_digits(close_d)
    
    # Pass chosen digits to panna sampler for strict mathematical enforcement
    open_p = sample_pannas_bias(open_pool, CONFIG["NUM_OPEN_PANNAS"], idx, asof, "open", sig, open_d)
    close_p = sample_pannas_bias(close_pool, CONFIG["NUM_CLOSE_PANNAS"], idx, asof, "close", sig, close_d)

    jodis = build_jodis_bias(open_d, close_d, sig)

    rule_conf = CONFIG["CONFIDENCE_BASE"]
    if fakeout.get("active"): rule_conf += 0.15
    if yctx.get("j") and mirror_jodi(yctx["j"]) in jodis: rule_conf += 0.15
    if any((int(a) + int(b)) % 10 == 0 for a in open_d for b in close_d if a != b): rule_conf += 0.10
    rule_conf = min(rule_conf, CONFIG["CONFIDENCE_CAP"])

    if accuracy is not None:
        w = CONFIG["RULE_WEIGHT_WHEN_BLENDED"]
        conf = w * rule_conf + (1 - w) * accuracy["overall"]
        basis = f"blended (rules + {accuracy['sample_size']}-day empirical accuracy)"
    else:
        conf = rule_conf
        basis = "rule_triggers_only (not enough scored history yet)"

    conf = max(CONFIG["CONFIDENCE_MIN"], min(conf, CONFIG["CONFIDENCE_CAP"]))

    return {
        "OpenDigits": open_d,
        "CloseDigits": close_d,
        "OpenPannas": open_p,
        "ClosePannas": close_p,
        "Jodis": jodis,
        "confidence": round(conf, 2),
        "confidence_basis": basis,
    }

# ==========================================
# =================  I/O ===================
# ==========================================
def to_app_structure(preds: Dict[str, Dict[str, Any]], run_date: datetime.date,
                      scored_summary: Dict[str, int]) -> Dict[str, Any]:
    markets_array = []
    date_str = run_date.isoformat()
    for market_key, pr in preds.items():
        markets_array.append({
            "market": market_key,
            "predictions": {
                "Date": date_str,
                "Open Digits": pr.get("OpenDigits", []),
                "Close Digits": pr.get("CloseDigits", []),
                "Open Pannas": pr.get("OpenPannas", []),
                "Close Pannas": pr.get("ClosePannas", []),
                "Jodis": pr.get("Jodis", [])
            },
            "confidence": pr.get("confidence", 0.0),
            "confidence_basis": pr.get("confidence_basis", "unknown"),
        })
    return {
        "meta": {
            "generated_at": iso_timestamp(),
            "engine": "v31_robust_kalyan",
            "schema_version": 1,
            "backtest_newly_scored": scored_summary,
        },
        "markets": markets_array
    }

def run_once(paths: Dict[str, Path], for_date: Optional[datetime.date] = None,
             force_prod: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    cfg = CONFIG
    if (not force_prod) and cfg["TEST_MODE"]:
        random.seed(cfg.get("TEST_SEED", 12345))
    else:
        random.seed()

    if for_date is None: for_date = datetime.date.today()

    for key in ("OUTPUT_DIR", "ASSETS_DIR"): safe_mkdir(paths[key])
    safe_mkdir(paths["STATE_FILE"].parent)

    all_hist = load_market_history(paths["HISTORY_FILE"])
    if not all_hist: raise RuntimeError(f"Missing or empty history file: {paths['HISTORY_FILE']}")
    validate_history_schema(all_hist)

    indexes: Dict[str, MarketIndex] = {mkt: MarketIndex(rows) for mkt, rows in all_hist.items()}

    state = load_state(paths["STATE_FILE"])
    scored_summary = backtest_update(state, indexes)
    if scored_summary:
        logger.info(f"Backtest scored new outcomes: {scored_summary}")

    internal_preds = {}
    date_str = for_date.isoformat()
    for mkt, idx in indexes.items():
        try:
            entries = state.get("markets", {}).get(mkt, [])
            accuracy = rolling_accuracy(entries, cfg["ACCURACY_LOOKBACK_DAYS"])
            internal_preds[mkt] = predict_market(mkt, idx, for_date, accuracy)
            record_prediction(state, mkt, date_str, {
                "OpenDigits": internal_preds[mkt]["OpenDigits"],
                "CloseDigits": internal_preds[mkt]["CloseDigits"],
                "OpenPannas": internal_preds[mkt]["OpenPannas"],
                "ClosePannas": internal_preds[mkt]["ClosePannas"],
                "Jodis": internal_preds[mkt]["Jodis"],
            })
        except Exception as e:
            logger.warning(f"{mkt}: prediction error: {e}", exc_info=True)
            internal_preds[mkt] = {"OpenDigits": [], "CloseDigits": [], "OpenPannas": [], "ClosePannas": [],
                                    "Jodis": [], "confidence": 0.0, "confidence_basis": "error"}

    app_json = to_app_structure(internal_preds, for_date, scored_summary)
    tag = today_str(for_date)

    dated_out = paths["OUTPUT_DIR"] / f"predictions_{tag}_v31.json"
    today_out = paths["TODAY_PROD_FILE"] if (force_prod or not cfg["TEST_MODE"]) else paths["TODAY_TEST_FILE"]
    assets_today = paths["TODAYS_PREDICTIONS_JSON"]
    assets_dated = paths["ASSETS_DIR"] / f"predictions_{tag}_v31.json"

    if not dry_run:
        save_json_atomic(dated_out, app_json)
        save_json_atomic(today_out, app_json)
        save_json_atomic(assets_today, app_json)
        save_json_atomic(assets_dated, app_json)
        save_json_atomic(paths["STATE_FILE"], state)

    logger.info(f"OK v31-robust — markets: {len(internal_preds)} | dated: {dated_out} | "
                f"newly_scored: {sum(scored_summary.values()) if scored_summary else 0}")
    return internal_preds

# ==========================================
# ================ SELF TEST ===============
# ==========================================
def selftest() -> bool:
    ok = True
    def check(name: str, cond: bool):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond: ok = False
        print(f"[{status}] {name}")

    check("mirror_jodi('37') == '73'", mirror_jodi("37") == "73")
    check("mirror_panna('123') == '321'", mirror_panna("123") == "321")
    check("is_triple_panna('555') is True", is_triple_panna("555") is True)
    check("is_triple_panna('556') is False", is_triple_panna("556") is False)

    rec = {"Open Digit": "7", "close_panna": "234"}
    check("get_field exact key", get_field(rec, "Open Digit") == "7")
    check("get_field underscore/case-insensitive fallback", get_field(rec, "ClosePanna") == "234")
    check("get_field missing returns ''", get_field(rec, "Nope") == "")

    items = ["0", "1", "2"]
    weights = [0.0, 0.0, 0.0]
    picks = weighted_choice(items, weights, 2)
    check("weighted_choice falls back to uniform on zero weights", len(picks) == 2)

    sig = compute_signals(
        idx=MarketIndex([]),
        asof=datetime.date.today(),
        yctx={"od": "3", "cd": "8", "j": "38"},
        fakeout={"active": False, "jodi_to_break": None},
    )
    check("cut_open = (3+5)%10 = 8", sig["cut_open"] == "8")
    check("sum_comp_open = (10-8)%10 = 2", sig["sum_comp_open"] == "2")

    # Test Panna Math Link
    pool = pannas_pool_for_digits(["5"])
    sampled = sample_pannas_bias(
        candidate_pool=pool, need_k=5, idx=MarketIndex([]), 
        asof=datetime.date.today(), role="open", sig=sig, chosen_digits=["5"]
    )
    check("sample_pannas_bias returns mathematically valid pannas for digit 5",
          all(sum(int(x) for x in p) % 10 == 5 for p in sampled))

    check("normalize_jodi rebuilds '04' from od=0,cd=4 even if raw jodi is broken '4'",
          normalize_jodi("4", "0", "4") == "04")
    check("normalize_panna pads '6' to '006'", normalize_panna("6") == "006")

    rec_bug = {"Date": "16/11/2022", "Open Digit": "0", "Close Digit": "4",
               "Open Panna": "235", "Close Panna": "220", "Jodi": "4"}
    res = extract_day_result(rec_bug)
    check("extract_day_result fixes the real leading-zero jodi bug end-to-end", res["j"] == "04")

    sample_rows = [
        {"Date": "01/01/2026", "Open Digit": "1", "Close Digit": "2", "Open Panna": "118", "Close Panna": "127", "Jodi": "12"},
        {"Date": "02/01/2026", "Open Digit": "3", "Close Digit": "4", "Open Panna": "120", "Close Panna": "130", "Jodi": "34"},
        {"Date": "03/01/2026", "Open Digit": "5", "Close Digit": "6", "Open Panna": "122", "Close Panna": "150", "Jodi": "56"},
    ]
    idx = MarketIndex(sample_rows)
    asof = datetime.date(2026, 1, 3)
    recent2 = idx.last_n_on_or_before(asof, 2)
    check("MarketIndex.last_n_on_or_before returns most-recent-first",
          [d.isoformat() for d, _ in recent2] == ["2026-01-03", "2026-01-02"])

    print("\nSELFTEST:", "ALL PASS" if ok else "FAILURES ABOVE")
    return ok

# ==========================================
# ================== MAIN ==================
# ==========================================
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SattaMatkaAI Prediction Engine v31 (Robust Adversarial Engine).")
    ap.add_argument("--prod", action="store_true", help="Force production output paths.")
    ap.add_argument("--dry-run", action="store_true", help="Compute but do not write any files.")
    ap.add_argument("--selftest", action="store_true", help="Run fast data-free sanity checks and exit.")
    ap.add_argument("--verbose", action="store_true", help="Also log to console.")
    ap.add_argument("--date", type=str, default=None, help="Override run date as YYYY-MM-DD.")

    ap.add_argument("--history", type=str, default=None, help="Path to all_markets_history.json")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--assets-dir", type=str, default=None)
    ap.add_argument("--state-file", type=str, default=None)
    ap.add_argument("--today-test-file", type=str, default=None)
    ap.add_argument("--today-prod-file", type=str, default=None)
    ap.add_argument("--assets-today-file", type=str, default=None)
    ap.add_argument("--log-file", type=str, default=None)
    return ap

def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.selftest:
        return 0 if selftest() else 1

    paths = {
        "HISTORY_FILE": _resolve_path("HISTORY_FILE", args.history),
        "OUTPUT_DIR": _resolve_path("OUTPUT_DIR", args.output_dir),
        "STATE_FILE": _resolve_path("STATE_FILE", args.state_file),
        "TODAY_TEST_FILE": _resolve_path("TODAY_TEST_FILE", args.today_test_file),
        "TODAY_PROD_FILE": _resolve_path("TODAY_PROD_FILE", args.today_prod_file),
        "TODAYS_PREDICTIONS_JSON": _resolve_path("TODAYS_PREDICTIONS_JSON", args.assets_today_file),
        "ASSETS_DIR": _resolve_path("ASSETS_DIR", args.assets_dir),
        "LOG_FILE": _resolve_path("LOG_FILE", args.log_file),
    }

    setup_logging(paths["LOG_FILE"], CONFIG["ENABLE_LOGGING"], args.verbose or CONFIG["LOG_TO_CONSOLE"])

    for_date = None
    if args.date:
        for_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()

    try:
        run_once(paths, for_date=for_date, force_prod=args.prod, dry_run=args.dry_run)
        return 0
    except Exception as e:
        logger.error(f"Engine run failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())