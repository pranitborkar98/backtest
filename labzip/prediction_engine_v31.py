# -*- coding: utf-8 -*-
"""
prediction_engine_v31.py — SattaMatkaAI (Human-Bias & Adversarial Engine — ROBUST)

This is v30 ("v20_optimized") hardened into a self-correcting, portable system.
No v10 (DNA-file) logic is used anywhere in this file — this is a pure evolution
of the v20/v30 adversarial-bias lineage only.

What's new vs v30:
  1. CLOSED FEEDBACK LOOP (real, not cosmetic):
     - STATE_FILE now actually gets used. Every prediction this engine makes is
       persisted with its date. On the *next* run, before generating new numbers,
       the engine looks back at its own past predictions, finds the now-known
       actual result for that date in history, and scores itself (digit/jodi/
       panna hits). Rolling accuracy per market is tracked automatically.
     - No external yesterday_results.json dependency anymore — "yesterday" and
       "actual outcome" are both derived from the same history file, so there is
       one source of truth instead of two files that can fall out of sync.

  2. CONFIDENCE IS NOW REAL:
     - Old confidence score = "+0.15 if a rule fired" (cosmetic, didn't reflect
       whether the engine is actually right). New score blends that rule-trigger
       signal with the market's own empirical hit-rate from the feedback loop
       (once enough history exists), and is reported with a `confidence_basis`
       field so you can tell whether a number was rule-based or evidence-based.

  3. CUT-DIGIT / SUM-MOD10 SIGNALS NOW APPLY EVERYWHERE:
     - In v30 these only nudged digit selection. Panna and jodi selection used a
       narrower signal set, so the "story" the engine told for digits wasn't
       always reflected in the panna/jodi picks. All three now read from one
       shared `compute_signals()` call.

  4. PORTABILITY:
     - Hardcoded C:\\Users\\VCOM\\... paths are now defaults only. Every path can
       be overridden by an environment variable or a CLI flag, and the engine
       will fall back to a local ./satta_data directory if the configured paths
       don't exist, instead of crashing outright.

  5. DEFENSIVE I/O:
     - Atomic writes (write to .tmp then os.replace) so a crash mid-write can't
       corrupt today's predictions file or the state file.
     - get_field() now also does case-insensitive / underscore-insensitive key
       matching so messier scraped history rows don't silently produce empties.

  6. --selftest mode: runs fast, data-free sanity checks on the core scoring
     functions (mirror, weighted_choice, signal computation) so a refactor that
     breaks the math fails immediately instead of silently shipping bad numbers.
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
# Every path below can be overridden three ways, in priority order:
#   1. CLI flag (e.g. --history /path/to/file.json)
#   2. Environment variable (e.g. SATTA_HISTORY_FILE)
#   3. The Windows default baked in here (kept for backwards compatibility)
# If the resolved path doesn't exist AND no override was given, the engine
# falls back to ./satta_data/<name> next to this script, instead of crashing.

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
    """Resolve a configured path: CLI > env var > Windows default > local fallback."""
    if cli_value:
        return Path(cli_value)
    env_val = os.environ.get(_ENV_PREFIX + key)
    if env_val:
        return Path(env_val)
    default = Path(_DEFAULTS[key])
    # If the default's parent exists (we're actually on the VCOM machine), use it.
    if default.parent.exists():
        return default
    # Otherwise fall back to a local, portable location so the engine still runs.
    return _LOCAL_FALLBACK_DIR / default.name


CONFIG = {
    "NUM_OPEN_DIGITS": 3,
    "NUM_CLOSE_DIGITS": 3,
    "NUM_OPEN_PANNAS": 8,
    "NUM_CLOSE_PANNAS": 8,
    "NUM_JODIS": 9,

    "EXCLUDE_TRIPLE_PANNAS": True,

    "FREQ_LOOKBACK_DAYS": 45,
    "HERD_WINDOW_DAYS": 5,
    "HERD_HOT_THRESHOLD": 3,
    "FAKEOUT_LOOKBACK": 2,

    "BREAK_PENALTY": 0.15,
    "STREAK_PENALTY": 0.10,
    "SUMMOD_BOOST": 1.45,
    "NEIGHBOR_BOOST": 1.25,
    "MIRROR_BOOST": 1.35,
    "CUT_BOOST": 1.40,
    "RED_JODI_PENALTY": 0.25,

    "PANNA_RECENT_PENALTY": 0.35,
    "PANNA_MIRROR_BONUS": 1.30,
    "PANNA_CUT_BONUS": 1.25,     # NEW: pannas containing the cut-digit get a lift
    "PANNA_SUMMOD_BONUS": 1.20,  # NEW: pannas containing the sum-complement digit get a lift

    "CONFIDENCE_BASE": 0.35,
    "CONFIDENCE_CAP": 0.80,
    "CONFIDENCE_MIN": 0.05,
    "MIN_BACKTEST_SAMPLES_FOR_BLEND": 5,   # need this many scored days before trusting empirical accuracy
    "ACCURACY_LOOKBACK_DAYS": 30,          # rolling window for empirical hit-rate
    "RULE_WEIGHT_WHEN_BLENDED": 0.45,      # how much of confidence still comes from rule-triggers once we trust accuracy
    "STATE_HISTORY_CAP_PER_MARKET": 90,    # bound state file growth

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
            logger.warning(f"Logging file handler failed, continuing without it: {e}")
    if enable_console or not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)


# ==========================================
# ================ UTILITIES ===============
# ==========================================
def today_str(dt: Optional[datetime.date] = None) -> str:
    if dt is None:
        dt = datetime.date.today()
    return dt.strftime("%Y%m%d")


def iso_timestamp() -> str:
    return datetime.datetime.now().isoformat()


def safe_mkdir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(p: Path) -> Any:
    p = Path(p)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to parse JSON at {p}: {e}")
        return None


def save_json_atomic(p: Path, obj: Any) -> None:
    """Write via a temp file + atomic rename so a crash mid-write can't corrupt the target."""
    p = Path(p)
    safe_mkdir(p.parent)
    fd, tmp_path = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, p)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def parse_date(d: Any) -> Optional[datetime.date]:
    if not d:
        return None
    s = str(d)
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


def is_triple_panna(p: str) -> bool:
    return isinstance(p, str) and len(p) == 3 and p.isdigit() and len(set(p)) == 1


def mirror_jodi(j: str) -> str:
    return f"{j[1]}{j[0]}" if isinstance(j, str) and len(j) == 2 else j


def mirror_panna(p: str) -> str:
    return p[::-1] if isinstance(p, str) and len(p) == 3 else p


def get_field(rec: Dict[str, Any], *keys: str) -> str:
    """Look up a value by any of several key spellings, including case/underscore
    variants, so messier scraped rows don't silently produce empty strings."""
    if not isinstance(rec, dict):
        return ""
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return str(rec[k]).strip()
    # Fallback: case/underscore-insensitive scan
    norm_keys = {k.lower().replace(" ", "").replace("_", ""): k for k in keys}
    for rk, rv in rec.items():
        nk = str(rk).lower().replace(" ", "").replace("_", "")
        if nk in norm_keys and rv not in (None, ""):
            return str(rv).strip()
    return ""


def normalize_jodi(raw_jodi: str, od: str, cd: str) -> str:
    """Real scraped history files routinely drop the leading zero of a Jodi
    when Open Digit is 0 (e.g. Open=0, Close=4 gets stored as "4" instead of
    "04"). Trusting the raw field in that case silently breaks every mirror,
    trap, and backtest-hit check downstream. Reconstructing from od+cd
    whenever both are valid single digits is far more reliable than trusting
    the raw string, so it always wins when available."""
    if od.isdigit() and cd.isdigit() and len(od) == 1 and len(cd) == 1:
        return f"{od}{cd}"
    if raw_jodi.isdigit() and len(raw_jodi) <= 2:
        return raw_jodi.zfill(2)
    return ""


def normalize_panna(raw_panna: str) -> str:
    """Same leading-zero issue shows up on pannas (e.g. "6" instead of "006").
    Zero-pad any all-digit value up to 3 chars; reject anything longer/invalid."""
    if raw_panna.isdigit() and 1 <= len(raw_panna) <= 3:
        return raw_panna.zfill(3)
    return ""


def extract_day_result(rec: Dict[str, Any]) -> Dict[str, str]:
    """Single source of truth for pulling a normalized {od, cd, j, op, cp} out
    of one history row. Used by both the live yesterday-context lookup and the
    backtest's actual-result lookup, so a data-quality fix here can't drift
    out of sync between the two call sites the way v31 risked."""
    y_od = get_field(rec, "Open Digit", "OpenDigit", "open_digit")
    y_cd = get_field(rec, "Close Digit", "CloseDigit", "close_digit")
    y_j_raw = get_field(rec, "Jodi", "jodi")
    y_op = normalize_panna(get_field(rec, "Open Panna", "OpenPanna", "open_panna"))
    y_cp = normalize_panna(get_field(rec, "Close Panna", "ClosePanna", "close_panna"))
    y_j = normalize_jodi(y_j_raw, y_od, y_cd)
    return {"od": y_od, "cd": y_cd, "j": y_j, "op": y_op, "cp": y_cp}


class MarketIndex:
    """Precomputed, sorted view of one market's history so repeated lookups
    (base frequency, herd digits, recent pannas, yesterday context, backtest
    lookups) don't each re-filter and re-sort the full row list from scratch.
    Built once per market per run instead of ~6 times."""

    __slots__ = ("dates_asc", "by_date")

    def __init__(self, rows: List[Dict[str, Any]]):
        by_date: Dict[datetime.date, Dict[str, Any]] = {}
        for r in rows:
            d = parse_date(r.get("Date") or r.get("date"))
            if d:
                by_date[d] = r  # last occurrence wins if a date is duplicated
        self.by_date = by_date
        self.dates_asc = sorted(by_date.keys())

    def last_n_on_or_before(self, asof: datetime.date, n: int) -> List[Tuple[datetime.date, Dict[str, Any]]]:
        """n most recent dated rows with date <= asof, most-recent first.
        O(log n + k) via bisect instead of O(n log n) full re-sort per call."""
        pos = bisect.bisect_right(self.dates_asc, asof)
        if pos == 0:
            return []
        start = max(0, pos - n)
        window = self.dates_asc[start:pos]
        return [(d, self.by_date[d]) for d in reversed(window)]

    def row_for_date(self, d: datetime.date) -> Optional[Dict[str, Any]]:
        return self.by_date.get(d)

    def __len__(self) -> int:
        return len(self.by_date)


def last_n_days(rows: List[Dict[str, Any]], asof: datetime.date, n: int) -> List[Tuple[datetime.date, Dict[str, Any]]]:
    """Kept for backwards compatibility / ad-hoc use; prefer MarketIndex for
    anything called more than once per market in a run."""
    dated = []
    for r in rows:
        d = parse_date(r.get("Date") or r.get("date"))
        if d and d <= asof:
            dated.append((d, r))
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
        if not isinstance(rows, list):
            raise TypeError(f"{mkt}: rows must be a list.")
        for r in rows:
            if not isinstance(r, dict):
                raise TypeError(f"{mkt}: each row must be a dict.")
            if not any(k in r for k in ("OpenPanna", "open_panna", "ClosePanna", "close_panna", "Jodi", "jodi", "Date", "date")):
                raise ValueError(f"{mkt}: row missing all expected keys.")
        check_limit += 1
        if check_limit >= 3:
            break


# ==========================================
# ========== HUMAN BIAS CORE ===============
# ==========================================
def build_base_freq(idx: "MarketIndex", asof: datetime.date, n_days: int) -> Dict[str, float]:
    """Dampened base frequency mapping so it doesn't dominate over the bias rules."""
    last = idx.last_n_on_or_before(asof, n_days)
    cnt = Counter()
    for _, r in last:
        res = extract_day_result(r)
        if res["od"].isdigit():
            cnt[res["od"]] += 1
        if res["cd"].isdigit():
            cnt[res["cd"]] += 1
    return {str(i): (cnt.get(str(i), 0) + 1.0) * 0.1 for i in range(10)}


def get_day_context(idx: "MarketIndex", asof: datetime.date, offset_days: int = 1) -> Dict[str, str]:
    """Get the actual result context for the day `offset_days` before `asof`.
    offset_days=1 -> 'yesterday' relative to asof. Used for live prediction bias
    signals (yesterday's result feeds today's trap/mirror/cut logic)."""
    target_rows = idx.last_n_on_or_before(asof, offset_days)
    if len(target_rows) < offset_days:
        return {}
    _, r = target_rows[offset_days - 1]
    return extract_day_result(r)


def find_actual_for_date(idx: "MarketIndex", target_date: datetime.date) -> Optional[Dict[str, str]]:
    """Find the recorded actual result for an exact date (used by the backtest loop)."""
    r = idx.row_for_date(target_date)
    if r is None:
        return None
    res = extract_day_result(r)
    if not (res["od"] or res["cd"] or res["j"] or res["op"] or res["cp"]):
        return None
    return res


def detect_2day_fakeout(idx: "MarketIndex", asof: datetime.date) -> Dict[str, Any]:
    last2 = idx.last_n_on_or_before(asof, 2)
    if len(last2) < 2:
        return {"active": False, "jodi_to_break": None}
    (_, r0), (_, r1) = last2[0], last2[1]
    res0, res1 = extract_day_result(r0), extract_day_result(r1)
    j0, j1 = res0["j"], res1["j"]
    op0, op1 = res0["op"], res1["op"]
    identical = (j0 == j1 and op0 == op1 and j0 != "")
    mirror = bool(j0 and j1 and j0 == mirror_jodi(j1))
    return {
        "active": identical or mirror,
        "jodi_to_break": j0 if identical else (mirror_jodi(j1) if mirror else None)
    }


def get_herd_digits(idx: "MarketIndex", asof: datetime.date, n_days: int, thr: int) -> Set[str]:
    last = idx.last_n_on_or_before(asof, n_days)
    cnt = Counter()
    for _, r in last:
        res = extract_day_result(r)
        if res["od"].isdigit():
            cnt[res["od"]] += 1
        if res["cd"].isdigit():
            cnt[res["cd"]] += 1
    return {d for d, c in cnt.items() if c >= thr}


def recent_pannas(idx: "MarketIndex", asof: datetime.date, n_days: int) -> Set[str]:
    last = idx.last_n_on_or_before(asof, n_days)
    out = set()
    for _, r in last:
        res = extract_day_result(r)
        if res["op"]:
            out.add(res["op"])
        if res["cp"]:
            out.add(res["cp"])
    return out


def weighted_choice(items: List[Any], weights: List[float], k: int) -> List[Any]:
    """Pick k items without replacement, proportional to weight. Falls back to
    uniform choice if weights collapse to zero (defensive against bad inputs)."""
    items = list(items)
    weights = [max(0.0, float(w)) for w in weights]
    picks = []
    for _ in range(min(k, len(items))):
        s = sum(weights)
        if s <= 0:
            # Defensive fallback: uniform pick instead of silently stopping.
            if not items:
                break
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
    """Single source of truth for the adversarial signals so digits, pannas, and
    jodis are all telling the same story instead of drifting apart (v30 only
    applied cut/sum10 to digits; this applies them everywhere)."""
    herd = get_herd_digits(idx, asof, CONFIG["HERD_WINDOW_DAYS"], CONFIG["HERD_HOT_THRESHOLD"])
    y_od, y_cd, y_j = yctx.get("od", ""), yctx.get("cd", ""), yctx.get("j", "")

    cut_open = str((int(y_od) + 5) % 10) if y_od.isdigit() else None
    cut_close = str((int(y_cd) + 5) % 10) if y_cd.isdigit() else None
    sum_comp_open = str((10 - int(y_cd)) % 10) if y_cd.isdigit() else None   # complement targets the *open* slot
    sum_comp_close = str((10 - int(y_od)) % 10) if y_od.isdigit() else None  # complement targets the *close* slot

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
                weights[ch] *= 0.3  # softened vs v30's 0.05 — that was aggressive enough to wipe a digit entirely

    items = list(weights.keys())
    ws = [weights[k] for k in items]
    want = CONFIG["NUM_OPEN_DIGITS"] if role == "open" else CONFIG["NUM_CLOSE_DIGITS"]
    picks = weighted_choice(items, ws, want)

    out, seen = [], set()
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= want:
            break
    while len(out) < want:
        for k in items:
            if k not in seen:
                seen.add(k)
                out.append(k)
                break
        else:
            break
    return out[:want]


def pannas_pool_for_digits(digs: List[str]) -> List[str]:
    raw: List[str] = []
    for d in digs:
        try:
            raw += VALID_PANNAS_BY_LAST_DIGIT.get(int(d), [])
        except Exception:
            continue
    seen, out = set(), []
    for p in raw:
        if p not in seen:
            seen.add(p)
            out.append(p)
    if CONFIG["EXCLUDE_TRIPLE_PANNAS"]:
        out = [p for p in out if not is_triple_panna(p)]
    return out


def sample_pannas_bias(candidate_pool: List[str], need_k: int, idx: "MarketIndex",
                        asof: datetime.date, role: str, sig: Dict[str, Any]) -> List[str]:
    if not candidate_pool:
        return []
    recent = recent_pannas(idx, asof, 5)
    cut_digit = sig["cut_open"] if role == "open" else sig["cut_close"]
    comp_digit = sig["sum_comp_open"] if role == "open" else sig["sum_comp_close"]

    items = list(candidate_pool)
    ws = []
    for p in items:
        w = 1.0
        if p in recent:
            w *= CONFIG["PANNA_RECENT_PENALTY"]
        mp = mirror_panna(p)
        if mp in recent and mp != p:
            w *= CONFIG["PANNA_MIRROR_BONUS"]
        # NEW: keep panna picks consistent with the same cut/sum-mod10 story used for digits
        if cut_digit and cut_digit in p:
            w *= CONFIG["PANNA_CUT_BONUS"]
        if comp_digit and comp_digit in p:
            w *= CONFIG["PANNA_SUMMOD_BONUS"]
        ws.append(max(w, 1e-6))

    picks = weighted_choice(items, ws, need_k)
    out, seen = [], set()
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= need_k:
            break
    while len(out) < need_k:
        progressed = False
        for p in items:
            if p not in seen:
                seen.add(p)
                out.append(p)
                progressed = True
                break
        if not progressed:
            break
    return out[:need_k]


def build_jodis_bias(open_d: List[str], close_d: List[str], sig: Dict[str, Any]) -> List[str]:
    cands = []
    y_j = sig["y_j"]
    fakeout = sig["fakeout"]
    for a in open_d:
        for b in close_d:
            j = f"{a}{b}"
            w = 1.0
            if j == y_j:
                w *= 0.2
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
            cands.append((j, w))

    cands.sort(key=lambda x: x[1], reverse=True)
    out, seen = [], set()
    for j, _ in cands:
        if j not in seen:
            seen.add(j)
            out.append(j)
        if len(out) >= CONFIG["NUM_JODIS"]:
            break
    return out


# ==========================================
# ============ FEEDBACK / BACKTEST ==========
# ==========================================
def load_state(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    if isinstance(data, dict) and "markets" in data:
        return data
    return {"markets": {}, "schema_version": 1}


def backtest_update(state: Dict[str, Any], indexes: Dict[str, "MarketIndex"]) -> Dict[str, int]:
    """For every stored prediction that doesn't have a scored outcome yet, check
    if the actual result has shown up in history now, and score it. Returns a
    summary of how many entries got newly scored, per market."""
    scored_counts: Dict[str, int] = {}
    markets = state.get("markets", {})
    for mkt, entries in markets.items():
        idx = indexes.get(mkt)
        if idx is None:
            continue
        n_scored = 0
        for entry in entries:
            if entry.get("actual") is not None:
                continue
            d = parse_date(entry.get("date"))
            if not d:
                continue
            actual = find_actual_for_date(idx, d)
            if actual is None:
                continue
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
        if n_scored:
            scored_counts[mkt] = n_scored
    return scored_counts


def rolling_accuracy(entries: List[Dict[str, Any]], lookback_days: int) -> Optional[Dict[str, float]]:
    """Hit-rate per category over the most recent N scored entries. Returns None
    if there isn't enough scored history yet to be meaningful."""
    scored = [e for e in entries if e.get("hits") is not None]
    if len(scored) < CONFIG["MIN_BACKTEST_SAMPLES_FOR_BLEND"]:
        return None
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
    # Don't double-record if we already have an entry for this date (idempotent re-runs)
    if any(e.get("date") == date_str for e in entries):
        return
    entries.append({"date": date_str, "predicted": predicted, "actual": None, "hits": None})
    # Bound growth
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
    fakeout = detect_2day_fakeout(idx, asof)
    sig = compute_signals(idx, asof, yctx, fakeout)

    open_d = compose_digits_bias(base_freq, "open", sig)
    close_d = compose_digits_bias(base_freq, "close", sig)

    open_pool = pannas_pool_for_digits(open_d)
    close_pool = pannas_pool_for_digits(close_d)
    open_p = sample_pannas_bias(open_pool, CONFIG["NUM_OPEN_PANNAS"], idx, asof, "open", sig)
    close_p = sample_pannas_bias(close_pool, CONFIG["NUM_CLOSE_PANNAS"], idx, asof, "close", sig)

    jodis = build_jodis_bias(open_d, close_d, sig)

    # --- Rule-trigger component (same spirit as v30, kept as one input) ---
    rule_conf = CONFIG["CONFIDENCE_BASE"]
    if fakeout.get("active"):
        rule_conf += 0.15
    if yctx.get("j") and mirror_jodi(yctx["j"]) in jodis:
        rule_conf += 0.15
    if any((int(a) + int(b)) % 10 == 0 for a in open_d for b in close_d if a != b):
        rule_conf += 0.10
    rule_conf = min(rule_conf, CONFIG["CONFIDENCE_CAP"])

    # --- Blend with empirical accuracy if we have enough backtest history ---
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
            "engine": "v31_robust",
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

    if for_date is None:
        for_date = datetime.date.today()

    for key in ("OUTPUT_DIR", "ASSETS_DIR"):
        safe_mkdir(paths[key])
    safe_mkdir(paths["STATE_FILE"].parent)

    all_hist = load_market_history(paths["HISTORY_FILE"])
    if not all_hist:
        raise RuntimeError(f"Missing or empty history file: {paths['HISTORY_FILE']}")
    validate_history_schema(all_hist)

    # Build each market's index exactly once per run; everything below reuses it
    # instead of re-filtering/re-sorting the raw row list repeatedly.
    indexes: Dict[str, MarketIndex] = {mkt: MarketIndex(rows) for mkt, rows in all_hist.items()}

    # --- 1. Close the loop: score yesterday's (and any older unscored) predictions ---
    state = load_state(paths["STATE_FILE"])
    scored_summary = backtest_update(state, indexes)
    if scored_summary:
        logger.info(f"Backtest scored new outcomes: {scored_summary}")

    # --- 2. Predict today, using each market's own rolling accuracy if available ---
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
    """Fast, data-free sanity checks on the core math. Run with --selftest."""
    ok = True

    def check(name: str, cond: bool):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
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
    check("weighted_choice falls back to uniform on zero weights (returns k items)", len(picks) == 2)

    sig = compute_signals(
        idx=MarketIndex([]),
        asof=datetime.date.today(),
        yctx={"od": "3", "cd": "8", "j": "38"},
        fakeout={"active": False, "jodi_to_break": None},
    )
    check("cut_open = (3+5)%10 = 8", sig["cut_open"] == "8")
    check("sum_comp_open = (10-8)%10 = 2", sig["sum_comp_open"] == "2")

    pool = pannas_pool_for_digits(["5"])
    check("panna pool for digit 5 non-empty and triples excluded",
          len(pool) > 0 and all(not is_triple_panna(p) for p in pool))

    entries = [{"date": f"2026-01-{i:02d}", "hits": {"open_digit": True, "close_digit": False,
                                                       "jodi": False, "open_panna": False, "close_panna": False}}
               for i in range(1, 8)]
    acc = rolling_accuracy(entries, 30)
    check("rolling_accuracy returns a dict once min sample size met", acc is not None and "overall" in acc)

    # --- Leading-zero data-quality bug coverage (found in real scraped history) ---
    check("normalize_jodi rebuilds '04' from od=0,cd=4 even if raw jodi is the broken '4'",
          normalize_jodi("4", "0", "4") == "04")
    check("normalize_jodi pads a bare single-digit jodi when od/cd unavailable",
          normalize_jodi("4", "", "") == "04")
    check("normalize_panna pads '6' to '006'", normalize_panna("6") == "006")
    check("normalize_panna pads '0' to '000'", normalize_panna("0") == "000")
    check("normalize_panna leaves a valid 3-digit panna untouched", normalize_panna("789") == "789")
    check("normalize_panna rejects garbage", normalize_panna("abc") == "")

    rec_bug = {"Date": "16/11/2022", "Open Digit": "0", "Close Digit": "4",
               "Open Panna": "235", "Close Panna": "220", "Jodi": "4"}
    res = extract_day_result(rec_bug)
    check("extract_day_result fixes the real leading-zero jodi bug end-to-end", res["j"] == "04")

    # --- MarketIndex correctness ---
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
    check("MarketIndex.row_for_date finds an exact date",
          idx.row_for_date(datetime.date(2026, 1, 1)) is not None)
    check("MarketIndex.row_for_date returns None for a missing date",
          idx.row_for_date(datetime.date(2026, 1, 9)) is None)

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
    ap.add_argument("--date", type=str, default=None, help="Override run date as YYYY-MM-DD (for backfilling/testing).")

    ap.add_argument("--history", type=str, default=None, help="Path to all_markets_history.json")
    ap.add_argument("--output-dir", type=str, default=None, help="Directory for dated prediction outputs.")
    ap.add_argument("--assets-dir", type=str, default=None, help="Directory mirrored to the Android app assets.")
    ap.add_argument("--state-file", type=str, default=None, help="Path to the engine's feedback-loop state file.")
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

# ==== BACKTEST LAB 1-LINER: unique dated output -> D:\backtest\predictions_<YYYY-MM-DD>_v31.json ====
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
        _p = _bt_dir / f"predictions_{ds}_v31.json"
        _t = _p.with_suffix(".json.tmp")
        _t.write_text(_bt_json.dumps(out_obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        _bt_os.replace(_t, _p)
        print(f"[BACKTEST] wrote {_p}")
    _bt_obj = None
    for _bt_k in ('app_json', 'output', 'result', 'payload', 'data'):
        _bt_v = globals().get(_bt_k)
        if isinstance(_bt_v, (dict, list)) and _bt_v:
            _bt_obj = _bt_v; break
    if _bt_obj is None and 'run_once' in dir():
        try:
            _bt_r = run_once()
            if isinstance(_bt_r, (dict, list)) and _bt_r:
                _bt_obj = _bt_r
        except Exception as _bt_r_e:
            print(f"[BACKTEST] skipped: engine run failed -> {_bt_r_e}")
    if _bt_obj is not None:
        _bt_write(_bt_obj)
except Exception as _bt_e:
    print(f"[BACKTEST] skipped: {_bt_e}")
# ==== END BACKTEST LAB 1-LINER ====
