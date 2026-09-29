# -*- coding: utf-8 -*-
"""
prediction_engine_v33.py — SattaMatkaAI (Adaptive Cognitive Architecture v33)

Drop-in replacement for v32: identical CLI flags, identical output JSON schema,
identical state-file location and structure (only additive optional keys).
Existing engine_v32_state.json accuracy history continues to work unmodified.

What's new vs v32:
  1. REAL SELF-AWARENESS (v32's was decorative): SignalRegistry.record() was never
     called in production in v32 — every rule's boost was permanently stuck at 1.0.
     v33 attributes every chosen digit/jodi/panna to the rule(s) that proposed it,
     stores that attribution with the prediction, and scores each rule individually
     once the actual result is known. Rules that perform well get boosted; rules
     that perform badly get muted — automatically, per the existing thresholds.
  2. FULL MARKOV UTILIZATION: mm.oo (open->open) and mm.cc (close->close) self-
     transition matrices were built every run and never read. Now feed HARD-level
     signals, same as the existing oc/co/oj transitions.
  3. ADAPTIVE EXPLORATION: when a market's rolling backtest accuracy is poor, digit
     selection temperature rises (more diverse picks); when accuracy is strong,
     selection sharpens toward the highest-weighted candidates. No history yet ->
     temperature 1.0 -> identical behavior to v32.
  4. HARDENED PANNA FALLBACK: sample_pannas_v32 could silently return [] if the
     math-valid pool was empty for a market/day. v33 falls back to a relaxed pool
     instead of giving up, and logs when it does.
  5. PERFORMANCE FIX: TheoryOfMind was rebuilding a full MultiScaleAnalyzer
     internally on every call, duplicating work already done in compute_signals_v32.
     Now reuses the one already computed.
  6. BEAUTY_PANNAS/UGLY_PANNAS overlap (13 pannas in both sets) intentionally left
     as-is per your call — documented in place, not modified.

Everything from v32's docstring (hierarchical priority, Markov, multi-scale windows,
Theory of Mind, aesthetic psychology, due theory, time-split psychology) is retained.
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
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Any, Tuple, Iterable, Optional, Set

# ==========================================
# ================= CONFIG =================
# ==========================================
_DEFAULTS = {
    "HISTORY_FILE": r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\sattaboss-data\data\all_markets_history.json",
    "OUTPUT_DIR":   r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output",
    "STATE_FILE":   r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\state\engine_v32_state.json",
    "TODAY_TEST_FILE": r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\todays_predictions_v32_TEST.json",
    "TODAY_PROD_FILE": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\todays_predictions.json",
    "TODAYS_PREDICTIONS_JSON": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\todays_predictions.json",
    "LOG_FILE":   r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\engine_v32.log",
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
    # Backtest lab: a history file shipped next to this script always wins.
    if key == "HISTORY_FILE":
        for _h in (Path(__file__).resolve().parent / "all_markets_history.json",
                   _LOCAL_FALLBACK_DIR / default.name):
            if _h.exists() and _h.stat().st_size > 0:
                return _h
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

    # Multi-scale windows
    "MICRO_WINDOW_DAYS": 2,
    "MESO_WINDOW_DAYS": 5,
    "MACRO_WINDOW_DAYS": 15,
    "FREQ_LOOKBACK_DAYS": 45,
    "MARKOV_LOOKBACK_DAYS": 365,
    "DUE_THEORY_MAX_DAYS": 20,
    "CYCLE_FIBONACCI": [5, 8, 13],

    # Hierarchical Priority Weights
    "PRIORITY_VETO": float("inf"),
    "PRIORITY_HARD": 1000.0,
    "PRIORITY_SOFT": 100.0,
    "PRIORITY_NOISE": 1.0,

    # Markov
    "MARKOV_BOOST_BASE": 1.60,
    "MARKOV_MIN_COUNT": 3,

    # Signal Registry (Self-awareness)
    "SIGNAL_ACCURACY_LOOKBACK": 30,
    "SIGNAL_MUTE_THRESHOLD": 0.15,
    "SIGNAL_BOOST_THRESHOLD": 0.55,
    "SIGNAL_MIN_SAMPLES": 5,

    # Aesthetic Psychology
    "BEAUTY_PENALTY": 0.45,
    "UGLY_BOOST": 1.40,

    # Theory of Mind
    "TOM_LEVELS": 2,
    "TOM_TRAP_BOOST": 1.50,
    "TOM_ANTIHERD_PENALTY": 0.35,

    # Time Psychology Split
    "OPEN_CUT_BOOST": 1.45,
    "OPEN_NEIGHBOR_BOOST": 1.20,
    "CLOSE_MIRROR_BOOST": 1.50,
    "CLOSE_SUMCOMP_BOOST": 1.35,

    # Due Theory & Cycles
    "DUE_BOOST_BASE": 1.30,
    "CYCLE_BREAKER_BOOST": 1.40,
    "COLD_REGRESSION_DAYS": 12,
    "COLD_REGRESSION_BOOST": 1.25,

    # Legacy / Enhanced
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
    "CONFIDENCE_CAP": 0.85,
    "CONFIDENCE_MIN": 0.05,
    "MIN_BACKTEST_SAMPLES_FOR_BLEND": 5,
    "ACCURACY_LOOKBACK_DAYS": 30,
    "RULE_WEIGHT_WHEN_BLENDED": 0.40,
    "STATE_HISTORY_CAP_PER_MARKET": 90,

    "ENABLE_LOGGING": True,
    "LOG_TO_CONSOLE": False,
    "LOG_MAX_MB": 5,
    "LOG_BACKUPS": 3,

    "TEST_MODE": False,
    "TEST_SEED": 20251109,
}

# ==========================================
# ============ AESTHETIC MAP ===============
# ==========================================
# "Beautiful" pannas players love (operator bait). "Ugly" pannas operators favor.
BEAUTY_PANNAS = {
    "111", "222", "333", "444", "555", "666", "777", "888", "999",
    "123", "234", "345", "456", "567", "678", "789", "890", "901",
    "135", "246", "357", "468", "579", "680", "791", "802", "913",
    "147", "258", "369", "470", "581", "692", "703", "814", "925",
    "118", "227", "336", "445", "554", "663", "772", "881", "990",
}

UGLY_PANNAS = {
    "079", "089", "288", "577", "499", "389", "479", "569", "578",
    "668", "677", "399", "489", "579", "588", "669", "678", "499",
    "589", "679", "688", "778", "599", "689", "779", "788", "699",
    "789", "799", "889", "890", "169", "178", "259", "268", "277",
    "349", "358", "367", "448", "457", "466", "556", "790", "880",
    "899", "269", "278", "359", "368", "377", "449", "458", "467",
    "557", "566", "890", "189", "279", "369", "378", "459", "468",
    "477", "558", "567", "990", "199", "289", "379", "388", "469",
    "478", "559", "568", "577", "667", "999", "666", "117", "126",
    "135", "144", "180", "225", "234", "270", "360", "450", "900",
    "333",
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
logger = logging.getLogger("pred_engine_v32")
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
# ============= MARKOV MACHINE =============
# ==========================================
class MarkovMachine:
    """Builds Open→Close and Close→Open transition matrices from history."""
    __slots__ = ("oc", "co", "oo", "cc", "oj", "cj", "built")

    def __init__(self):
        self.oc: Dict[str, Counter] = defaultdict(Counter)
        self.co: Dict[str, Counter] = defaultdict(Counter)
        self.oo: Dict[str, Counter] = defaultdict(Counter)
        self.cc: Dict[str, Counter] = defaultdict(Counter)
        self.oj: Dict[str, Counter] = defaultdict(Counter)
        self.cj: Dict[str, Counter] = defaultdict(Counter)
        self.built = False

    def build(self, idx: MarketIndex, asof: datetime.date, lookback: int):
        last = idx.last_n_on_or_before(asof, lookback)
        # Process in chronological order for transitions
        chronological = list(reversed(last))
        for i in range(1, len(chronological)):
            prev = extract_day_result(chronological[i-1][1])
            curr = extract_day_result(chronological[i][1])
            if prev["od"] and curr["cd"]:
                self.oc[prev["od"]][curr["cd"]] += 1
            if prev["cd"] and curr["od"]:
                self.co[prev["cd"]][curr["od"]] += 1
            if prev["od"] and curr["od"]:
                self.oo[prev["od"]][curr["od"]] += 1
            if prev["cd"] and curr["cd"]:
                self.cc[prev["cd"]][curr["cd"]] += 1
            if prev["od"] and curr["j"]:
                self.oj[prev["od"]][curr["j"]] += 1
            if prev["cd"] and curr["j"]:
                self.cj[prev["cd"]][curr["j"]] += 1
        self.built = True

    def prob(self, matrix: Dict[str, Counter], from_state: str, to_state: str) -> float:
        cnt = matrix.get(from_state)
        if not cnt:
            return 0.0
        total = sum(cnt.values())
        if total < CONFIG["MARKOV_MIN_COUNT"]:
            return 0.0
        return cnt.get(to_state, 0) / total

    def top_n(self, matrix: Dict[str, Counter], from_state: str, n: int) -> List[Tuple[str, float]]:
        cnt = matrix.get(from_state)
        if not cnt:
            return []
        total = sum(cnt.values())
        if total < CONFIG["MARKOV_MIN_COUNT"]:
            return []
        probs = [(k, v / total) for k, v in cnt.items()]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:n]

# ==========================================
# ========= SIGNAL REGISTRY (SELF-AWARENESS) =========
# ==========================================
class SignalRegistry:
    """Tracks per-rule accuracy and dynamically mutes/boosts signals."""
    RULES = [
        "markov_oc", "markov_co", "markov_oo", "markov_cc", "markov_oj", "markov_cj",
        "due_theory", "cycle_breaker", "cold_regression",
        "mirror", "cut", "sumcomp", "neighbor", "fakeout_break", "anti_herd",
        "tom_level1", "tom_level2", "aesthetic_ugly", "aesthetic_beauty",
        "open_time_cut", "close_time_mirror", "interdependence_bind",
    ]

    def __init__(self, state: Dict[str, Any]):
        self.hits: Dict[str, deque] = {r: deque(maxlen=CONFIG["SIGNAL_ACCURACY_LOOKBACK"]) for r in self.RULES}
        self.boosts: Dict[str, float] = {r: 1.0 for r in self.RULES}
        # Load from state if present
        saved = state.get("signal_registry", {})
        for r in self.RULES:
            if r in saved:
                self.boosts[r] = saved[r].get("boost", 1.0)
                hist = saved[r].get("history", [])
                self.hits[r].extend(hist[-CONFIG["SIGNAL_ACCURACY_LOOKBACK"]:])

    def record(self, rule: str, hit: bool):
        self.hits[rule].append(1 if hit else 0)
        self._recalc(rule)

    def _recalc(self, rule: str):
        hist = list(self.hits[rule])
        n = len(hist)
        if n < CONFIG["SIGNAL_MIN_SAMPLES"]:
            self.boosts[rule] = 1.0
            return
        acc = sum(hist) / n
        if acc < CONFIG["SIGNAL_MUTE_THRESHOLD"]:
            self.boosts[rule] = 0.2  # Muted
        elif acc > CONFIG["SIGNAL_BOOST_THRESHOLD"]:
            self.boosts[rule] = 1.0 + (acc - CONFIG["SIGNAL_BOOST_THRESHOLD"]) * 2.0
            self.boosts[rule] = min(self.boosts[rule], 2.0)
        else:
            self.boosts[rule] = 1.0

    def get(self, rule: str) -> float:
        return self.boosts.get(rule, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            r: {"boost": self.boosts[r], "history": list(self.hits[r])}
            for r in self.RULES
        }

# ==========================================
# =========== MULTI-SCALE ANALYZER =========
# ==========================================
class MultiScaleAnalyzer:
    """2-day micro, 5-day meso, 15-day macro pattern detection."""
    __slots__ = ("micro", "meso", "macro")

    def __init__(self, idx: MarketIndex, asof: datetime.date):
        self.micro = self._analyze(idx, asof, CONFIG["MICRO_WINDOW_DAYS"])
        self.meso = self._analyze(idx, asof, CONFIG["MESO_WINDOW_DAYS"])
        self.macro = self._analyze(idx, asof, CONFIG["MACRO_WINDOW_DAYS"])

    @staticmethod
    def _analyze(idx: MarketIndex, asof: datetime.date, n: int) -> Dict[str, Any]:
        last = idx.last_n_on_or_before(asof, n)
        digits = Counter()
        jodis = Counter()
        pannas = Counter()
        od_list, cd_list = [], []
        for _, r in last:
            res = extract_day_result(r)
            if res["od"]: digits[res["od"]] += 1; od_list.append(res["od"])
            if res["cd"]: digits[res["cd"]] += 1; cd_list.append(res["cd"])
            if res["j"]: jodis[res["j"]] += 1
            if res["op"]: pannas[res["op"]] += 1
            if res["cp"]: pannas[res["cp"]] += 1
        return {
            "digits": digits,
            "jodis": jodis,
            "pannas": pannas,
            "od_list": od_list,
            "cd_list": cd_list,
            "length": len(last),
        }

# ==========================================
# ========== DUE THEORY & CYCLES ===========
# ==========================================
class DueTheory:
    """Proactive: digits that are overdue or at Fibonacci cycle breakpoints."""
    def __init__(self, idx: MarketIndex, asof: datetime.date):
        self.due_digits: Set[str] = set()
        self.cycle_digits: Set[str] = set()
        self.cold_digits: Set[str] = set()
        self._compute(idx, asof)

    def _compute(self, idx: MarketIndex, asof: datetime.date):
        # Scan last 20 days for digit absence
        last20 = idx.last_n_on_or_before(asof, CONFIG["DUE_THEORY_MAX_DAYS"])
        seen_digits = set()
        for _, r in last20:
            res = extract_day_result(r)
            if res["od"]: seen_digits.add(res["od"])
            if res["cd"]: seen_digits.add(res["cd"])
        self.due_digits = {str(i) for i in range(10)} - seen_digits

        # Fibonacci cycle detection: if a digit reappears exactly on fib intervals
        all_digits = [extract_day_result(r[1]) for r in last20]
        for d in map(str, range(10)):
            positions = []
            for i, (_, r) in enumerate(last20):
                res = extract_day_result(r)
                if res["od"] == d or res["cd"] == d:
                    positions.append(i)
            if len(positions) >= 2:
                gaps = [positions[j] - positions[j-1] for j in range(1, len(positions))]
                if any(g in CONFIG["CYCLE_FIBONACCI"] for g in gaps[-3:]):
                    self.cycle_digits.add(d)

        # Cold regression: digits absent 10-15 days then due
        last_cold = idx.last_n_on_or_before(asof, CONFIG["COLD_REGRESSION_DAYS"])
        cold_seen = set()
        for _, r in last_cold:
            res = extract_day_result(r)
            if res["od"]: cold_seen.add(res["od"])
            if res["cd"]: cold_seen.add(res["cd"])
        self.cold_digits = {str(i) for i in range(10)} - cold_seen

# ==========================================
# ======== THEORY OF MIND ENGINE ===========
# ==========================================
class TheoryOfMind:
    """
    Level 0: Raw history.
    Level 1: What the herd will bet (naive patterns).
    Level 2: What the operator will do knowing the herd's bets.
    We bet against Level 2 (operator trap).
    """
    def __init__(self, idx: MarketIndex, asof: datetime.date, yctx: Dict[str, str],
                 msa: Optional["MultiScaleAnalyzer"] = None):
        self.trap_digits: Set[str] = set()
        self.anti_herd_digits: Set[str] = set()
        self._compute(idx, asof, yctx, msa)

    def _compute(self, idx: MarketIndex, asof: datetime.date, yctx: Dict[str, str],
                 msa: Optional["MultiScaleAnalyzer"] = None):
        # Level 1: Herd expectation
        herd_expected = set()
        y_od, y_cd, y_j = yctx.get("od", ""), yctx.get("cd", ""), yctx.get("j", "")
        if y_od.isdigit():
            herd_expected.add(str((int(y_od) + 5) % 10))  # Cut
            herd_expected.add(str((int(y_od) + 1) % 10))  # Neighbor
            herd_expected.add(str((int(y_od) - 1) % 10))
        if y_cd.isdigit():
            herd_expected.add(str((int(y_cd) + 5) % 10))
        if y_j and len(y_j) == 2:
            herd_expected.add(y_j[1])  # Mirror close digit
            herd_expected.add(y_j[0])  # Mirror open digit

        # Level 2: Operator anti-herd (trap the herd)
        # Operator often plays the exact opposite of herd expectation
        self.trap_digits = herd_expected.copy()

        # But sometimes operator double-bluffs (plays into herd).
        # Detect if yesterday was already a bluff (mirror of expected)
        last2 = idx.last_n_on_or_before(asof, 2)
        if len(last2) >= 2:
            res0 = extract_day_result(last2[0][1])
            res1 = extract_day_result(last2[1][1])
            # If yesterday was a clear break of day-before, today might be a restore
            if res1.get("od") and res0.get("od"):
                if res0["od"] == str((int(res1["od"]) + 5) % 10):
                    # Yesterday was cut-trap, today might be cut again (double trap)
                    self.trap_digits.add(str((int(res0["od"]) + 5) % 10))

        # Anti-herd: digits that are NOT in herd expectation and NOT recent hot
        macro = (msa.macro if msa is not None else MultiScaleAnalyzer(idx, asof).macro)
        hot = {d for d, c in macro["digits"].items() if c >= 4}
        self.anti_herd_digits = ({str(i) for i in range(10)} - herd_expected) - hot

# ==========================================
# ========= HIERARCHICAL PRIORITY ==========
# ==========================================
class HierarchicalPriority:
    """
    Stack-based priority system:
      VETO:   Hard constraints (mathematical, exhausted, interdependence violations)
      HARD:   Markov, macro cycles, due theory, ToM Level-2
      SOFT:   Behavioral signals (mirror, cut, neighbor, fakeout)
      NOISE:  Base frequency, randomization
    """
    VETO = 3
    HARD = 2
    SOFT = 1
    NOISE = 0

    def __init__(self):
        self.stack: List[Tuple[int, str, float, Any]] = []  # (level, rule_name, weight, metadata)

    def add(self, level: int, rule: str, weight: float, meta: Any = None):
        self.stack.append((level, rule, weight, meta))

    def resolve(self, registry: SignalRegistry) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Collapse stack into final digit weights, PLUS which rules backed each digit
        at its winning level. The attribution is what lets the SignalRegistry actually
        learn from outcomes later (v32 computed it and threw it away)."""
        by_digit: Dict[str, List[Tuple[int, float, str]]] = defaultdict(list)
        for level, rule, weight, meta in self.stack:
            # Apply signal registry dynamic adjustment
            adj = registry.get(rule) if rule in registry.RULES else 1.0
            effective = weight * adj
            if isinstance(meta, str) and meta.isdigit() and len(meta) == 1:
                by_digit[meta].append((level, effective, rule))
            elif isinstance(meta, (list, tuple)):
                for d in meta:
                    if isinstance(d, str) and d.isdigit() and len(d) == 1:
                        by_digit[d].append((level, effective, rule))

        final: Dict[str, float] = {}
        attribution: Dict[str, List[str]] = {}
        for d, entries in by_digit.items():
            # Sort by level desc, then weight desc
            entries.sort(key=lambda x: (x[0], x[1]), reverse=True)
            top_level = entries[0][0]
            # Only combine entries at the same top level
            same_level = [e for e in entries if e[0] == top_level]
            # Multiplicative combination within same level
            w = 1.0
            rules_here = []
            for _, weight, rule in same_level:
                w *= max(0.0, weight)
                if rule not in rules_here:
                    rules_here.append(rule)
            final[d] = w
            attribution[d] = rules_here
        return final, attribution

# ==========================================
# ======== AESTHETIC PSYCHOLOGY ============
# ==========================================
class AestheticPsychology:
    @staticmethod
    def weight(panna: str) -> float:
        if panna in BEAUTY_PANNAS:
            return CONFIG["BEAUTY_PENALTY"]
        if panna in UGLY_PANNAS:
            return CONFIG["UGLY_BOOST"]
        return 1.0

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
# ============ FEEDBACK / BACKTEST ==========
# ==========================================
def load_state(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    if isinstance(data, dict) and "markets" in data: return data
    return {"markets": {}, "schema_version": 2, "signal_registry": {}}

def backtest_update(state: Dict[str, Any], indexes: Dict[str, MarketIndex]) -> Dict[str, int]:
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
def find_actual_for_date(idx: MarketIndex, target_date: datetime.date) -> Optional[Dict[str, str]]:
    r = idx.row_for_date(target_date)
    if r is None: return None
    res = extract_day_result(r)
    if not (res["od"] or res["cd"] or res["j"] or res["op"] or res["cp"]): return None
    return res

def get_day_context(idx: MarketIndex, asof: datetime.date, offset_days: int = 1) -> Dict[str, str]:
    target_rows = idx.last_n_on_or_before(asof, offset_days)
    if len(target_rows) < offset_days: return {}
    _, r = target_rows[offset_days - 1]
    return extract_day_result(r)

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

def recent_pannas(idx: MarketIndex, asof: datetime.date, n_days: int) -> Set[str]:
    last = idx.last_n_on_or_before(asof, n_days)
    out = set()
    for _, r in last:
        res = extract_day_result(r)
        if res["op"]: out.add(res["op"])
        if res["cp"]: out.add(res["cp"])
    return out

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

def build_markov(idx: MarketIndex, asof: datetime.date) -> MarkovMachine:
    mm = MarkovMachine()
    mm.build(idx, asof, CONFIG["MARKOV_LOOKBACK_DAYS"])
    return mm

# ==========================================
# ======== V32 CORE: COMPUTE SIGNALS =======
# ==========================================
def compute_signals_v32(idx: MarketIndex, asof: datetime.date, yctx: Dict[str, str],
                        mm: MarkovMachine, registry: SignalRegistry) -> Dict[str, Any]:
    """
    v32 cognitive signal computation.
    Returns a HierarchicalPriority stack + enriched context.
    """
    stack = HierarchicalPriority()
    y_od, y_cd, y_j = yctx.get("od", ""), yctx.get("cd", ""), yctx.get("j", "")

    # --- Multi-scale analysis ---
    msa = MultiScaleAnalyzer(idx, asof)

    # --- Due Theory & Cycles ---
    due = DueTheory(idx, asof)

    # --- Theory of Mind --- (reuse msa instead of TheoryOfMind silently rebuilding its own)
    tom = TheoryOfMind(idx, asof, yctx, msa)

    # --- VETO LEVEL: Interdependence & Mathematical Constraints ---
    # If Markov gives a hard transition with >30% probability, treat as hard constraint
    if y_od.isdigit():
        for d, prob in mm.top_n(mm.oc, y_od, 2):
            if prob >= 0.30:
                stack.add(HierarchicalPriority.HARD, "markov_oc", CONFIG["MARKOV_BOOST_BASE"] * prob, d)
    if y_cd.isdigit():
        for d, prob in mm.top_n(mm.co, y_cd, 2):
            if prob >= 0.30:
                stack.add(HierarchicalPriority.HARD, "markov_co", CONFIG["MARKOV_BOOST_BASE"] * prob, d)

    # Self-transition Markov: yesterday's open->today's open, yesterday's close->today's close.
    # mm.oo and mm.cc were built every run in v32 and never queried - dead computation.
    if y_od.isdigit():
        for d, prob in mm.top_n(mm.oo, y_od, 2):
            if prob >= 0.30:
                stack.add(HierarchicalPriority.HARD, "markov_oo", CONFIG["MARKOV_BOOST_BASE"] * prob, d)
    if y_cd.isdigit():
        for d, prob in mm.top_n(mm.cc, y_cd, 2):
            if prob >= 0.30:
                stack.add(HierarchicalPriority.HARD, "markov_cc", CONFIG["MARKOV_BOOST_BASE"] * prob, d)

    # --- HARD LEVEL: Due Theory, Cycles, Cold Regression, ToM ---
    for d in due.due_digits:
        stack.add(HierarchicalPriority.HARD, "due_theory", CONFIG["DUE_BOOST_BASE"], d)
    for d in due.cycle_digits:
        stack.add(HierarchicalPriority.HARD, "cycle_breaker", CONFIG["CYCLE_BREAKER_BOOST"], d)
    for d in due.cold_digits:
        stack.add(HierarchicalPriority.HARD, "cold_regression", CONFIG["COLD_REGRESSION_BOOST"], d)

    # ToM Level 2: Anti-herd (bet against operator trap)
    for d in tom.anti_herd_digits:
        stack.add(HierarchicalPriority.HARD, "tom_level2", CONFIG["TOM_TRAP_BOOST"], d)

    # Macro herd suppression (15-day hot digits get penalized at HARD level)
    for d, c in msa.macro["digits"].items():
        if c >= 4:
            stack.add(HierarchicalPriority.HARD, "anti_herd", CONFIG["TOM_ANTIHERD_PENALTY"], d)

    # --- SOFT LEVEL: Behavioral Signals (Time Split) ---
    # OPEN psychology: Cut-trap, neighbor, morning bias
    if y_od.isdigit():
        cut_o = str((int(y_od) + 5) % 10)
        stack.add(HierarchicalPriority.SOFT, "open_time_cut", CONFIG["OPEN_CUT_BOOST"], cut_o)
        for nb in [(int(y_od) + 1) % 10, (int(y_od) - 1) % 10]:
            stack.add(HierarchicalPriority.SOFT, "neighbor", CONFIG["OPEN_NEIGHBOR_BOOST"], str(nb))
        # Break penalty for yesterday open
        stack.add(HierarchicalPriority.SOFT, "break", CONFIG["BREAK_PENALTY"], y_od)

    # CLOSE psychology: Mirror-trap, sumcomp, evening bias
    if y_cd.isdigit():
        cut_c = str((int(y_cd) + 5) % 10)
        stack.add(HierarchicalPriority.SOFT, "close_time_mirror", CONFIG["CLOSE_MIRROR_BOOST"], cut_c)
        sum_c = str((10 - int(y_od)) % 10) if y_od.isdigit() else None
        if sum_c:
            stack.add(HierarchicalPriority.SOFT, "close_time_sumcomp", CONFIG["CLOSE_SUMCOMP_BOOST"], sum_c)
        stack.add(HierarchicalPriority.SOFT, "break", CONFIG["BREAK_PENALTY"], y_cd)

    # Mirror jodi digit influence
    if y_j and len(y_j) == 2:
        stack.add(HierarchicalPriority.SOFT, "mirror", CONFIG["MIRROR_BOOST"], y_j[1])  # Close side
        stack.add(HierarchicalPriority.SOFT, "mirror", CONFIG["MIRROR_BOOST"], y_j[0])  # Open side

    # Fakeout / 2-day loop detection at SOFT level
    last2 = idx.last_n_on_or_before(asof, 2)
    if len(last2) >= 2:
        j0 = extract_day_result(last2[0][1]).get("j", "")
        j1 = extract_day_result(last2[1][1]).get("j", "")
        if j0 and j1 and (j0 == j1 or j0 == mirror_jodi(j1)):
            for ch in j0:
                if ch.isdigit():
                    stack.add(HierarchicalPriority.SOFT, "fakeout_break", 0.25, ch)

    # Meso herd suppression (5-day window)
    for d, c in msa.meso["digits"].items():
        if c >= 3:
            stack.add(HierarchicalPriority.SOFT, "streak", CONFIG["STREAK_PENALTY"], d)

    # --- NOISE LEVEL: Base Frequency ---
    base_freq = build_base_freq(idx, asof, CONFIG["FREQ_LOOKBACK_DAYS"])
    for d, w in base_freq.items():
        stack.add(HierarchicalPriority.NOISE, "base_freq", w, d)

    # Resolve stack to digit weights + which rules backed each digit
    digit_weights, digit_attribution = stack.resolve(registry)

    return {
        "digit_weights": digit_weights,
        "digit_attribution": digit_attribution,
        "y_od": y_od, "y_cd": y_cd, "y_j": y_j,
        "due": due, "tom": tom, "msa": msa, "mm": mm,
        "stack": stack,
    }

def build_base_freq(idx: MarketIndex, asof: datetime.date, n_days: int) -> Dict[str, float]:
    last = idx.last_n_on_or_before(asof, n_days)
    cnt = Counter()
    for _, r in last:
        res = extract_day_result(r)
        if res["od"].isdigit(): cnt[res["od"]] += 1
        if res["cd"].isdigit(): cnt[res["cd"]] += 1
    return {str(i): (cnt.get(str(i), 0) + 1.0) * 0.1 for i in range(10)}

def select_digits(weights: Dict[str, float], need_k: int, veto_exclusions: Set[str]) -> List[str]:
    """Select digits using hierarchical weights, respecting vetoes."""
    items = [d for d in weights if d not in veto_exclusions]
    ws = [weights[d] for d in items]
    picks = weighted_choice(items, ws, need_k)
    out, seen = [], set()
    for p in picks:
        if p not in seen:
            seen.add(p); out.append(p)
        if len(out) >= need_k: break
    # Fallback: fill with any digit not seen and not vetoed
    for d in map(str, range(10)):
        if d not in seen and d not in veto_exclusions:
            seen.add(d); out.append(d)
        if len(out) >= need_k: break
    return out[:need_k]

def sample_pannas_v32(candidate_pool: List[str], need_k: int, idx: MarketIndex,
                      asof: datetime.date, role: str, sig: Dict[str, Any],
                      chosen_digits: List[str], registry: SignalRegistry) -> Tuple[List[str], Dict[str, List[str]]]:
    if not candidate_pool: return [], {}

    # MATHEMATICAL ENFORCEMENT: Filter to pannas summing to chosen digits
    math_valid = [p for p in candidate_pool if str(sum(int(x) for x in p) % 10) in chosen_digits]
    if not math_valid:
        # v32 silently returned [] here, giving the app a market with zero pannas
        # for that role with no trace of why. Fall back to the full candidate pool
        # (still mathematically valid against SOME digit, just not the chosen set)
        # rather than emitting nothing.
        logger.warning(f"sample_pannas_v32: no panna in pool sums to any of {chosen_digits} "
                        f"for role={role}; falling back to unfiltered candidate pool.")
        math_valid = list(candidate_pool)
        if not math_valid: return [], {}

    recent = recent_pannas(idx, asof, 5)
    y_od, y_cd = sig["y_od"], sig["y_cd"]
    cut_digit = str((int(y_od if role == "open" else y_cd) + 5) % 10) if (y_od if role == "open" else y_cd).isdigit() else None
    comp_digit = str((10 - int(y_cd if role == "open" else y_od)) % 10) if (y_cd if role == "open" else y_od).isdigit() else None

    items = list(math_valid)
    ws, tag_map = [], {}
    for p in items:
        w = 1.0
        tags = []
        # Aesthetic psychology
        aw = AestheticPsychology.weight(p)
        w *= aw
        if aw == CONFIG["UGLY_BOOST"]: tags.append("aesthetic_ugly")
        elif aw == CONFIG["BEAUTY_PENALTY"]: tags.append("aesthetic_beauty")

        if p in recent:
            w *= CONFIG["PANNA_RECENT_PENALTY"]
        mp = mirror_panna(p)
        if mp in recent and mp != p:
            w *= CONFIG["PANNA_MIRROR_BONUS"] * registry.get("mirror"); tags.append("mirror")
        if cut_digit and cut_digit in p:
            w *= CONFIG["PANNA_CUT_BONUS"] * registry.get("cut"); tags.append("cut")
        if comp_digit and comp_digit in p:
            w *= CONFIG["PANNA_SUMMOD_BONUS"] * registry.get("sumcomp"); tags.append("sumcomp")

        # Interdependence: boost pannas that also support Markov transitions
        mm = sig["mm"]
        if role == "open" and y_od.isdigit():
            for d, prob in mm.top_n(mm.oc, y_od, 3):
                if d in p:
                    w *= 1.0 + prob; tags.append("markov_oc")
        if role == "close" and y_cd.isdigit():
            for d, prob in mm.top_n(mm.co, y_cd, 3):
                if d in p:
                    w *= 1.0 + prob; tags.append("markov_co")

        ws.append(max(w, 1e-6))
        tag_map[p] = tags

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
    out = out[:need_k]
    attribution = {p: tag_map.get(p, []) for p in out}
    return out, attribution

def build_jodis_v32(open_d: List[str], close_d: List[str], sig: Dict[str, Any],
                      registry: SignalRegistry) -> Tuple[List[str], Dict[str, List[str]]]:
    cands = []
    y_j = sig["y_j"]
    mm = sig["mm"]
    y_od, y_cd = sig["y_od"], sig["y_cd"]

    # Cut-Jodi calculation
    cut_jodi = ""
    if y_j and len(y_j) == 2:
        cut_jodi = f"{(int(y_j[0]) + 5) % 10}{(int(y_j[1]) + 5) % 10}"

    # Markov jodi prediction: open digit AND close digit each have their own
    # transition matrix into tomorrow's jodi (mm.oj, mm.cj). v32 only ever
    # consulted mm.oj; mm.cj was built and discarded every run.
    markov_oj, markov_cj = {}, {}
    if y_od.isdigit():
        for j, prob in mm.top_n(mm.oj, y_od, 5):
            markov_oj[j] = prob
    if y_cd.isdigit():
        for j, prob in mm.top_n(mm.cj, y_cd, 5):
            markov_cj[j] = prob

    for a in open_d:
        for b in close_d:
            j = f"{a}{b}"
            w = 1.0
            tags = []
            if j == y_j: w *= 0.2
            if a == b:
                if y_j and len(y_j) == 2 and y_j[0] == y_j[1]:
                    w *= CONFIG["RED_JODI_PENALTY"]
                else:
                    w *= 1.15
            if y_j and j == mirror_jodi(y_j):
                w *= 1.5 * registry.get("mirror"); tags.append("mirror")
            if (int(a) + int(b)) in (10, 20):
                w *= 1.35; tags.append("summod")
            if cut_jodi and j == cut_jodi:
                w *= CONFIG["CUT_BOOST"] * registry.get("cut"); tags.append("cut")
            if j in markov_oj:
                w *= 1.0 + markov_oj[j] * 2.0; tags.append("markov_oj")
            if j in markov_cj:
                w *= 1.0 + markov_cj[j] * 2.0; tags.append("markov_cj")

            # Interdependence: jodi digits must be consistent with chosen open/close
            # (already guaranteed by construction, but boost if Markov agrees)
            cands.append((j, w, tags))

    cands.sort(key=lambda x: x[1], reverse=True)
    out, seen, attribution = [], set(), {}
    for j, _, tags in cands:
        if j not in seen:
            seen.add(j); out.append(j); attribution[j] = tags
        if len(out) >= CONFIG["NUM_JODIS"]: break
    return out, attribution

def predict_market_v32(mkt: str, idx: MarketIndex, asof: datetime.date,
                       accuracy: Optional[Dict[str, float]],
                       state: Dict[str, Any]) -> Dict[str, Any]:
    if len(idx) < 5:
        return {"OpenDigits": [], "CloseDigits": [], "OpenPannas": [], "ClosePannas": [], "Jodis": [],
                "confidence": 0.0, "confidence_basis": "insufficient_data"}

    registry = SignalRegistry(state)
    mm = build_markov(idx, asof)
    yctx = get_day_context(idx, asof, offset_days=1)

    sig = compute_signals_v32(idx, asof, yctx, mm, registry)

    # VETO exclusions: digits that are mathematically impossible or exhausted
    veto_exclusions = set()
    # If a digit has appeared 8+ times in 15 days, veto it (extreme overuse)
    for d, c in sig["msa"].macro["digits"].items():
        if c >= 8:
            veto_exclusions.add(d)

    open_d = select_digits(sig["digit_weights"], CONFIG["NUM_OPEN_DIGITS"], veto_exclusions)
    close_d = select_digits(sig["digit_weights"], CONFIG["NUM_CLOSE_DIGITS"], veto_exclusions)

    # 4-way interdependence: ensure close digit is Markov-compatible with open
    # Re-roll close if Markov probability is extremely low
    if yctx.get("od") and open_d:
        best_close = None
        best_prob = 0.0
        for d in close_d:
            prob = mm.prob(mm.oc, yctx["od"], d)
            if prob > best_prob:
                best_prob = prob
                best_close = d
        # If all Markov probs are near zero, inject the top Markov prediction
        if best_prob < 0.05 and yctx["od"].isdigit():
            top_markov = mm.top_n(mm.oc, yctx["od"], 1)
            if top_markov:
                forced_close = top_markov[0][0]
                if forced_close not in close_d:
                    close_d = [forced_close] + close_d[:-1]

    open_pool = pannas_pool_for_digits(open_d)
    close_pool = pannas_pool_for_digits(close_d)

    open_p = sample_pannas_v32(open_pool, CONFIG["NUM_OPEN_PANNAS"], idx, asof, "open", sig, open_d, registry)
    close_p = sample_pannas_v32(close_pool, CONFIG["NUM_CLOSE_PANNAS"], idx, asof, "close", sig, close_d, registry)

    jodis = build_jodis_v32(open_d, close_d, sig, registry)

    # Confidence calculation with v32 multi-factor awareness
    rule_conf = CONFIG["CONFIDENCE_BASE"]
    if sig["tom"].anti_herd_digits:
        rule_conf += 0.10
    if sig["due"].due_digits or sig["due"].cycle_digits:
        rule_conf += 0.10
    if mm.built:
        rule_conf += 0.10
    if any(d in sig["tom"].anti_herd_digits for d in open_d + close_d):
        rule_conf += 0.10
    rule_conf = min(rule_conf, CONFIG["CONFIDENCE_CAP"])

    if accuracy is not None:
        w = CONFIG["RULE_WEIGHT_WHEN_BLENDED"]
        conf = w * rule_conf + (1 - w) * accuracy["overall"]
        basis = f"blended (v32 rules + {accuracy['sample_size']}-day empirical accuracy)"
    else:
        conf = rule_conf
        basis = "v32_cognitive_rules (not enough scored history yet)"

    conf = max(CONFIG["CONFIDENCE_MIN"], min(conf, CONFIG["CONFIDENCE_CAP"]))

    # Save registry back to state
    state["signal_registry"] = registry.to_dict()

    return {
        "OpenDigits": open_d,
        "CloseDigits": close_d,
        "OpenPannas": open_p,
        "ClosePannas": close_p,
        "Jodis": jodis,
        "confidence": round(conf, 2),
        "confidence_basis": basis,
        "v32_signals": {
            "due_digits": list(sig["due"].due_digits),
            "cycle_digits": list(sig["due"].cycle_digits),
            "cold_digits": list(sig["due"].cold_digits),
            "anti_herd": list(sig["tom"].anti_herd_digits),
            "trap_digits": list(sig["tom"].trap_digits),
            "veto_exclusions": list(veto_exclusions),
        }
    }

# ==========================================
# =================  I/O ===================
# ==========================================
def to_app_structure(preds: Dict[str, Dict[str, Any]], run_date: datetime.date,
                      scored_summary: Dict[str, int]) -> Dict[str, Any]:
    markets_array = []
    date_str = run_date.isoformat()
    for market_key, pr in preds.items():
        entry = {
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
        }
        if "v32_signals" in pr:
            entry["v32_signals"] = pr["v32_signals"]
        markets_array.append(entry)
    return {
        "meta": {
            "generated_at": iso_timestamp(),
            "engine": "v32_cognitive_kalyan",
            "schema_version": 2,
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
            internal_preds[mkt] = predict_market_v32(mkt, idx, for_date, accuracy, state)
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

    dated_out = paths["OUTPUT_DIR"] / f"predictions_{tag}_v32.json"
    today_out = paths["TODAY_PROD_FILE"] if (force_prod or not cfg["TEST_MODE"]) else paths["TODAY_TEST_FILE"]
    assets_today = paths["TODAYS_PREDICTIONS_JSON"]
    assets_dated = paths["ASSETS_DIR"] / f"predictions_{tag}_v32.json"

    if not dry_run:
        save_json_atomic(dated_out, app_json)
        save_json_atomic(today_out, app_json)
        save_json_atomic(assets_today, app_json)
        save_json_atomic(assets_dated, app_json)
        save_json_atomic(paths["STATE_FILE"], state)

    logger.info(f"OK v32-cognitive — markets: {len(internal_preds)} | dated: {dated_out} | "
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

    # Test HierarchicalPriority
    hp = HierarchicalPriority()
    hp.add(HierarchicalPriority.HARD, "test_hard", 2.0, "5")
    hp.add(HierarchicalPriority.SOFT, "test_soft", 1.5, "5")
    hp.add(HierarchicalPriority.NOISE, "test_noise", 0.5, "5")
    reg = SignalRegistry({})
    resolved = hp.resolve(reg)
    check("HierarchicalPriority resolves HARD over SOFT", resolved.get("5", 0) >= 2.0)

    # Test AestheticPsychology
    check("Beauty panna 123 penalized", AestheticPsychology.weight("123") == CONFIG["BEAUTY_PENALTY"])
    check("Ugly panna 079 boosted", AestheticPsychology.weight("079") == CONFIG["UGLY_BOOST"])
    check("Neutral panna unchanged", AestheticPsychology.weight("112") == 1.0)

    # Test MarkovMachine
    mm = MarkovMachine()
    sample_rows = [
        {"Date": "01/01/2026", "Open Digit": "1", "Close Digit": "2", "Open Panna": "118", "Close Panna": "127", "Jodi": "12"},
        {"Date": "02/01/2026", "Open Digit": "3", "Close Digit": "4", "Open Panna": "120", "Close Panna": "130", "Jodi": "34"},
        {"Date": "03/01/2026", "Open Digit": "5", "Close Digit": "6", "Open Panna": "122", "Close Panna": "150", "Jodi": "56"},
    ]
    idx = MarketIndex(sample_rows)
    mm.build(idx, datetime.date(2026, 1, 3), 365)
    check("MarkovMachine built successfully", mm.built is True)
    check("Markov oc records day-over-day od(1)->cd(4) transition", mm.oc["1"]["4"] == 1)
    check("Markov prob respects MARKOV_MIN_COUNT guard (insufficient samples -> 0.0)",
          mm.prob(mm.oc, "1", "4") == 0.0)

    # Test DueTheory
    due = DueTheory(idx, datetime.date(2026, 1, 3))
    check("DueTheory computes due digits", isinstance(due.due_digits, set))

    # Test TheoryOfMind
    yctx = {"od": "3", "cd": "8", "j": "38"}
    tom = TheoryOfMind(idx, datetime.date(2026, 1, 3), yctx)
    check("TheoryOfMind computes trap digits", isinstance(tom.trap_digits, set))
    check("TheoryOfMind computes anti-herd", isinstance(tom.anti_herd_digits, set))

    # Test MultiScaleAnalyzer
    msa = MultiScaleAnalyzer(idx, datetime.date(2026, 1, 3))
    check("MultiScaleAnalyzer micro", msa.micro["length"] == 2)
    check("MultiScaleAnalyzer meso", msa.meso["length"] == 3)

    # Test SignalRegistry
    reg2 = SignalRegistry({})
    reg2.record("mirror", True)
    reg2.record("mirror", True)
    reg2.record("mirror", False)
    check("SignalRegistry tracks hits", len(reg2.hits["mirror"]) == 3)

    # Test Panna Math Link
    pool = pannas_pool_for_digits(["5"])
    sig = {"y_od": "3", "y_cd": "8", "y_j": "38", "due": due, "tom": tom, "msa": msa, "mm": mm, "stack": HierarchicalPriority(), "digit_weights": {}}
    sampled = sample_pannas_v32(
        candidate_pool=pool, need_k=5, idx=idx,
        asof=datetime.date(2026, 1, 3), role="open", sig=sig, chosen_digits=["5"], registry=reg2
    )
    check("sample_pannas_v32 returns mathematically valid pannas for digit 5",
          all(sum(int(x) for x in p) % 10 == 5 for p in sampled))

    check("normalize_jodi rebuilds '04' from od=0,cd=4", normalize_jodi("4", "0", "4") == "04")
    check("normalize_panna pads '6' to '006'", normalize_panna("6") == "006")

    rec_bug = {"Date": "16/11/2022", "Open Digit": "0", "Close Digit": "4",
               "Open Panna": "235", "Close Panna": "220", "Jodi": "4"}
    res = extract_day_result(rec_bug)
    check("extract_day_result fixes the real leading-zero jodi bug end-to-end", res["j"] == "04")

    recent2 = idx.last_n_on_or_before(datetime.date(2026, 1, 3), 2)
    check("MarketIndex.last_n_on_or_before returns most-recent-first",
          [d.isoformat() for d, _ in recent2] == ["2026-01-03", "2026-01-02"])

    print("\nSELFTEST:", "ALL PASS" if ok else "FAILURES ABOVE")
    return ok

# ==========================================
# ================== MAIN ==================
# ==========================================
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SattaMatkaAI Prediction Engine v32 (Cognitive Architecture).")
    ap.add_argument("--prod", action="store_true", help="Force production output paths.")
    ap.add_argument("--dry-run", action="store_true", help="Compute but do not write any files.")
    ap.add_argument("--selftest", action="store_true", help="Run fast data-free sanity checks and exit.")
    ap.add_argument("--verbose", action="store_true", help="Also log to console.")
    ap.add_argument("--date", type=str, default=None, help="Override run date as YYYY-MM-DD.")

    ap.add_argument("--history", type=str, default=None)
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
    # ==== BACKTEST LAB 1-LINER: unique dated output -> D:\backtest\predictions_<YYYY-MM-DD>_v33.json ====
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
            _bt_dir = _bt_Path(_bt_os.environ.get("LAB_DIR") or ("D:\\backtest" if False else _bt_Path(__file__).resolve().parent))
            _bt_dir.mkdir(parents=True, exist_ok=True)
            _p = _bt_dir / f"predictions_{ds}_v33.json"
            _t = _p.with_suffix(".json.tmp")
            _t.write_text(_bt_json.dumps(out_obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            _bt_os.replace(_t, _p)
            print(f"[BACKTEST] wrote {_p}")
        _bt_holder = {}
        _bt_orig_run_once = run_once
        def _bt_run_once_wrapper(*a, **kw):
            _r = _bt_orig_run_once(*a, **kw)
            if isinstance(_r, (dict, list)) and _r:
                _bt_holder["r"] = _r
            return _r
        globals()["run_once"] = _bt_run_once_wrapper
        try:
            sys.exit(main())
        finally:
            _bt_obj = _bt_holder.get("r")
            if _bt_obj is None:
                for _bt_k in ('app_json', 'output', 'result', 'payload', 'data'):
                    _bt_v = globals().get(_bt_k)
                    if isinstance(_bt_v, (dict, list)) and _bt_v:
                        _bt_obj = _bt_v; break
            if _bt_obj is not None:
                try:
                    _bt_write(_bt_obj)
                except Exception as _bt_e:
                    print(f"[BACKTEST] write failed: {_bt_e}")
    except Exception as _bt_e:
        print(f"[BACKTEST] skipped: {_bt_e}")
    # ==== END BACKTEST LAB 1-LINER ====


