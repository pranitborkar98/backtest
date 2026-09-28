# -*- coding: utf-8 -*-
"""
prediction_engine_ml_panna_first_v2.py — SattaMatkaAI (Panna-First ML Engine v2)

Fixes vs v1:
  1. Added missing `import logging.handlers` (Windows crash fix).
  2. Model caching via joblib: daily runs drop from ~7 min → ~20 sec.
  3. --clear-cache flag to force full retrain when needed.
  4. Explicit save confirmation so you know files actually exist.
"""

import os
import sys
import json
import random
import datetime
import argparse
import logging
import logging.handlers  # <-- FIX: Windows requires this explicit import
import tempfile
import hashlib
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

# ------------------------------------------------------------------
# OPTIONAL ML DEPS
# ------------------------------------------------------------------
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    warnings.warn("pip install scikit-learn pandas numpy")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# ------------------------------------------------------------------
# CONFIG & PATHS
# ------------------------------------------------------------------
_DEFAULTS = {
    "HISTORY_FILE": r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\sattaboss-data\data\all_markets_history.json",
    "OUTPUT_DIR":   r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output",
    "STATE_FILE":   r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\state\engine_ml_panna_state.json",
    "TODAY_TEST_FILE": r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\todays_predictions_ml_TEST.json",
    "TODAY_PROD_FILE": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\todays_predictions.json",
    "TODAYS_PREDICTIONS_JSON": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\todays_predictions.json",
    "LOG_FILE":   r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\engine_ml_panna.log",
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

    "NEGATIVE_SAMPLES_PER_POSITIVE": 15,
    "MARKOV_PANNA_LOOKBACK_DAYS": 365,
    "MIN_HISTORY_FOR_ML": 60,

    "CONFIDENCE_BASE": 0.35,
    "CONFIDENCE_CAP": 0.85,
    "CONFIDENCE_MIN": 0.05,

    "STATE_HISTORY_CAP_PER_MARKET": 90,
    "ENABLE_LOGGING": True,
    "LOG_TO_CONSOLE": False,
    "LOG_MAX_MB": 5,
    "LOG_BACKUPS": 3,

    "TEST_MODE": False,
    "TEST_SEED": 20251109,
}

# ------------------------------------------------------------------
# VALID PANNAS (identical pool to v33)
# ------------------------------------------------------------------
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

ALL_VALID_PANNAS: List[str] = []
for lst in VALID_PANNAS_BY_LAST_DIGIT.values():
    ALL_VALID_PANNAS.extend(lst)
ALL_VALID_PANNAS = sorted(set(ALL_VALID_PANNAS))
if CONFIG["EXCLUDE_TRIPLE_PANNAS"]:
    ALL_VALID_PANNAS = [p for p in ALL_VALID_PANNAS if not (len(set(p)) == 1 and len(p) == 3)]

PANNA_TO_DIGIT = {p: str(sum(int(ch) for ch in p) % 10) for p in ALL_VALID_PANNAS}
DIGIT_TO_PANNAS: Dict[str, List[str]] = defaultdict(list)
for p, d in PANNA_TO_DIGIT.items():
    DIGIT_TO_PANNAS[d].append(p)

# ------------------------------------------------------------------
# AESTHETIC MAPS
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
logger = logging.getLogger("pred_engine_ml_panna")
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
            print(f"File logging failed: {e}")
    if enable_console or not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

# ------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------
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
        logger.warning(f"JSON parse failed at {p}: {e}")
        return None

def save_json_atomic(p: Path, obj: Any) -> None:
    p = Path(p)
    safe_mkdir(p.parent)
    fd, tmp_path = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
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

def normalize_panna(raw: str) -> str:
    if raw.isdigit() and 1 <= len(raw) <= 3:
        return raw.zfill(3)
    return ""

def normalize_jodi(raw: str, od: str, cd: str) -> str:
    if od.isdigit() and cd.isdigit() and len(od) == 1 and len(cd) == 1:
        return f"{od}{cd}"
    if raw.isdigit() and len(raw) <= 2:
        return raw.zfill(2)
    return ""

def get_field(rec: Dict[str, Any], *keys: str) -> str:
    if not isinstance(rec, dict):
        return ""
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return str(rec[k]).strip()
    norm = {k.lower().replace(" ", "").replace("_", ""): k for k in keys}
    for rk, rv in rec.items():
        nk = str(rk).lower().replace(" ", "").replace("_", "")
        if nk in norm and rv not in (None, ""):
            return str(rv).strip()
    return ""

def extract_day_result(rec: Dict[str, Any]) -> Dict[str, str]:
    od = get_field(rec, "Open Digit", "OpenDigit", "open_digit")
    cd = get_field(rec, "Close Digit", "CloseDigit", "close_digit")
    j_raw = get_field(rec, "Jodi", "jodi")
    op = normalize_panna(get_field(rec, "Open Panna", "OpenPanna", "open_panna"))
    cp = normalize_panna(get_field(rec, "Close Panna", "ClosePanna", "close_panna"))
    j = normalize_jodi(j_raw, od, cd)
    return {"od": od, "cd": cd, "j": j, "op": op, "cp": cp}

# ------------------------------------------------------------------
# DATA INGEST
# ------------------------------------------------------------------
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
        raise RuntimeError("Historical data unreadable/empty.")
    check = 0
    for mkt, rows in all_hist.items():
        if not isinstance(rows, list):
            raise TypeError(f"{mkt}: rows must be list.")
        for r in rows:
            if not isinstance(r, dict):
                raise TypeError(f"{mkt}: row must be dict.")
            if not any(k in r for k in ("OpenPanna", "open_panna", "ClosePanna", "close_panna", "Jodi", "jodi", "Date", "date")):
                raise ValueError(f"{mkt}: missing expected keys.")
        check += 1
        if check >= 3:
            break

# ------------------------------------------------------------------
# FEATURE ENGINEERING (Panna-First)
# ------------------------------------------------------------------
class PannaFeatureBuilder:
    @staticmethod
    def build_context(rows: List[Dict[str, Any]], asof: datetime.date) -> Dict[str, Any]:
        dated = []
        for r in rows:
            d = parse_date(r.get("Date") or r.get("date"))
            if d and d <= asof:
                dated.append((d, extract_day_result(r)))
        dated.sort(key=lambda x: x[0], reverse=True)

        ctx: Dict[str, Any] = {
            "n_history": len(dated),
            "dow": asof.weekday(),
            "dom": asof.day,
            "month": asof.month,
        }

        for lag, (_, res) in enumerate(dated[:7], 1):
            ctx[f"op_lag{lag}"] = res["op"]
            ctx[f"cp_lag{lag}"] = res["cp"]
            ctx[f"od_lag{lag}"] = res["od"]
            ctx[f"cd_lag{lag}"] = res["cd"]
            ctx[f"j_lag{lag}"] = res["j"]

        last15 = dated[:15]
        digit_cnt = Counter()
        panna_cnt = Counter()
        for _, res in last15:
            if res["od"]: digit_cnt[res["od"]] += 1
            if res["cd"]: digit_cnt[res["cd"]] += 1
            if res["op"]: panna_cnt[res["op"]] += 1
            if res["cp"]: panna_cnt[res["cp"]] += 1
        ctx["max_digit_freq_15d"] = max(digit_cnt.values()) if digit_cnt else 0
        ctx["unique_digits_15d"] = len(digit_cnt)

        j1 = dated[0][1]["j"] if len(dated) > 0 else ""
        j2 = dated[1][1]["j"] if len(dated) > 1 else ""
        ctx["fakeout_active"] = 1 if (j1 and j2 and (j1 == j2 or j1 == j2[::-1])) else 0

        ctx["y_od"] = dated[0][1]["od"] if dated else ""
        ctx["y_cd"] = dated[0][1]["cd"] if dated else ""
        ctx["y_op"] = dated[0][1]["op"] if dated else ""
        ctx["y_cp"] = dated[0][1]["cp"] if dated else ""

        ctx["cut_od"] = str((int(ctx["y_od"]) + 5) % 10) if ctx["y_od"].isdigit() else ""
        ctx["cut_cd"] = str((int(ctx["y_cd"]) + 5) % 10) if ctx["y_cd"].isdigit() else ""

        last12_digits = set()
        for _, res in dated[:12]:
            if res["od"]: last12_digits.add(res["od"])
            if res["cd"]: last12_digits.add(res["cd"])
        ctx["cold_digits"] = [d for d in map(str, range(10)) if d not in last12_digits]

        panna_recency: Dict[str, int] = {}
        for days_back, (_, res) in enumerate(dated[:30], 1):
            for p in (res["op"], res["cp"]):
                if p and p not in panna_recency:
                    panna_recency[p] = days_back
        ctx["panna_recency"] = panna_recency

        markov: Dict[str, Counter] = defaultdict(Counter)
        for i in range(len(dated) - 1):
            curr_op_digit = PANNA_TO_DIGIT.get(dated[i][1]["op"], "")
            next_op_digit = PANNA_TO_DIGIT.get(dated[i + 1][1]["op"], "")
            if curr_op_digit and next_op_digit:
                markov[curr_op_digit][next_op_digit] += 1
        ctx["markov_digit"] = markov

        return ctx

    @staticmethod
    def build_candidate_features(ctx: Dict[str, Any], panna: str, role: str = "open") -> List[float]:
        d = PANNA_TO_DIGIT.get(panna, "0")
        feats = []
        y_digit = ctx.get("y_od", "") if role == "open" else ctx.get("y_cd", "")
        cut_digit = ctx.get("cut_od", "") if role == "open" else ctx.get("cut_cd", "")
        y_panna = ctx.get("y_op", "") if role == "open" else ctx.get("y_cp", "")

        feats.append(int(d))
        feats.append(1 if len(set(panna)) == 2 else 0)
        feats.append(1 if panna in BEAUTY_PANNAS else 0)
        feats.append(1 if panna in UGLY_PANNAS else 0)
        rec = ctx["panna_recency"].get(panna, 30)
        feats.append(min(rec, 30))
        feats.append(1 if panna == y_panna else 0)
        mirror_y = y_panna[::-1] if len(y_panna) == 3 else ""
        feats.append(1 if panna == mirror_y else 0)
        feats.append(1 if y_digit and y_digit in panna else 0)
        feats.append(1 if cut_digit and cut_digit in panna else 0)
        feats.append(ctx.get("max_digit_freq_15d", 0) / 15.0)

        markov_score = 0.0
        if y_digit and d:
            mc = ctx.get("markov_digit", {}).get(y_digit, Counter())
            total = sum(mc.values())
            if total:
                markov_score = mc.get(d, 0) / total
        feats.append(markov_score)
        feats.append(1 if d in ctx.get("cold_digits", []) else 0)
        feats.append(ctx.get("fakeout_active", 0))
        feats.append(0 if role == "open" else 1)
        feats.append(ctx.get("dow", 0))
        feats.append(ctx.get("dom", 1))
        feats.append(ctx.get("month", 1))
        return feats

# ------------------------------------------------------------------
# ML RANKER
# ------------------------------------------------------------------
class PannaRanker:
    def __init__(self):
        self.model: Optional[Any] = None
        self.fitted = False

    def _build_training_data(self, rows: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[int]]:
        X: List[List[float]] = []
        y: List[int] = []
        dated = []
        for r in rows:
            d = parse_date(r.get("Date") or r.get("date"))
            if d:
                dated.append((d, extract_day_result(r)))
        dated.sort(key=lambda x: x[0])

        neg_count = CONFIG["NEGATIVE_SAMPLES_PER_POSITIVE"]

        for i in range(1, len(dated)):
            asof = dated[i][0]
            today_res = dated[i][1]
            ctx = PannaFeatureBuilder.build_context([r for _, r in dated[:i]], asof)

            for role, target_panna in (("open", today_res["op"]), ("close", today_res["cp"])):
                if not target_panna or target_panna not in ALL_VALID_PANNAS:
                    continue
                X.append(PannaFeatureBuilder.build_candidate_features(ctx, target_panna, role))
                y.append(1)
                pool = [p for p in ALL_VALID_PANNAS if p != target_panna]
                negs = random.sample(pool, min(neg_count, len(pool)))
                for neg in negs:
                    X.append(PannaFeatureBuilder.build_candidate_features(ctx, neg, role))
                    y.append(0)
        return X, y

    def fit(self, rows: List[Dict[str, Any]]) -> bool:
        if not ML_AVAILABLE:
            logger.warning("ML unavailable; using heuristic fallback.")
            return False
        X, y = self._build_training_data(rows)
        if len(X) < 200 or len(set(y)) < 2:
            logger.info("Insufficient training data.")
            return False
        try:
            self.model = HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.1, max_depth=5, random_state=42, early_stopping=False
            )
            self.model.fit(np.array(X), np.array(y))
            self.fitted = True
            logger.info(f"Trained on {len(X)} samples ({sum(y)} positives).")
            return True
        except Exception as e:
            logger.warning(f"Training failed: {e}")
            return False

    def score_all(self, ctx: Dict[str, Any], role: str = "open") -> List[Tuple[str, float]]:
        if not self.fitted or self.model is None:
            return self._heuristic_score_all(ctx, role)
        candidates = ALL_VALID_PANNAS
        X = np.array([PannaFeatureBuilder.build_candidate_features(ctx, p, role) for p in candidates])
        try:
            probs = self.model.predict_proba(X)[:, 1]
            return sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.warning(f"ML scoring failed: {e}")
            return self._heuristic_score_all(ctx, role)

    @staticmethod
    def _heuristic_score_all(ctx: Dict[str, Any], role: str = "open") -> List[Tuple[str, float]]:
        y_digit = ctx.get("y_od", "") if role == "open" else ctx.get("y_cd", "")
        cut_digit = ctx.get("cut_od", "") if role == "open" else ctx.get("cut_cd", "")
        y_panna = ctx.get("y_op", "") if role == "open" else ctx.get("y_cp", "")
        mirror_y = y_panna[::-1] if len(y_panna) == 3 else ""
        recency = ctx.get("panna_recency", {})
        scored = []
        for p in ALL_VALID_PANNAS:
            d = PANNA_TO_DIGIT[p]
            w = 1.0
            if p == y_panna: w *= 0.2
            if p == mirror_y: w *= 1.3
            if d == y_digit: w *= 0.3
            if d == cut_digit: w *= 1.4
            if d in ctx.get("cold_digits", []): w *= 1.2
            if p in BEAUTY_PANNAS: w *= 0.6
            if p in UGLY_PANNAS: w *= 1.3
            if p in recency and recency[p] <= 3: w *= 0.4
            scored.append((p, w))
        return sorted(scored, key=lambda x: x[1], reverse=True)

# ------------------------------------------------------------------
# MODEL CACHE
# ------------------------------------------------------------------
def _hash_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "empty"
    h = hashlib.md5()
    h.update(str(len(rows)).encode())
    sample = json.dumps(rows[-30:], sort_keys=True, default=str)
    h.update(sample.encode())
    return h.hexdigest()[:16]

def get_cached_or_train_ranker(mkt: str, rows: List[Dict[str, Any]], cache_dir: Path, clear_cache: bool) -> PannaRanker:
    if not JOBLIB_AVAILABLE or not ML_AVAILABLE:
        ranker = PannaRanker()
        ranker.fit(rows)
        return ranker

    data_hash = _hash_rows(rows)
    cache_file = cache_dir / f"{mkt}_{data_hash}.joblib"

    if not clear_cache and cache_file.exists():
        try:
            ranker = joblib.load(cache_file)
            logger.debug(f"Cache hit for {mkt}")
            return ranker
        except Exception:
            pass

    ranker = PannaRanker()
    ranker.fit(rows)
    if ranker.fitted:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(ranker, cache_file)
            logger.debug(f"Cache saved for {mkt}")
        except Exception as e:
            logger.warning(f"Cache save failed for {mkt}: {e}")
    return ranker

# ------------------------------------------------------------------
# SELECTION LOGIC
# ------------------------------------------------------------------
def select_pannas_and_digits(scored: List[Tuple[str, float]], need_pannas: int, need_digits: int) -> Tuple[List[str], List[str]]:
    digit_best: Dict[str, Tuple[str, float]] = {}
    for p, score in scored:
        d = PANNA_TO_DIGIT[p]
        if d not in digit_best or score > digit_best[d][1]:
            digit_best[d] = (p, score)

    top_digits = [d for d, _ in sorted(digit_best.items(), key=lambda x: x[1][1], reverse=True)]
    if len(top_digits) < need_digits:
        missing = [d for d in map(str, range(10)) if d not in top_digits]
        top_digits.extend(missing)
    selected_digits = top_digits[:need_digits]

    selected_pannas = []
    seen = set()
    for p, _ in scored:
        if p in seen:
            continue
        if PANNA_TO_DIGIT[p] in selected_digits:
            selected_pannas.append(p)
            seen.add(p)
        if len(selected_pannas) >= need_pannas:
            break
    if len(selected_pannas) < need_pannas:
        for p, _ in scored:
            if p not in seen:
                selected_pannas.append(p)
                seen.add(p)
            if len(selected_pannas) >= need_pannas:
                break
    return selected_pannas, selected_digits

def build_jodis(open_digits: List[str], close_digits: List[str], ctx: Dict[str, Any]) -> List[str]:
    y_j = ctx.get("j_lag1", "")
    cut_jodi = ""
    if y_j and len(y_j) == 2:
        cut_jodi = f"{(int(y_j[0]) + 5) % 10}{(int(y_j[1]) + 5) % 10}"

    cands = []
    for a in open_digits:
        for b in close_digits:
            j = f"{a}{b}"
            w = 1.0
            if j == y_j: w *= 0.2
            if y_j and j == y_j[::-1]: w *= 1.5
            if (int(a) + int(b)) in (10, 20): w *= 1.35
            if cut_jodi and j == cut_jodi: w *= 1.4
            cands.append((j, w))

    cands.sort(key=lambda x: x[1], reverse=True)
    seen: Set[str] = set()
    out: List[str] = []
    for j, _ in cands:
        if j not in seen:
            seen.add(j)
            out.append(j)
        if len(out) >= CONFIG["NUM_JODIS"]:
            break
    return out

# ------------------------------------------------------------------
# STATE / BACKTEST
# ------------------------------------------------------------------
def load_state(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    if isinstance(data, dict) and "markets" in data:
        return data
    return {"markets": {}, "schema_version": 2}

def backtest_update(state: Dict[str, Any], all_hist: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    scored_counts: Dict[str, int] = {}
    markets = state.get("markets", {})
    for mkt, entries in markets.items():
        rows = all_hist.get(mkt, [])
        dated = [(parse_date(r.get("Date") or r.get("date")), extract_day_result(r)) for r in rows if parse_date(r.get("Date") or r.get("date"))]
        dated.sort(key=lambda x: x[0])
        by_date = {d: res for d, res in dated if d}

        n_scored = 0
        for entry in entries:
            if entry.get("actual") is not None:
                continue
            d = parse_date(entry.get("date"))
            if not d:
                continue
            actual = by_date.get(d)
            if not actual:
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

def rolling_accuracy(entries: List[Dict[str, Any]], lookback: int = 30) -> Optional[Dict[str, float]]:
    scored = [e for e in entries if e.get("hits") is not None]
    if len(scored) < 5:
        return None
    scored.sort(key=lambda e: e.get("date", ""), reverse=True)
    window = scored[:lookback]
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
    if any(e.get("date") == date_str for e in entries):
        return
    entries.append({"date": date_str, "predicted": predicted, "actual": None, "hits": None})
    if len(entries) > CONFIG["STATE_HISTORY_CAP_PER_MARKET"]:
        del entries[: len(entries) - CONFIG["STATE_HISTORY_CAP_PER_MARKET"]]

# ------------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------------
def predict_market(mkt: str, rows: List[Dict[str, Any]], asof: datetime.date,
                   state: Dict[str, Any], cache_dir: Path, clear_cache: bool) -> Dict[str, Any]:
    if len(rows) < CONFIG["MIN_HISTORY_FOR_ML"]:
        return {
            "OpenDigits": [], "CloseDigits": [], "OpenPannas": [], "ClosePannas": [], "Jodis": [],
            "confidence": 0.0, "confidence_basis": "insufficient_data"
        }

    ctx = PannaFeatureBuilder.build_context(rows, asof)
    ranker = get_cached_or_train_ranker(mkt, rows, cache_dir, clear_cache)

    open_scored = ranker.score_all(ctx, role="open")
    open_pannas, open_digits = select_pannas_and_digits(
        open_scored, CONFIG["NUM_OPEN_PANNAS"], CONFIG["NUM_OPEN_DIGITS"]
    )

    ctx["predicted_open_digits"] = open_digits
    close_scored = ranker.score_all(ctx, role="close")
    close_pannas, close_digits = select_pannas_and_digits(
        close_scored, CONFIG["NUM_CLOSE_PANNAS"], CONFIG["NUM_CLOSE_DIGITS"]
    )

    jodis = build_jodis(open_digits, close_digits, ctx)

    entries = state.get("markets", {}).get(mkt, [])
    accuracy = rolling_accuracy(entries, 30)

    top_open_score = open_scored[0][1] if open_scored else 0.0
    top_close_score = close_scored[0][1] if close_scored else 0.0
    raw_conf = CONFIG["CONFIDENCE_BASE"] + (top_open_score + top_close_score) / 2.0
    raw_conf = min(raw_conf, CONFIG["CONFIDENCE_CAP"])

    if accuracy is not None:
        conf = 0.5 * raw_conf + 0.5 * accuracy["overall"]
        basis = f"ml_panna + {accuracy['sample_size']}-day empirical accuracy"
    else:
        conf = raw_conf
        basis = "ml_panna_rules (not enough scored history)"

    conf = max(CONFIG["CONFIDENCE_MIN"], min(conf, CONFIG["CONFIDENCE_CAP"]))

    return {
        "OpenDigits": open_digits,
        "CloseDigits": close_digits,
        "OpenPannas": open_pannas,
        "ClosePannas": close_pannas,
        "Jodis": jodis,
        "confidence": round(float(conf), 2),
        "confidence_basis": basis,
    }

# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------
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
        markets_array.append(entry)

    return {
        "meta": {
            "generated_at": iso_timestamp(),
            "engine": "ml_panna_first_v2",
            "schema_version": 2,
            "backtest_newly_scored": scored_summary,
        },
        "markets": markets_array
    }

def run_once(paths: Dict[str, Path], for_date: Optional[datetime.date] = None,
             force_prod: bool = False, dry_run: bool = False,
             clear_cache: bool = False) -> Dict[str, Any]:
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

    # Cache directory lives next to state file
    cache_dir = paths["STATE_FILE"].parent / "ml_model_cache"
    safe_mkdir(cache_dir)

    all_hist = load_market_history(paths["HISTORY_FILE"])
    if not all_hist:
        raise RuntimeError(f"Missing history: {paths['HISTORY_FILE']}")
    validate_history_schema(all_hist)

    state = load_state(paths["STATE_FILE"])
    scored_summary = backtest_update(state, all_hist)
    if scored_summary:
        logger.info(f"Backtest scored new outcomes: {scored_summary}")

    internal_preds = {}
    date_str = for_date.isoformat()

    for mkt, rows in all_hist.items():
        try:
            internal_preds[mkt] = predict_market(mkt, rows, for_date, state, cache_dir, clear_cache)
            record_prediction(state, mkt, date_str, {
                "OpenDigits": internal_preds[mkt]["OpenDigits"],
                "CloseDigits": internal_preds[mkt]["CloseDigits"],
                "OpenPannas": internal_preds[mkt]["OpenPannas"],
                "ClosePannas": internal_preds[mkt]["ClosePannas"],
                "Jodis": internal_preds[mkt]["Jodis"],
            })
        except Exception as e:
            logger.warning(f"{mkt}: prediction error: {e}", exc_info=True)
            internal_preds[mkt] = {
                "OpenDigits": [], "CloseDigits": [], "OpenPannas": [],
                "ClosePannas": [], "Jodis": [],
                "confidence": 0.0, "confidence_basis": "error"
            }

    app_json = to_app_structure(internal_preds, for_date, scored_summary)
    tag = today_str(for_date)

    dated_out = paths["OUTPUT_DIR"] / f"predictions_{tag}_ml_panna.json"
    today_out = paths["TODAY_PROD_FILE"] if (force_prod or not cfg["TEST_MODE"]) else paths["TODAY_TEST_FILE"]
    assets_today = paths["TODAYS_PREDICTIONS_JSON"]
    assets_dated = paths["ASSETS_DIR"] / f"predictions_{tag}_ml_panna.json"

    saved_paths = []
    if not dry_run:
        save_json_atomic(dated_out, app_json)
        save_json_atomic(today_out, app_json)
        save_json_atomic(assets_today, app_json)
        save_json_atomic(assets_dated, app_json)
        save_json_atomic(paths["STATE_FILE"], state)
        saved_paths = [str(dated_out), str(today_out), str(assets_today), str(assets_dated), str(paths["STATE_FILE"])]
        logger.info(f"Saved {len(saved_paths)} files.")
    else:
        logger.info("DRY RUN: No files written.")

    logger.info(f"OK ml-panna-v2 — markets: {len(internal_preds)} | dated: {dated_out}")
    if not dry_run:
        print("\n" + "="*70)
        print("FILES SAVED:")
        for p in saved_paths:
            print(f"  {p}")
        print("="*70)
    return internal_preds

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SattaMatkaAI ML Panna-First Engine v2")
    ap.add_argument("--prod", action="store_true", help="Production paths.")
    ap.add_argument("--dry-run", action="store_true", help="Calculate but do NOT write files.")
    ap.add_argument("--clear-cache", action="store_true", help="Force retrain all models (ignore cache).")
    ap.add_argument("--verbose", action="store_true", help="Log to console.")
    ap.add_argument("--date", type=str, default=None, help="Override date YYYY-MM-DD.")

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
        run_once(paths, for_date=for_date, force_prod=args.prod, dry_run=args.dry_run, clear_cache=args.clear_cache)
        return 0
    except Exception as e:
        logger.error(f"Engine run failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())