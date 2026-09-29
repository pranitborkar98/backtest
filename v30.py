# -*- coding: utf-8 -*-
"""
prediction_engine_v20_optimized.py — SattaMatkaAI (Human-Bias & Adversarial Engine)

Optimizations:
• Stripped out static "frequency/trend conformity" models.
• Implements Adversarial Human-Bias Model built explicitly around operator behavior.
• 2-Day Fakeout Core: Breaks sequential matching/mirror trapping loops.
• Family/Cut-Digit Tracker: Dynamically hunts for targeted Digit ± 5 substitutions.
• Dynamic Twin (Red) Jodi Filters: Throttles back-to-back duplicate pair dumps.
• Algebraic Sum-Mod10 Cycle: Targets operator favoritism for Open + Close = 10 / 20.
• Structural Reverse Panel Mechanics: Leverages mirrored expectations for panel choices.
• Dynamic App Validation & Confidence Matrix output.
"""

import os, json, random, datetime, argparse, logging, logging.handlers
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Iterable, Optional

# ==========================================
# ================= CONFIG =================
# ==========================================
CONFIG = {
    # System Paths (Strict App-Mirror Alignment)
    "HISTORY_FILE": r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\sattaboss-data\data\all_markets_history.json",
    "DNA_FILE":     r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\market_dna.json",
    "YDAY_RESULTS_FILE": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\yesterday_results.json",

    "OUTPUT_DIR":  r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output",
    "STATE_FILE":  r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\state\engine_v10_state.json",
    "TODAY_TEST_FILE": r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\todays_predictions_v10_TEST.json",
    "TODAY_PROD_FILE": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\todays_predictions.json",
    "TODAYS_PREDICTIONS_JSON": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\todays_predictions.json",

    "LOG_FILE":    r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\engine_v10.log",
    "ASSETS_DIR":  r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets",

    "TEST_MODE": False,
    "TEST_SEED": 20251109,

    # UI / Pipeline Targets
    "NUM_OPEN_DIGITS": 3,
    "NUM_CLOSE_DIGITS": 3,
    "NUM_OPEN_PANNAS": 8,
    "NUM_CLOSE_PANNAS": 8,
    "NUM_JODIS": 9,

    # Engine Tuning
    "EXCLUDE_TRIPLE_PANNAS": True,
    "ENABLE_CONFIDENCE": True,
    
    # Bounded Psychological Windows
    "FREQ_LOOKBACK_DAYS": 45,   # Baseline range for non-dominant tracking
    "HERD_WINDOW_DAYS": 5,      # Hot digit evaluation window
    "HERD_HOT_THRESHOLD": 3,    # Appearance benchmark matching market traps
    "FAKEOUT_LOOKBACK": 2,      
    
    # Adversarial Scoring Profiles
    "BREAK_PENALTY": 0.15,      # Immediate flattening multiplier for explicit repeats
    "STREAK_PENALTY": 0.10,     # Multiplier for active bait numbers
    "SUMMOD_BOOST": 1.45,       # Complement cycle premium
    "NEIGHBOR_BOOST": 1.25,     # Proximity trap premium (+/- 1)
    "MIRROR_BOOST": 1.35,       # Reverse alignment premium
    "CUT_BOOST": 1.40,          # Family Cut Shift premium (Digit ± 5)
    "RED_JODI_PENALTY": 0.25,   # Throttles duplicate pairing back-to-back
    
    "PANNA_RECENT_PENALTY": 0.35,
    "PANNA_MIRROR_BONUS": 1.30,
    
    "CONFIDENCE_BASE": 0.35,
    "CONFIDENCE_CAP": 0.80,
    
    "ENABLE_LOGGING": True,
    "LOG_MAX_MB": 5,
    "LOG_BACKUPS": 3,
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
# ---- BACKTEST LAB path portability: remap Windows paths to this folder when absent ----
try:
    import os as _p_os
    if CONFIG.get("LAB_MODE") or (not _p_os.path.exists(_p_os.path.dirname(CONFIG["HISTORY_FILE"]))):
        _P_HERE = _p_os.path.dirname(_p_os.path.abspath(__file__))
        _P_MAP = {
            "HISTORY_FILE": "all_markets_history.json",
            "YDAY_RESULTS_FILE": "yesterday_results.json",
            "OUTPUT_DIR": "output",
            "STATE_FILE": "state/engine_v30_state.json",
            "TODAY_TEST_FILE": "output/todays_predictions_TEST.json",
            "TODAY_PROD_FILE": "output/todays_predictions.json",
            "TODAYS_PREDICTIONS_JSON": "output/todays_predictions.json",
            "DNA_FILE": "output/market_dna.json",
            "LOG_FILE": "output/engine.log",
            "ASSETS_DIR": "assets",
        }
        for _pk, _pf in _P_MAP.items():
            if _pk in CONFIG and isinstance(CONFIG[_pk], str):
                CONFIG[_pk] = _p_os.path.join(_P_HERE, _pf)
except Exception:
    pass

# ================ LOGGING =================
# ==========================================
logger = logging.getLogger("pred_engine_v20")
logger.setLevel(logging.INFO)
if CONFIG["ENABLE_LOGGING"]:
    try:
        os.makedirs(os.path.dirname(CONFIG["LOG_FILE"]), exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            CONFIG["LOG_FILE"], maxBytes=CONFIG["LOG_MAX_MB"] * 1024 * 1024,
            backupCount=CONFIG["LOG_BACKUPS"], encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logger.warning(f"Logging setup failed: {e}")
else:
    logging.basicConfig(level=logging.INFO)

# ==========================================
# ================ UTILITIES ===============
# ==========================================
def today_str(dt=None) -> str:
    if dt is None: dt = datetime.date.today()
    return dt.strftime("%Y%m%d")

def iso_timestamp() -> str:
    return datetime.datetime.now().isoformat()

def safe_mkdir(path) -> None:
    os.makedirs(path, exist_ok=True)

def load_json(p) -> Any:
    if not os.path.exists(p): return None
    with open(p, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except: return None

def save_json(p, obj) -> None:
    safe_mkdir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def parse_date(d) -> Optional[datetime.date]:
    if not d: return None
    s = str(d)
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try: return datetime.datetime.strptime(s, fmt).date()
        except: pass
    return None

def is_triple_panna(p) -> bool:
    return isinstance(p, str) and len(p) == 3 and p.isdigit() and len(set(p)) == 1

def mirror_jodi(j) -> str:
    return f"{j[1]}{j[0]}" if isinstance(j, str) and len(j) == 2 else j

def mirror_panna(p) -> str:
    return p[::-1] if isinstance(p, str) and len(p) == 3 else p

def get_field(rec, *keys) -> str:
    for k in keys:
        if k in rec: return str(rec[k] or "").strip()
    return ""

def last_n_days(rows, asof, n) -> List[Dict[str, Any]]:
    dated = []
    for r in rows:
        d = parse_date(r.get("Date") or r.get("date"))
        if d and d <= asof:
            dated.append((d, r))
    dated.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in dated[:n]]

# ==========================================
# ============== DATA INGEST ===============
# ==========================================
def load_market_history(path) -> Dict[str, List[Dict[str, Any]]]:
    data = load_json(path)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, list)}
    if isinstance(data, list):
        grouped = defaultdict(list)
        for rec in data:
            m = rec.get("Market") or rec.get("market") or "UNKNOWN"
            grouped[str(m)].append(rec)
        return dict(grouped)
    return {}

def validate_history_schema(all_hist) -> None:
    if not isinstance(all_hist, dict) or not all_hist:
        raise RuntimeError("Historical data structure unreadable/empty.")
    check_limit = 0
    for mkt, rows in all_hist.items():
        if not isinstance(rows, list): raise TypeError(f"{mkt}: Row structure must be sequential list.")
        for r in rows:
            if not isinstance(r, dict): raise TypeError(f"{mkt}: Entry block must be dict.")
            if not any(k in r for k in ("OpenPanna", "open_panna", "ClosePanna", "close_panna", "Jodi", "jodi", "Date", "date")):
                raise ValueError(f"{mkt}: Data attributes failed schema matching.")
        check_limit += 1
        if check_limit >= 3: break

# ==========================================
# ========== HUMAN BIAS CORE ===============
# ==========================================
def build_base_freq(rows, asof, n_days) -> Dict[str, float]:
    """Builds a dampened base frequency mapping to isolate and limit long-term dominance."""
    last = last_n_days(rows, asof, n_days)
    cnt = Counter()
    for r in last:
        od = get_field(r, "Open Digit", "OpenDigit", "open_digit")
        cd = get_field(r, "Close Digit", "CloseDigit", "close_digit")
        if od.isdigit(): cnt[od] += 1
        if cd.isdigit(): cnt[cd] += 1
    base = {str(i): (cnt.get(str(i), 0) + 1.0) * 0.1 for i in range(10)}
    return base

def get_yesterday_context(rows, asof) -> Dict[str, str]:
    last = last_n_days(rows, asof, 1)
    if not last: return {}
    r = last[0]
    y_od = get_field(r, "Open Digit", "OpenDigit", "open_digit")
    y_cd = get_field(r, "Close Digit", "CloseDigit", "close_digit")
    y_j  = get_field(r, "Jodi", "jodi")
    y_op = get_field(r, "Open Panna", "OpenPanna", "open_panna")
    y_cp = get_field(r, "Close Panna", "ClosePanna", "close_panna")
    
    # Reconstruct jodi formatting safely if raw field fell back to int/null padding
    if not y_j and y_od.isdigit() and y_cd.isdigit():
        y_j = f"{y_od}{y_cd}"
    return {"od": y_od, "cd": y_cd, "j": y_j, "op": y_op, "cp": y_cp}

def detect_2day_fakeout(rows, asof) -> Dict[str, Any]:
    """Detects sequential loop duplication and mirror trapping indicators."""
    last2 = last_n_days(rows, asof, 2)
    if len(last2) < 2: return {"active": False, "jodi_to_break": None}
    r0, r1 = last2[0], last2[1]
    j0 = get_field(r0, "Jodi", "jodi")
    j1 = get_field(r1, "Jodi", "jodi")
    op0 = get_field(r0, "Open Panna", "OpenPanna", "open_panna")
    op1 = get_field(r1, "Open Panna", "OpenPanna", "open_panna")
    
    identical = (j0 == j1 and op0 == op1 and j0 != "")
    mirror = (j0 and j1 and j0 == mirror_jodi(j1))
    return {
        "active": identical or mirror,
        "jodi_to_break": j0 if identical else (mirror_jodi(j1) if mirror else None)
    }

def get_herd_digits(rows, asof, n_days, thr) -> set:
    """Isolates targeted over-played herd digits to trigger streak blocks."""
    last = last_n_days(rows, asof, n_days)
    cnt = Counter()
    for r in last:
        od = get_field(r, "Open Digit", "OpenDigit", "open_digit")
        cd = get_field(r, "Close Digit", "CloseDigit", "close_digit")
        if od.isdigit(): cnt[od] += 1
        if cd.isdigit(): cnt[cd] += 1
    return {d for d, c in cnt.items() if c >= thr}

def recent_pannas(rows, asof, n_days) -> set:
    last = last_n_days(rows, asof, n_days)
    out = set()
    for r in last:
        for k in ("Open Panna", "Close Panna", "OpenPanna", "ClosePanna", "open_panna", "close_panna"):
            p = get_field(r, k)
            if len(p) == 3 and p.isdigit(): out.add(p)
    return out

def weighted_choice(items, weights, k) -> List[Any]:
    items = list(items); weights = list(weights)
    picks = []
    for _ in range(min(k, len(items))):
        s = sum(weights)
        if s <= 0: break
        r, acc = random.random() * s, 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                picks.append(items.pop(i)); weights.pop(i); break
    return picks

# ==========================================
# ============ PREDICTION LOGIC ============
# ==========================================
def compose_digits_bias(base_freq, rows, asof, role, yctx, fakeout) -> List[str]:
    """Calculates tactical adversarial weights to bypass operator layout traps."""
    weights = dict(base_freq)
    herd = get_herd_digits(rows, asof, CONFIG["HERD_WINDOW_DAYS"], CONFIG["HERD_HOT_THRESHOLD"])
    
    y_od = yctx.get("od", "")
    y_cd = yctx.get("cd", "")
    y_j  = yctx.get("j", "")
    
    target_y = y_od if role == "open" else y_cd
    other_y  = y_cd if role == "open" else y_od
    
    # 1. Anti-Herd Streak Neutralizer
    for d in herd:
        weights[d] *= CONFIG["STREAK_PENALTY"]
        
    # 2. Sequential Break Penalty (Avoid direct yesterday matches)
    if target_y and target_y.isdigit():
        weights[target_y] *= CONFIG["BREAK_PENALTY"]
        
    # 3. Cut-Digit Shift Tracker (Countering family value mutations: Digit ± 5)
    if target_y and target_y.isdigit():
        cut_digit = str((int(target_y) + 5) % 10)
        weights[cut_digit] *= CONFIG["CUT_BOOST"]
        
    # 4. Sum-Mod10 Cycle Alignment (Premium for Cut-complement totals)
    if other_y and other_y.isdigit():
        comp = str((10 - int(other_y)) % 10)
        weights[comp] *= CONFIG["SUMMOD_BOOST"]
        
    # 5. Proximity / Neighbor Trap Mechanics (+/-1 Offset)
    if other_y and other_y.isdigit():
        for nb in [(int(other_y) + 1) % 10, (int(other_y) - 1) % 10]:
            weights[str(nb)] *= CONFIG["NEIGHBOR_BOOST"]
            
    # 6. Mirror Flipping Mitigation
    if y_j and len(y_j) == 2:
        weights[y_j[1] if role == "open" else y_j[0]] *= CONFIG["MIRROR_BOOST"]

    # Selection & Verification execution
    items = list(weights.keys())
    ws = [weights[k] for k in items]
    picks = weighted_choice(items, ws, CONFIG["NUM_OPEN_DIGITS"] if role == "open" else CONFIG["NUM_CLOSE_DIGITS"])
    
    out, seen = [], set()
    for p in picks:
        if p not in seen:
            seen.add(p); out.append(p)
        if len(out) >= 3: break
    while len(out) < 3:
        for k in items:
            if k not in seen:
                seen.add(k); out.append(k); break
    return out[:3]

def pannas_pool_for_digits(digs) -> List[str]:
    raw = []
    for d in digs:
        try: raw += VALID_PANNAS_BY_LAST_DIGIT.get(int(d), [])
        except: continue
    seen, out = set(), []
    for p in raw:
        if p not in seen:
            seen.add(p); out.append(p)
    if CONFIG["EXCLUDE_TRIPLE_PANNAS"]:
        out = [p for p in out if not is_triple_panna(p)]
    return out

def sample_pannas_bias(candidate_pool, need_k, rows, asof) -> List[str]:
    if not candidate_pool: return []
    recent = recent_pannas(rows, asof, 5)
    items = list(candidate_pool)
    ws = []
    for p in items:
        w = 1.0
        if p in recent:
            w *= CONFIG["PANNA_RECENT_PENALTY"]
        mp = mirror_panna(p)
        if mp in recent and mp != p:
            w *= CONFIG["PANNA_MIRROR_BONUS"]
        ws.append(max(w, 1e-6))
        
    picks = weighted_choice(items, ws, need_k)
    out, seen = [], set()
    for p in picks:
        if p not in seen:
            seen.add(p); out.append(p)
        if len(out) >= need_k: break
    while len(out) < need_k:
        for p in items:
            if p not in seen:
                seen.add(p); out.append(p); break
    return out[:need_k]

def build_jodis_bias(open_d, close_d, yctx, fakeout) -> List[str]:
    """Constructs explicit Jodi combinations optimized via dynamic structural filters."""
    cands = []
    y_j = yctx.get("j", "")
    
    for a in open_d:
        for b in close_d:
            j = f"{a}{b}"
            w = 1.0
            
            # Direct Sequence Block
            if j == y_j: w *= 0.2
            
            # Twin / Red Pair Dynamic Structural Suppression Filter
            if a == b:
                if y_j and y_j[0] == y_j[1]:
                    w *= CONFIG["RED_JODI_PENALTY"]
                else:
                    w *= 1.15 # System default tracking for hidden twin variants
            
            # Loop Reset
            if fakeout.get("active") and j == fakeout.get("jodi_to_break"):
                w *= 0.1
                
            # Mirror Trapping Strategy Boost
            if y_j and j == mirror_jodi(y_j):
                w *= 1.5
                
            # Algebraic Cut Modulation validation
            if (int(a) + int(b)) in (10, 20):
                w *= 1.35
                
            cands.append((j, w))
            
    cands.sort(key=lambda x: x[1], reverse=True)
    out, seen = [], set()
    for j, _ in cands:
        if j not in seen:
            seen.add(j); out.append(j)
        if len(out) >= CONFIG["NUM_JODIS"]: break
    return out

def predict_market(mkt, rows, asof) -> Dict[str, Any]:
    if len(rows) < 5: 
        return {"OpenDigits": [], "CloseDigits": [], "OpenPannas": [], "ClosePannas": [], "Jodis": [], "confidence": 0.0}
        
    base_freq = build_base_freq(rows, asof, CONFIG["FREQ_LOOKBACK_DAYS"])
    yctx = get_yesterday_context(rows, asof)
    fakeout = detect_2day_fakeout(rows, asof)
    
    open_d  = compose_digits_bias(base_freq, rows, asof, "open", yctx, fakeout)
    close_d = compose_digits_bias(base_freq, rows, asof, "close", yctx, fakeout)
    
    open_pool  = pannas_pool_for_digits(open_d)
    close_pool = pannas_pool_for_digits(close_d)
    open_p     = sample_pannas_bias(open_pool, CONFIG["NUM_OPEN_PANNAS"], rows, asof)
    close_p    = sample_pannas_bias(close_pool, CONFIG["NUM_CLOSE_PANNAS"], rows, asof)
    
    jodis = build_jodis_bias(open_d, close_d, yctx, fakeout)
    
    # Metrics Verification / Mathematical Confidence Matrix Assessment
    conf = CONFIG["CONFIDENCE_BASE"]
    if fakeout.get("active"): conf += 0.15
    if yctx.get("j") and mirror_jodi(yctx["j"]) in jodis: conf += 0.15
    if any((int(a) + int(b)) % 10 == 0 for a in open_d for b in close_d if a != b): conf += 0.10
    conf = min(conf, CONFIG["CONFIDENCE_CAP"])
    
    return {
        "OpenDigits": open_d,
        "CloseDigits": close_d,
        "OpenPannas": open_p,
        "ClosePannas": close_p,
        "Jodis": jodis,
        "confidence": round(conf, 2)
    }

# ==========================================
# =================  I/O ===================
# ==========================================
def to_app_structure(preds, run_date) -> Dict[str, Any]:
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
            "confidence": pr.get("confidence", 0.0)
        })
    return {
        "meta": { "generated_at": iso_timestamp(), "engine": "v20_Optimal_Bias" },
        "markets": markets_array
    }

def run_once(for_date=None, force_prod=False, dry_run=False) -> Dict[str, Any]:
    cfg = CONFIG
    if (not force_prod) and cfg["TEST_MODE"]:
        random.seed(cfg.get("TEST_SEED", 12345))
    else:
        random.seed()

    if for_date is None: for_date = datetime.date.today()

    safe_mkdir(cfg["OUTPUT_DIR"])
    safe_mkdir(os.path.dirname(cfg["STATE_FILE"]))
    safe_mkdir(cfg["ASSETS_DIR"])

    all_hist = load_market_history(cfg["HISTORY_FILE"])
    if not all_hist: raise RuntimeError(f"Source history mapping trace unreadable at: {cfg['HISTORY_FILE']}")
    validate_history_schema(all_hist)

    internal_preds = {}
    for mkt, rows in all_hist.items():
        try:
            internal_preds[mkt] = predict_market(mkt, rows, for_date)
        except Exception as e:
            logger.warning(f"{mkt}: Processing block execution interruption: {e}")
            internal_preds[mkt] = {"OpenDigits": [], "CloseDigits": [], "OpenPannas": [], "ClosePannas": [], "Jodis": [], "confidence": 0.0}

    app_json = to_app_structure(internal_preds, for_date)
    tag = today_str(for_date)
    
    dated_out    = os.path.join(cfg["OUTPUT_DIR"], f"predictions_{tag}_v10.json")
    today_out    = cfg["TODAY_PROD_FILE"] if (force_prod or not cfg["TEST_MODE"]) else cfg["TODAY_TEST_FILE"]
    assets_today = cfg["TODAYS_PREDICTIONS_JSON"]
    assets_dated = os.path.join(cfg["ASSETS_DIR"], f"predictions_{tag}_v10.json")

    if not dry_run:
        save_json(dated_out, app_json)
        save_json(today_out, app_json)
        save_json(assets_today, app_json)
        save_json(assets_dated, app_json)

    logger.info(f"Execution Successful [v20-Optimal-Bias] — Synced Markets: {len(internal_preds)} | Output Track: {dated_out}")
    return internal_preds

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SattaMatkaAI Mathematical Adversarial Engine (v20 Optimized).")
    ap.add_argument("--prod", action="store_true", help="Force production deployment tracks.")
    ap.add_argument("--date", type=str, default=None, help="Override run date YYYY-MM-DD (backtesting).")
    ap.add_argument("--dry-run", action="store_true", help="Execute calculations without performing filesystem writes.")
    args = ap.parse_args()
    run_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date() if getattr(args, "date", None) else None
    run_once(for_date=run_date, force_prod=args.prod, dry_run=args.dry_run)

# ==== BACKTEST LAB 1-LINER: unique dated output -> D:\backtest\predictions_<YYYY-MM-DD>_v30.json ====
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
        _p = _bt_dir / f"predictions_{ds}_v30.json"
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
