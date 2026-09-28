# -*- coding: utf-8 -*-
"""
prediction_engine_v10.py — SattaMatkaAI (DNA Adaptive Engine)

What’s new vs v9:
• Reads per-market DNA (repeat/mirror/volatility/DoW) from output/market_dna.json
• Per-market adaptive strategy mix (REPEAT/TRAP/FAKE) + penalties
• Protected repeat slots (digits/jodi/pannas) when market DNA says repeats are common
• DoW nudges and streak-aware soft caps on over-repetition
• Same output structure your app expects (meta + markets[] with spaced keys)
"""

import os, json, random, datetime, argparse, logging, logging.handlers
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Iterable, Optional

# =========================
# ========= CONFIG ========
# =========================
CONFIG = {
    # PATHS (align with your setup)
    "HISTORY_FILE": r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\sattaboss-data\data\all_markets_history.json",
    "DNA_FILE":     r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\market_dna.json",
    "YDAY_RESULTS_FILE": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\yesterday_results.json",

    "OUTPUT_DIR":  r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output",
    "STATE_FILE":  r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\state\engine_v10_state.json",
    "TODAY_TEST_FILE": r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\todays_predictions_v10_TEST.json",
    "TODAY_PROD_FILE": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\todays_predictions.json",
    "TODAYS_PREDICTIONS_JSON": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\todays_predictions.json",

    "LOG_FILE":    r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\engine_v10.log",

    # Mirror to Android app assets
    "ASSETS_DIR":  r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets",

    # ENGINE TOGGLES
    "TEST_MODE": False,
    "TEST_SEED": 20251109,

    # Counts
    "NUM_OPEN_DIGITS": 3,
    "NUM_CLOSE_DIGITS": 3,
    "NUM_OPEN_PANNAS": 8,
    "NUM_CLOSE_PANNAS": 8,
    "NUM_JODIS": 9,

    # Base weights (will be adapted per market via DNA)
    "BASE_STRATEGY": {"REPEAT": 0.25, "TRAP": 0.45, "FAKE": 0.20},  # remainder NEUTRAL
    "BASE_PENALTIES": {
        "YDAY_DIGIT_PENALTY": 0.25,
        "YDAY_JODI_PENALTY":  0.30,
        "YDAY_PANNA_PENALTY": 0.28,
        "TRAP_BONUS": 0.20,  # bonus for mirror/±1 near-miss
    },

    # Pannas
    "DOUBLE_PANNA_WEIGHT": 0.50,
    "SINGLE_PANNA_WEIGHT": 1.00,
    "EXCLUDE_TRIPLE_PANNAS": True,
    "ENFORCE_SUMMOD_DIGIT": False,

    # Recency weighting
    "MIN_HISTORY_ROWS_FOR_MARKET": 8,
    "RECENT_STRONG_DAYS": 4,
    "RECENT_MEDIUM_DAYS": 8,
    "WEIGHT_RECENT": 1.0,
    "WEIGHT_MEDIUM": 0.6,
    "WEIGHT_OLD": 0.3,

    # Protected repeat slots (enabled when DNA says repeats are common)
    "PROTECT_DIGIT_SLOTS": 1,   # 0/1
    "PROTECT_JODI_SLOTS":  1,   # 0/1
    "PROTECT_PANNA_SLOTS": 1,   # 0/1-2

    # DNA thresholds (tune if needed)
    "DNA_THRESH": {
        "HIGH_OD_REPEAT": 0.28,      # if open_digit repeat rate ≥ this → repeat-friendly
        "HIGH_CD_REPEAT": 0.28,
        "HIGH_J_REPEAT":  0.08,      # exact jodi repeat
        "HIGH_J_MIRROR":  0.12,      # mirror jodi rate
        "HIGH_DOUBLE_SHARE": 0.55,   # prefers doubles
        "HIGH_VOLATILITY": 0.70,     # normalized entropy proxy
    },

    # Streak soft cap
    "HARD_STREAK_CAP": 3,   # avoid pushing 3+ day exact repeats unless DNA suggests it

    # Logging
    "ENABLE_LOGGING": True,
    "LOG_MAX_MB": 5,
    "LOG_BACKUPS": 3,
}

# =========================
# ====== FIXED PANNAS =====
# =========================
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

# =========================
# ========= LOGGING =======
# =========================
logger = logging.getLogger("pred_engine_v10")
logger.setLevel(logging.INFO)
if CONFIG["ENABLE_LOGGING"]:
    try:
        os.makedirs(os.path.dirname(CONFIG["LOG_FILE"]), exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            CONFIG["LOG_FILE"],
            maxBytes=CONFIG["LOG_MAX_MB"] * 1024 * 1024,
            backupCount=CONFIG["LOG_BACKUPS"],
            encoding="utf-8"
        )
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception as e:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logger.warning(f"Log file handler error: {e}")
else:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# =========================
# ======== UTILITIES ======
# =========================
def today_str(dt: Optional[datetime.date]=None) -> str:
    if dt is None: dt = datetime.date.today()
    return dt.strftime("%Y%m%d")

def iso_timestamp() -> str:
    return datetime.datetime.now().isoformat()

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_json(p: str) -> Any:
    if not os.path.exists(p): return None
    with open(p, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return None

def save_json(p: str, obj: Any) -> None:
    safe_mkdir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def parse_date(d: Any) -> Optional[datetime.date]:
    if not d: return None
    s = str(d)
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None

def recency_weight(delta_days: int) -> float:
    if delta_days < 0: delta_days = 0
    if delta_days < CONFIG["RECENT_STRONG_DAYS"]: return CONFIG["WEIGHT_RECENT"]
    if delta_days < CONFIG["RECENT_MEDIUM_DAYS"]: return CONFIG["WEIGHT_MEDIUM"]
    return CONFIG["WEIGHT_OLD"]

def is_triple_panna(p: str) -> bool:
    return isinstance(p, str) and len(p)==3 and p.isdigit() and len(set(p))==1

def panna_weight(p: str) -> float:
    # double pannas get different base weight (single pannas 3 distinct digits)
    if not p or len(p)!=3 or not p.isdigit(): return 1.0
    return CONFIG["DOUBLE_PANNA_WEIGHT"] if len(set(p))==2 else CONFIG["SINGLE_PANNA_WEIGHT"]

def sum_mod10(p: str) -> int:
    return (int(p[0])+int(p[1])+int(p[2])) % 10

def neighbors_digit(d: int) -> List[int]:
    x = int(d) % 10
    return [(x+1)%10, (x-1)%10]

def mirror_jodi(j: str) -> str:
    return f"{j[1]}{j[0]}" if isinstance(j,str) and len(j)==2 else j

# =========================
# ====== DATA INGEST ======
# =========================
def load_market_history(path: str) -> Dict[str, List[Dict[str, Any]]]:
    data = load_json(path)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, list)}
    if isinstance(data, list):
        grouped = defaultdict(list)
        for rec in data:
            m = rec.get("Market") or rec.get("market") or rec.get("market_name") or "UNKNOWN"
            grouped[str(m)].append(rec)
        return dict(grouped)
    return {}

def validate_history_schema(all_hist: Dict[str, List[Dict[str, Any]]]) -> None:
    if not isinstance(all_hist, dict) or not all_hist:
        raise RuntimeError("History file missing or not a dict/list → dict.")
    take = 0
    for mkt, rows in all_hist.items():
        if not isinstance(rows, list): raise TypeError(f"{mkt}: history must be list")
        for r in rows:
            if not isinstance(r, dict): raise TypeError(f"{mkt}: row must be dict")
            if not any(k in r for k in ("OpenPanna","open_panna","ClosePanna","close_panna","Jodi","jodi","date","Date")):
                raise ValueError(f"{mkt}: missing expected keys")
        take += 1
        if take>=3: break

# =========================
# === FREQUENCY BUILDERS ==
# =========================
def build_digit_freq_for_market(rows: List[Dict[str, Any]], asof: datetime.date) -> Dict[str, float]:
    freq = Counter()
    for rec in rows:
        d = parse_date(rec.get("date") or rec.get("Date"))
        w = recency_weight((asof - d).days) if d else 0.5

        for k in ("OpenPanna","ClosePanna","open_panna","close_panna"):
            p = rec.get(k)
            if isinstance(p, str) and p.isdigit() and len(p)==3:
                for ch in p:
                    freq[ch] += w

        j = rec.get("Jodi") or rec.get("jodi")
        if isinstance(j, str) and j.isdigit() and len(j)==2:
            freq[j[0]] += 0.7*w
            freq[j[1]] += 0.7*w

        for dk in ("OpenDigit","CloseDigit","open_digit","close_digit"):
            dg = rec.get(dk)
            s = str(dg) if dg is not None else None
            if s and s.isdigit() and len(s)==1:
                freq[s] += 0.4*w

    base = {str(i): freq.get(str(i), 0.0) + 1.0 for i in range(10)}
    tot = sum(base.values()) or 1.0
    return {k: base[k]/tot for k in base}

def build_global_digit_freq(all_hist: Dict[str, List[Dict[str, Any]]], asof: datetime.date) -> Dict[str, float]:
    agg = Counter()
    for _, rows in all_hist.items():
        local = build_digit_freq_for_market(rows, asof)
        for k, v in local.items():
            agg[k] += v
    tot = sum(agg.values()) or 1.0
    return {k: agg.get(k, 0.0)/tot for k in map(str, range(10))}

# =========================
# ======= DNA ADAPT =======
# =========================
def load_dna(path: str) -> Dict[str, Any]:
    data = load_json(path) or {}
    if isinstance(data, dict) and "markets" in data and isinstance(data["markets"], dict):
        return data["markets"]
    # if already a dict of market→dna
    return data if isinstance(data, dict) else {}

def dow_index(dt: datetime.date) -> int:
    # Monday=0..Sunday=6
    return dt.weekday()

def adapt_params_from_dna(market: str, dna: Dict[str, Any], asof: datetime.date) -> Dict[str, Any]:
    """Return per-market adapted parameters (strategy mix, penalties, protected slots, pannas weights)."""
    base_mix = dict(CONFIG["BASE_STRATEGY"])
    pen = dict(CONFIG["BASE_PENALTIES"])
    dthr = CONFIG["DNA_THRESH"]

    # defaults if no DNA
    if not isinstance(dna, dict) or dna.get("count", 0) < 4:
        return {
            "mix": base_mix,
            "pen": pen,
            "protect": {"digit": 0, "jodi": 0, "panna": 0},
            "double_weight": CONFIG["DOUBLE_PANNA_WEIGHT"],
            "trap_bonus": pen["TRAP_BONUS"],
        }

    rr   = dna.get("repeat_rates", {}) or {}
    pv   = dna.get("volatility", {}) or {}
    panna= dna.get("panna", {}) or {}
    dowb = dna.get("dow_bias", {}) or {}
    # repeat friendliness flags
    repeat_friendly = (
        (rr.get("open_digit",0.0)  >= dthr["HIGH_OD_REPEAT"]) or
        (rr.get("close_digit",0.0) >= dthr["HIGH_CD_REPEAT"]) or
        (rr.get("jodi_exact",0.0)  >= dthr["HIGH_J_REPEAT"])
    )
    mirror_heavy = rr.get("jodi_mirror",0.0) >= dthr["HIGH_J_MIRROR"]
    volatile = max(pv.get("open_digit_volatility",0.0), pv.get("jodi_volatility",0.0)) >= dthr["HIGH_VOLATILITY"]
    double_pref = panna.get("double_share",0.0) >= dthr["HIGH_DOUBLE_SHARE"]

    # DoW lift
    di = dow_index(asof)
    dow_lift = 0.0
    try:
        dow_lift = 0.25*(
            (dowb.get("od_repeat",{}).get(str(di),0.0) +
             dowb.get("cd_repeat",{}).get(str(di),0.0) +
             dowb.get("j_repeat",{}).get(str(di),0.0)) / 3.0
        )
    except Exception:
        dow_lift = 0.0

    # Start from base mix
    mix = dict(base_mix)

    # Adjust by DNA
    if volatile:
        mix["FAKE"]  = min(0.50, mix["FAKE"] + 0.15)
        mix["REPEAT"]= max(0.10, mix["REPEAT"] - 0.10)
    if repeat_friendly:
        mix["REPEAT"]= min(0.50, mix["REPEAT"] + 0.12 + dow_lift*0.5)
        pen["YDAY_DIGIT_PENALTY"] = max(0.12, pen["YDAY_DIGIT_PENALTY"] - 0.08)
        pen["YDAY_JODI_PENALTY"]  = max(0.15, pen["YDAY_JODI_PENALTY"]  - 0.08)
        pen["YDAY_PANNA_PENALTY"] = max(0.16, pen["YDAY_PANNA_PENALTY"] - 0.06)
    if mirror_heavy:
        mix["TRAP"]  = min(0.60, mix["TRAP"] + 0.12)
        pen["TRAP_BONUS"] = pen["TRAP_BONUS"] + 0.10

    # Renormalize (leave NEUTRAL as the remainder)
    alloc = sum(mix.values())
    if alloc > 0.95:
        scale = 0.95/alloc
        for k in mix: mix[k] *= scale

    # Protected slots
    prot = {"digit": 0, "jodi": 0, "panna": 0}
    if repeat_friendly:
        prot["digit"] = CONFIG["PROTECT_DIGIT_SLOTS"]
        prot["jodi"]  = CONFIG["PROTECT_JODI_SLOTS"]
        prot["panna"] = CONFIG["PROTECT_PANNA_SLOTS"]

    # Panna double weight
    dweight = CONFIG["DOUBLE_PANNA_WEIGHT"]
    if double_pref:
        dweight = min(0.70, dweight + 0.10)  # tilt toward doubles

    return {
        "mix": mix,
        "pen": pen,
        "protect": prot,
        "double_weight": dweight,
        "trap_bonus": pen["TRAP_BONUS"],
    }

# =========================
# ====== Y-DAY LEARN ======
# =========================
def normalize_yday_results(data: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        conv = {}
        for e in data:
            m = e.get("Market") or e.get("market") or "UNKNOWN"
            conv[m] = {k:v for k,v in e.items() if str(k).lower()!="market"}
        return conv
    return {}

def learn_from_yesterday(asof: datetime.date, all_hist: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    yraw = load_json(CONFIG["YDAY_RESULTS_FILE"]) or {}
    yact = normalize_yday_results(yraw)
    adj = {"last": {}}
    for market in all_hist.keys():
        act = yact.get(market, {})
        op = str(act.get("Open Panna") or act.get("OpenPanna") or "")
        cp = str(act.get("Close Panna") or act.get("ClosePanna") or "")
        jd = str(act.get("Jodi") or "")
        od = op[0] if len(op)==3 and op.isdigit() else ""
        cd = cp[-1] if len(cp)==3 and cp.isdigit() else ""
        adj["last"][market] = {"op": op, "cp": cp, "j": jd, "od": od, "cd": cd}
    return adj

# =========================
# ===== STRATEGY CORE =====
# =========================
def split_counts(total: int, weights_dict: Dict[str, float]) -> Dict[str, int]:
    parts, alloc = {}, 0
    for k, w in weights_dict.items():
        c = int(round(total * max(0.0, min(1.0, float(w)))))
        parts[k] = c; alloc += c
    parts["NEUTRAL"] = max(0, total - alloc)
    cur = sum(parts.values())
    if cur != total:
        parts["NEUTRAL"] += (total - cur)
    return parts

def weighted_choice(items: List[Any], weights: List[float], k: int) -> List[Any]:
    items = list(items); weights = list(weights)
    picks = []
    for _ in range(min(k, len(items))):
        s = sum(weights)
        if s <= 0: break
        r, acc = random.random()*s, 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                picks.append(items.pop(i)); weights.pop(i); break
    return picks

def compose_digits(freq_map: Dict[str, float], want_n: int, dna_mix: Dict[str, float],
                   trap_bonus: float, y_digits: List[str], protected: List[str]) -> List[str]:
    parts = split_counts(want_n, dna_mix)
    chosen, forbid = [], set(protected)

    # REPEAT (biased to recent strong digits)
    if parts["REPEAT"] > 0:
        items = list(freq_map.keys())
        weights = [(freq_map[k]**1.15) * (0.85 if k in y_digits else 1.0) for k in items]
        r = weighted_choice(items, weights, parts["REPEAT"])
        r = [x for x in r if x not in forbid]
        chosen += r; forbid.update(r)

    # TRAP (neighbors of yesterday digits)
    if parts["TRAP"] > 0 and y_digits:
        boost = set()
        for yd in y_digits:
            if yd.isdigit():
                for nb in neighbors_digit(int(yd)):
                    boost.add(str(nb))
        items = list(freq_map.keys())
        weights = [freq_map[k]*(1.0+trap_bonus if k in boost else 1.0) for k in items]
        t = weighted_choice(items, weights, parts["TRAP"])
        t = [x for x in t if x not in forbid]
        chosen += t; forbid.update(t)

    # FAKE (anti-popular within freq_map)
    if parts["FAKE"] > 0:
        items = list(freq_map.keys())
        base = [freq_map[k] for k in items]
        mx = max(base) if base else 1.0
        weights = [(mx - b + 1e-6) for b in base]
        f = weighted_choice(items, weights, parts["FAKE"])
        f = [x for x in f if x not in forbid]
        chosen += f; forbid.update(f)

    # NEUTRAL (proportional)
    if parts["NEUTRAL"] > 0:
        items = list(freq_map.keys())
        weights = [freq_map[k]*(0.85 if k in y_digits else 1.0) for k in items]
        pool = [(it,w) for it,w in zip(items,weights) if it not in forbid]
        if pool:
            ii, ww = zip(*pool)
            n = weighted_choice(list(ii), list(ww), parts["NEUTRAL"])
            chosen += n

    # Uniq + Fill
    out, seen = [], set()
    for d in protected + chosen:
        if d and d not in seen:
            seen.add(d); out.append(d)
        if len(out) >= want_n: break
    if len(out) < want_n:
        items = list(freq_map.keys())
        weights = [freq_map[k] for k in items]
        pool = [(it,w) for it,w in zip(items,weights) if it not in seen]
        if pool:
            ii, ww = zip(*pool)
            filler = weighted_choice(list(ii), list(ww), want_n-len(out))
            out += filler
    return out[:want_n]

def pannas_pool_for_digits(digs: List[str]) -> List[str]:
    raw = []
    for d in digs:
        try:
            raw += VALID_PANNAS_BY_LAST_DIGIT.get(int(d), [])
        except Exception:
            continue
    seen, out = set(), []
    for p in raw:
        if p not in seen:
            seen.add(p); out.append(p)
    if CONFIG["EXCLUDE_TRIPLE_PANNAS"]:
        out = [p for p in out if not is_triple_panna(p)]
    return out

def sample_pannas(candidate_pool: List[str], need_k: int, y_pannas_set: Iterable[str], double_weight: float) -> List[str]:
    if not candidate_pool:
        return []
    items = list(candidate_pool)
    base_w = []
    for p in items:
        w = (double_weight if len(set(p))==2 else CONFIG["SINGLE_PANNA_WEIGHT"])
        if y_pannas_set and p in y_pannas_set:
            w *= 0.75
        base_w.append(max(w, 1e-6))

    # simple proportional + slight anti-repeat diversity
    picks = weighted_choice(items, base_w, need_k)
    # Fill if duplicates (shouldn’t happen)
    seen, out = set(), []
    for p in picks:
        if p not in seen:
            seen.add(p); out.append(p)
        if len(out) >= need_k: break
    if len(out) < need_k:
        rest = [(it, base_w[items.index(it)]) for it in items if it not in seen]
        if rest:
            its, ws = zip(*rest)
            fill = weighted_choice(list(its), list(ws), need_k-len(out))
            out += fill
    return out[:need_k]

def build_jodis_grid(open_d: List[str], close_d: List[str],
                     jodi_hist_freq: Counter, y_jodi: Optional[str],
                     trap_bonus: float, protected: List[str]) -> List[str]:
    cands = []
    prot_set = set(protected or [])
    for a in open_d:
        for b in close_d:
            j = f"{a}{b}"
            w = jodi_hist_freq.get(j, 0.0) + 1.0
            if j[0] != j[1]:
                w *= 1.05
            if y_jodi and j == y_jodi:
                w *= 0.75  # mild penalty, DNA already adjusted elsewhere
            # TRAP bonus for mirror/±1 of yesterday
            if y_jodi:
                if j == mirror_jodi(y_jodi): w *= (1.0 + trap_bonus)
                a0, b0 = int(y_jodi[0]), int(y_jodi[1])
                if j in {f"{(a0+1)%10}{b0}", f"{(a0-1)%10}{b0}", f"{a0}{(b0+1)%10}", f"{a0}{(b0-1)%10}"}:
                    w *= (1.0 + trap_bonus*0.75)
            if j in prot_set:
                w *= 1.35
            cands.append((j, w))
    cands.sort(key=lambda x: x[1], reverse=True)
    uniq = []
    seen = set()
    for j,_ in cands:
        if j not in seen:
            seen.add(j); uniq.append(j)
        if len(uniq) >= CONFIG["NUM_JODIS"]:
            break
    return uniq

# =========================
# ====== PREDICTION =======
# =========================
def predict_market(mkt: str, rows: List[Dict[str, Any]], all_hist: Dict[str, List[Dict[str, Any]]],
                   asof: datetime.date, dna_map: Dict[str, Any], yctx: Dict[str, Any]) -> Dict[str, Any]:
    # choose freq source
    freq_map = build_digit_freq_for_market(rows, asof) if len(rows) >= CONFIG["MIN_HISTORY_ROWS_FOR_MARKET"] else build_global_digit_freq(all_hist, asof)

    # DNA-driven params
    dnap = adapt_params_from_dna(mkt, dna_map.get(mkt, {}), asof)
    mix = dnap["mix"]
    pen = dnap["pen"]
    prot = dnap["protect"]
    double_w = dnap["double_weight"]
    trap_bonus = dnap["trap_bonus"]

    last = yctx.get("last", {}).get(mkt, {}) or {}
    y_od = [ch for ch in (last.get("od") or "") if ch.isdigit()]
    y_cd = [ch for ch in (last.get("cd") or "") if ch.isdigit()]
    y_j  = last.get("j") if isinstance(last.get("j"), str) and last.get("j").isdigit() and len(last.get("j"))==2 else None
    y_p  = {p for p in [last.get("op"), last.get("cp")] if isinstance(p, str) and len(p)==3 and p.isdigit()}

    # protected repeat slots (if enabled by DNA)
    prot_open_digits  = y_od[:1] if prot["digit"] and y_od else []
    prot_close_digits = y_cd[:1] if prot["digit"] and y_cd else []
    prot_jodis        = [y_j] if (prot["jodi"] and y_j) else []

    # digits
    open_d  = compose_digits(freq_map, CONFIG["NUM_OPEN_DIGITS"], mix, trap_bonus, y_od, prot_open_digits)
    close_d = compose_digits(freq_map, CONFIG["NUM_CLOSE_DIGITS"], mix, trap_bonus, y_cd, prot_close_digits)
    open_d  = list(dict.fromkeys(open_d))[:CONFIG["NUM_OPEN_DIGITS"]]
    close_d = list(dict.fromkeys(close_d))[:CONFIG["NUM_CLOSE_DIGITS"]]

    # pannas pools
    open_pool  = pannas_pool_for_digits(open_d)
    close_pool = pannas_pool_for_digits(close_d)
    if CONFIG["ENFORCE_SUMMOD_DIGIT"]:
        open_pool  = [p for p in open_pool if sum_mod10(p) in set(map(int, open_d))]
        close_pool = [p for p in close_pool if sum_mod10(p) in set(map(int, close_d))]

    open_p  = sample_pannas(open_pool,  CONFIG["NUM_OPEN_PANNAS"],  y_p, double_w)
    close_p = sample_pannas(close_pool, CONFIG["NUM_CLOSE_PANNAS"], y_p, double_w)

    # jodis grid
    jfreq = Counter()
    for rec in rows:
        jj = rec.get("Jodi") or rec.get("jodi")
        if isinstance(jj, str) and jj.isdigit() and len(jj)==2:
            jfreq[jj] += 1
    jprot = prot_jodis
    jodis = build_jodis_grid(open_d, close_d, jfreq, y_j, trap_bonus, jprot)

    return {
        "OpenDigits": open_d,
        "CloseDigits": close_d,
        "OpenPannas": open_p,
        "ClosePannas": close_p,
        "Jodis": jodis
    }

# =========================
# ========== I/O ==========
# =========================
def to_app_structure(preds: Dict[str, Dict[str, Any]], run_date: datetime.date) -> Dict[str, Any]:
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
            }
        })
    return {
        "meta": { "generated_at": iso_timestamp(), "engine": "v10" },
        "markets": markets_array
    }

def run_once(for_date: Optional[datetime.date]=None, *, force_prod: bool=False, dry_run: bool=False) -> Dict[str, Any]:
    cfg = CONFIG
    # seed
    if (not force_prod) and cfg["TEST_MODE"]:
        random.seed(cfg.get("TEST_SEED", 12345))
    else:
        random.seed()

    if for_date is None:
        for_date = datetime.date.today()

    # ensure dirs
    safe_mkdir(cfg["OUTPUT_DIR"]); safe_mkdir(os.path.dirname(cfg["STATE_FILE"]))
    safe_mkdir(cfg["ASSETS_DIR"])  # ensure assets dir for mirroring

    # load inputs
    all_hist = load_market_history(cfg["HISTORY_FILE"])
    if not all_hist:
        raise RuntimeError(f"Missing or empty history file: {cfg['HISTORY_FILE']}")
    validate_history_schema(all_hist)

    dna_map = load_dna(cfg["DNA_FILE"])
    yctx    = learn_from_yesterday(for_date, all_hist)

    # build predictions
    internal_preds = {}
    for mkt, rows in all_hist.items():
        try:
            internal_preds[mkt] = predict_market(mkt, rows, all_hist, for_date, dna_map, yctx)
        except Exception as e:
            logger.warning(f"{mkt}: prediction error {e}")
            internal_preds[mkt] = {"OpenDigits": [],"CloseDigits": [],"OpenPannas": [],"ClosePannas": [],"Jodis": []}

    app_json = to_app_structure(internal_preds, for_date)

    # outputs
    tag = today_str(for_date)
    dated_out = os.path.join(cfg["OUTPUT_DIR"], f"predictions_{tag}_v10.json")
    today_out = cfg["TODAY_PROD_FILE"] if (force_prod or not cfg["TEST_MODE"]) else cfg["TODAY_TEST_FILE"]
    assets_today = cfg["TODAYS_PREDICTIONS_JSON"]
    assets_dated = os.path.join(cfg["ASSETS_DIR"], f"predictions_{tag}_v10.json")

    if not dry_run:
        save_json(dated_out, app_json)
        save_json(today_out, app_json)
        save_json(assets_today, app_json)
        save_json(assets_dated, app_json)

    logger.info(f"OK v10 — markets: {len(internal_preds)} | dated: {dated_out} | today: {today_out} | mirrored: {assets_today}")
    return internal_preds

# =========================
# ========= MAIN ==========
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SattaMatkaAI Prediction Engine v10 (DNA Adaptive).")
    ap.add_argument("--prod", action="store_true", help="Force PROD mode (override TEST_MODE)")
    ap.add_argument("--dry-run", action="store_true", help="Compute but do not write JSON outputs")
    args = ap.parse_args()
    run_once(force_prod=args.prod, dry_run=args.dry_run)