# -*- coding: utf-8 -*-
"""
prediction_engine_v51_unified.py — SattaMatkaAI (Psychology + ML Ensemble) v51

CHANGES FROM v50:
  1. ROLE-AWARE SELECTION: Open/Close digits and pannas are forced to diverge.
  2. ENHANCED CONFIDENCE: Granular agreement bonus with cap raised to 0.88.
  3. DEEPER PSYCHOLOGY: Post-trap, post-repeat, DoW, deep-cold, double-panna.
  4. GRANULAR BACKTEST STATS: per-signal, DoW, trend, coverage in meta.
  5. ML ranker respects role exclusion.

CLI:
  python prediction_engine_v51_unified.py --prod --verbose
"""

import os, sys, json, random, datetime, argparse, logging, logging.handlers, tempfile, bisect, hashlib, warnings
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

try:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    warnings.warn("pip install scikit-learn numpy")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

_DEFAULTS = {
    "HISTORY_FILE": r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\sattaboss-data\data\all_markets_history.json",
    "OUTPUT_DIR":   r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output",
    "STATE_FILE":   r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\state\engine_v51_unified_state.json",
    "TODAY_TEST_FILE": r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\todays_predictions_v51_TEST.json",
    "TODAY_PROD_FILE": r"C:\Users\VCOM\AndroidStudioProjects\multi_market_predictions_v3\assets\todays_predictions.json",
    "LOG_FILE":   r"C:\Users\VCOM\Desktop\Kalyan_Penal_Scraper\output\engine_v51_unified.log",
}

_ENV_PREFIX = "SATTA_"
_LOCAL_FALLBACK_DIR = Path(__file__).resolve().parent / "satta_data"

def _resolve_path(key: str, cli_value: Optional[str]) -> Path:
    if cli_value: return Path(cli_value)
    env_val = os.environ.get(_ENV_PREFIX + key)
    if env_val: return Path(env_val)
    default = Path(_DEFAULTS[key])
    if default.parent.exists(): return default
    return _LOCAL_FALLBACK_DIR / default.name

CONFIG = {
    "NUM_OPEN_DIGITS": 3, "NUM_CLOSE_DIGITS": 3,
    "NUM_OPEN_PANNAS": 8, "NUM_CLOSE_PANNAS": 8, "NUM_JODIS": 9,
    "EXCLUDE_TRIPLE_PANNAS": True,
    "BACKTEST_DAYS_FIRST_RUN": 15,
    "MIN_HISTORY_FOR_ML": 60,
    "NEGATIVE_SAMPLES_PER_POSITIVE": 15,
    "BACKTEST_ML_MAX_ITER": 50,
    "MICRO_WINDOW_DAYS": 2, "MESO_WINDOW_DAYS": 5, "MACRO_WINDOW_DAYS": 15,
    "FREQ_LOOKBACK_DAYS": 45, "MARKOV_LOOKBACK_DAYS": 365,
    "DUE_THEORY_MAX_DAYS": 20, "CYCLE_FIBONACCI": [5, 8, 13],
    "PRIORITY_HARD": 1000.0, "PRIORITY_SOFT": 100.0, "PRIORITY_NOISE": 1.0,
    "MARKOV_BOOST_BASE": 1.60, "MARKOV_MIN_COUNT": 3,
    "DUE_BOOST_BASE": 1.30, "CYCLE_BREAKER_BOOST": 1.40,
    "COLD_REGRESSION_DAYS": 12, "COLD_REGRESSION_BOOST": 1.25,
    "TOM_TRAP_BOOST": 1.50, "TOM_ANTIHERD_PENALTY": 0.35,
    "BREAK_PENALTY": 0.15, "STREAK_PENALTY": 0.10,
    "SUMMOD_BOOST": 1.45, "NEIGHBOR_BOOST": 1.25,
    "MIRROR_BOOST": 1.35, "CUT_BOOST": 1.40,
    "RED_JODI_PENALTY": 0.25,
    "PANNA_RECENT_PENALTY": 0.35, "PANNA_MIRROR_BONUS": 1.30,
    "PANNA_CUT_BONUS": 1.25, "PANNA_SUMMOD_BONUS": 1.20,
    "BEAUTY_PENALTY": 0.45, "UGLY_BOOST": 1.40,
    "OPEN_CUT_BOOST": 1.45, "OPEN_NEIGHBOR_BOOST": 1.20,
    "CLOSE_MIRROR_BOOST": 1.50, "CLOSE_SUMCOMP_BOOST": 1.35,
    "ROLE_OVERLAP_DIGIT_PENALTY": 0.35,
    "ROLE_OVERLAP_PANNA_PENALTY": 0.20,
    "POST_TRAP_BOOST": 1.45, "POST_REPEAT_BOOST": 1.35,
    "POST_TRAP_DIGITS": ["0","1","9"], "POST_REPEAT_DIGITS": ["0","4","5"],
    "DOUBLE_PANNA_RATE": 0.256, "DOUBLE_PANNA_BOOST": 1.15,
    "COLD_REGRESSION_DEEP_DAYS": 15, "COLD_REGRESSION_DEEP_BOOST": 1.40,
    "SIGNAL_ACCURACY_LOOKBACK": 30,
    "SIGNAL_MUTE_THRESHOLD": 0.15, "SIGNAL_BOOST_THRESHOLD": 0.55, "SIGNAL_MIN_SAMPLES": 5,
    "ENSEMBLE_PSYCH_WEIGHT": 0.45, "ENSEMBLE_ML_WEIGHT": 0.45, "ENSEMBLE_AGREEMENT_BONUS": 0.30,
    "CONFIDENCE_BASE": 0.35, "CONFIDENCE_CAP": 0.88, "CONFIDENCE_MIN": 0.05,
    "RULE_WEIGHT_WHEN_BLENDED": 0.40, "ACCURACY_LOOKBACK_DAYS": 30,
    "MIN_BACKTEST_SAMPLES_FOR_BLEND": 5, "STATE_HISTORY_CAP_PER_MARKET": 90,
    "ENABLE_LOGGING": True, "LOG_TO_CONSOLE": False,
    "LOG_MAX_MB": 5, "LOG_BACKUPS": 3,
    "TEST_MODE": False, "TEST_SEED": 20251109,
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

# FIX: Strictly disjoint BEAUTY and UGLY sets
BEAUTY_PANNAS = {"111","222","333","444","555","666","777","888","999","123","234","345","456","567","678","789","890","901","135","246","357","468","579","680","791","802","913","147","258","369","470","581","692","703","814","925","118","227","336","445","554","663","772","881","990"}
UGLY_PANNAS = {"079","089","288","577","499","389","479","569","578","668","677","399","489","579","588","669","678","589","679","688","778","599","689","779","788","699","789","799","889","890","169","178","259","268","277","349","358","367","448","457","466","556","790","880","899","269","278","359","368","377","449","458","467","557","566","189","279","369","378","459","468","477","558","567","990","199","289","379","388","469","478","559","568","577","667","999","666","117","126","135","144","180","225","234","270","360","450","900","333","246","357","468","579","680"}
UGLY_PANNAS = UGLY_PANNAS - BEAUTY_PANNAS  # Enforce disjoint

logger = logging.getLogger("pred_engine_v51")
logger.setLevel(logging.INFO)

def setup_logging(log_file: Path, enable_file: bool, enable_console: bool) -> None:
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if enable_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(str(log_file), maxBytes=CONFIG["LOG_MAX_MB"]*1024*1024, backupCount=CONFIG["LOG_BACKUPS"], encoding="utf-8")
            fh.setFormatter(fmt); logger.addHandler(fh)
        except Exception as e:
            print(f"File logging failed: {e}")
    if enable_console or not logger.handlers:
        ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)

def today_str(dt=None): return (dt or datetime.date.today()).strftime("%Y%m%d")
def iso_timestamp(): return datetime.datetime.now().isoformat()
def safe_mkdir(p): Path(p).mkdir(parents=True, exist_ok=True)

def load_json(p: Path):
    if not Path(p).exists(): return None
    try:
        with open(p,"r",encoding="utf-8") as f: return json.load(f)
    except Exception as e:
        logger.warning(f"JSON parse failed: {e}"); return None

def save_json_atomic(p: Path, obj):
    p = Path(p); safe_mkdir(p.parent)
    fd, tmp = tempfile.mkstemp(prefix=p.name+".", dir=str(p.parent))
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, p)
    except Exception:
        try: os.remove(tmp)
        except OSError: pass
        raise

def parse_date(d):
    if not d: return None
    s = str(d)
    for fmt in ("%Y-%m-%d","%d-%m-%Y","%d/%m/%Y","%Y/%m/%d"):
        try: return datetime.datetime.strptime(s,fmt).date()
        except: pass
    return None

def normalize_panna(raw: str) -> str:
    return raw.zfill(3) if raw.isdigit() and 1 <= len(raw) <= 3 else ""

def normalize_jodi(raw: str, od: str, cd: str) -> str:
    if od.isdigit() and cd.isdigit() and len(od)==1 and len(cd)==1: return f"{od}{cd}"
    if raw.isdigit() and len(raw) <= 2: return raw.zfill(2)
    return ""

def get_field(rec: Dict[str,Any], *keys: str) -> str:
    if not isinstance(rec,dict): return ""
    for k in keys:
        if k in rec and rec[k] not in (None,""): return str(rec[k]).strip()
    norm = {k.lower().replace(" ","").replace("_",""): k for k in keys}
    for rk,rv in rec.items():
        nk = str(rk).lower().replace(" ","").replace("_","")
        if nk in norm and rv not in (None,""): return str(rv).strip()
    return ""

def extract_day_result(rec: Dict[str,Any]) -> Dict[str,str]:
    od = get_field(rec,"Open Digit","OpenDigit","open_digit")
    cd = get_field(rec,"Close Digit","CloseDigit","close_digit")
    j_raw = get_field(rec,"Jodi","jodi")
    op = normalize_panna(get_field(rec,"Open Panna","OpenPanna","open_panna"))
    cp = normalize_panna(get_field(rec,"Close Panna","ClosePanna","close_panna"))
    return {"od":od,"cd":cd,"j":normalize_jodi(j_raw,od,cd),"op":op,"cp":cp}

def weighted_choice(items: List[Any], weights: List[float], k: int) -> List[Any]:
    items = list(items); weights = [max(0.0,float(w)) for w in weights]
    picks = []
    for _ in range(min(k,len(items))):
        s = sum(weights)
        if s <= 0:
            if not items: break
            idx = random.randrange(len(items))
            picks.append(items.pop(idx)); weights.pop(idx)
            continue
        r, acc = random.random()*s, 0.0
        for i,w in enumerate(weights):
            acc += w
            if r <= acc:
                picks.append(items.pop(i)); weights.pop(i); break
    return picks

def mirror_jodi(j: str) -> str:
    if len(j) == 2: return f"{(int(j[0])+5)%10}{(int(j[1])+5)%10}"
    return ""

def mirror_panna(p: str) -> str:
    return p[::-1] if len(p) == 3 else ""

def load_market_history(path: Path) -> Dict[str, List[Dict[str,Any]]]:
    data = load_json(path)
    if isinstance(data,dict): return {k:v for k,v in data.items() if isinstance(v,list)}
    if isinstance(data,list):
        grouped: Dict[str,List[Dict[str,Any]]] = defaultdict(list)
        for rec in data:
            m = rec.get("Market") or rec.get("market") or "UNKNOWN"
            grouped[str(m)].append(rec)
        return dict(grouped)
    return {}

def validate_history_schema(all_hist):
    if not isinstance(all_hist,dict) or not all_hist: raise RuntimeError("Empty history")
    check=0
    for mkt,rows in all_hist.items():
        if not isinstance(rows,list): raise TypeError(f"{mkt}: not list")
        for r in rows:
            if not isinstance(r,dict): raise TypeError(f"{mkt}: row not dict")
            if not any(k in r for k in ("OpenPanna","open_panna","ClosePanna","close_panna","Jodi","jodi","Date","date")):
                raise ValueError(f"{mkt}: missing keys")
        check+=1
        if check>=3: break

class MarketIndex:
    __slots__ = ("dates_asc","by_date")
    def __init__(self, rows: List[Dict[str,Any]]):
        by_date: Dict[datetime.date, Dict[str,Any]] = {}
        for r in rows:
            d = parse_date(r.get("Date") or r.get("date"))
            if d: by_date[d] = r
        self.by_date = by_date
        self.dates_asc = sorted(by_date.keys())
    def last_n_on_or_before(self, asof: datetime.date, n: int):
        pos = bisect.bisect_right(self.dates_asc, asof)
        if pos==0: return []
        start = max(0, pos-n)
        return [(d,self.by_date[d]) for d in reversed(self.dates_asc[start:pos])]
    def row_for_date(self, d: datetime.date): return self.by_date.get(d)
    def __len__(self): return len(self.by_date)

# ------------------------------------------------------------------
# PSYCHOLOGICAL ENGINE (v51 cognitive)
# ------------------------------------------------------------------
class DoWPredictor:
    def __init__(self, idx: MarketIndex, asof: datetime.date):
        self.dow_boost: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._build(idx, asof)
    def _build(self, idx, asof):
        last60 = idx.last_n_on_or_before(asof, 60)
        dow_digits: Dict[int, Counter] = defaultdict(Counter)
        for d, r in last60:
            res = extract_day_result(r)
            dow = d.weekday()
            if res["od"]: dow_digits[dow][res["od"]] += 1
            if res["cd"]: dow_digits[dow][res["cd"]] += 1
        for dow, cnt in dow_digits.items():
            total = sum(cnt.values())
            if total < 10: continue
            for digit, c in cnt.items():
                freq = c / total
                if freq > 0.18:
                    self.dow_boost[dow][digit] = 1.0 + (freq - 0.18) * 3.0
    def boost(self, dow: int, digit: str) -> float:
        return self.dow_boost.get(dow, {}).get(digit, 1.0)

class PostSequenceBias:
    def __init__(self, idx: MarketIndex, asof: datetime.date):
        self.trap_boost_digits: Set[str] = set()
        self.repeat_boost_digits: Set[str] = set()
        self._compute(idx, asof)
    def _compute(self, idx, asof):
        last5 = idx.last_n_on_or_before(asof, 5)
        if len(last5) < 3: return
        for i in range(min(3, len(last5)-1)):
            r0 = extract_day_result(last5[i][1])
            r1 = extract_day_result(last5[i+1][1])
            if r0["od"] and r1["od"]:
                diff = abs(int(r0["od"]) - int(r1["od"]))
                if diff == 1 or diff == 9:
                    # FIX: use update instead of overwrite
                    self.trap_boost_digits.update(set(CONFIG["POST_TRAP_DIGITS"]))
            if r0["cd"] and r1["cd"]:
                diff = abs(int(r0["cd"]) - int(r1["cd"]))
                if diff == 1 or diff == 9:
                    self.trap_boost_digits.update(set(CONFIG["POST_TRAP_DIGITS"]))
        for i in range(min(3, len(last5)-1)):
            r0 = extract_day_result(last5[i][1])
            r1 = extract_day_result(last5[i+1][1])
            if r0["od"] and r1["od"] and r0["od"] == r1["od"]:
                self.repeat_boost_digits.update(set(CONFIG["POST_REPEAT_DIGITS"]))
            if r0["cd"] and r1["cd"] and r0["cd"] == r1["cd"]:
                self.repeat_boost_digits.update(set(CONFIG["POST_REPEAT_DIGITS"]))

class MultiScaleAnalyzer:
    __slots__ = ("micro","meso","macro")
    def __init__(self, idx: MarketIndex, asof: datetime.date):
        self.micro = self._analyze(idx, asof, CONFIG["MICRO_WINDOW_DAYS"])
        self.meso = self._analyze(idx, asof, CONFIG["MESO_WINDOW_DAYS"])
        self.macro = self._analyze(idx, asof, CONFIG["MACRO_WINDOW_DAYS"])
    @staticmethod
    def _analyze(idx, asof, n):
        last = idx.last_n_on_or_before(asof, n)
        digits=Counter(); jodis=Counter(); pannas=Counter()
        od_list=[]; cd_list=[]
        for _,r in last:
            res = extract_day_result(r)
            if res["od"]: digits[res["od"]]+=1; od_list.append(res["od"])
            if res["cd"]: digits[res["cd"]]+=1; cd_list.append(res["cd"])
            if res["j"]: jodis[res["j"]]+=1
            if res["op"]: pannas[res["op"]]+=1
            if res["cp"]: pannas[res["cp"]]+=1
        return {"digits":digits,"jodis":jodis,"pannas":pannas,"od_list":od_list,"cd_list":cd_list,"length":len(last)}

class DueTheory:
    __slots__ = ("due_digits","cycle_digits","cold_digits","deep_cold_digits")
    def __init__(self, idx: MarketIndex, asof: datetime.date):
        self.due_digits=set(); self.cycle_digits=set(); self.cold_digits=set(); self.deep_cold_digits=set()
        self._compute(idx,asof)
    def _compute(self, idx, asof):
        last20 = idx.last_n_on_or_before(asof, CONFIG["DUE_THEORY_MAX_DAYS"])
        seen=set()
        for _,r in last20:
            res=extract_day_result(r)
            if res["od"]: seen.add(res["od"])
            if res["cd"]: seen.add(res["cd"])
        self.due_digits = {str(i) for i in range(10)} - seen
        for d in map(str,range(10)):
            positions=[]
            for i,(_,r) in enumerate(last20):
                res=extract_day_result(r)
                if res["od"]==d or res["cd"]==d: positions.append(i)
            if len(positions)>=2:
                gaps=[positions[j]-positions[j-1] for j in range(1,len(positions))]
                if any(g in CONFIG["CYCLE_FIBONACCI"] for g in gaps[-3:]): self.cycle_digits.add(d)
        last_cold = idx.last_n_on_or_before(asof, CONFIG["COLD_REGRESSION_DAYS"])
        cold_seen=set()
        for _,r in last_cold:
            res=extract_day_result(r)
            if res["od"]: cold_seen.add(res["od"])
            if res["cd"]: cold_seen.add(res["cd"])
        self.cold_digits = {str(i) for i in range(10)} - cold_seen
        deep = idx.last_n_on_or_before(asof, CONFIG["COLD_REGRESSION_DEEP_DAYS"])
        deep_seen=set()
        for _,r in deep:
            res=extract_day_result(r)
            if res["od"]: deep_seen.add(res["od"])
            if res["cd"]: deep_seen.add(res["cd"])
        self.deep_cold_digits = {str(i) for i in range(10)} - deep_seen

class TheoryOfMind:
    __slots__ = ("trap_digits","anti_herd_digits")
    def __init__(self, idx: MarketIndex, asof: datetime.date, yctx: Dict[str,str], msa: Optional[MultiScaleAnalyzer]=None):
        self.trap_digits=set(); self.anti_herd_digits=set()
        self._compute(idx,asof,yctx,msa)
    def _compute(self, idx, asof, yctx, msa):
        herd=set()
        y_od,y_cd,y_j = yctx.get("od",""), yctx.get("cd",""), yctx.get("j","")
        if y_od.isdigit():
            herd.add(str((int(y_od)+5)%10)); herd.add(str((int(y_od)+1)%10)); herd.add(str((int(y_od)-1)%10))
        if y_cd.isdigit(): herd.add(str((int(y_cd)+5)%10))
        if y_j and len(y_j)==2: herd.add(y_j[1]); herd.add(y_j[0])
        self.trap_digits = herd.copy()
        last2 = idx.last_n_on_or_before(asof,2)
        if len(last2)>=2:
            r0=extract_day_result(last2[0][1]); r1=extract_day_result(last2[1][1])
            if r1.get("od") and r0.get("od") and r0["od"]==str((int(r1["od"])+5)%10):
                self.trap_digits.add(str((int(r0["od"])+5)%10))
        macro = (msa.macro if msa is not None else MultiScaleAnalyzer(idx,asof).macro)
        hot = {d for d,c in macro["digits"].items() if c>=4}
        self.anti_herd_digits = ({str(i) for i in range(10)} - herd) - hot

class MarkovMachine:
    __slots__ = ("oc","co","oo","cc","oj","cj","built")
    def __init__(self):
        self.oc=defaultdict(Counter); self.co=defaultdict(Counter); self.oo=defaultdict(Counter)
        self.cc=defaultdict(Counter); self.oj=defaultdict(Counter); self.cj=defaultdict(Counter)
        self.built=False
    def build(self, idx: MarketIndex, asof: datetime.date, lookback: int):
        last = idx.last_n_on_or_before(asof, lookback)
        chronological = list(reversed(last))
        for i in range(1,len(chronological)):
            prev=extract_day_result(chronological[i-1][1]); curr=extract_day_result(chronological[i][1])
            if prev["od"] and curr["cd"]: self.oc[prev["od"]][curr["cd"]]+=1
            if prev["cd"] and curr["od"]: self.co[prev["cd"]][curr["od"]]+=1
            if prev["od"] and curr["od"]: self.oo[prev["od"]][curr["od"]]+=1
            if prev["cd"] and curr["cd"]: self.cc[prev["cd"]][curr["cd"]]+=1
            if prev["od"] and curr["j"]: self.oj[prev["od"]][curr["j"]]+=1
            if prev["cd"] and curr["j"]: self.cj[prev["cd"]][curr["j"]]+=1
        self.built=True
    def prob(self, matrix, from_state, to_state):
        cnt=matrix.get(from_state)
        if not cnt: return 0.0
        total=sum(cnt.values())
        if total<CONFIG["MARKOV_MIN_COUNT"]: return 0.0
        return cnt.get(to_state,0)/total
    def top_n(self, matrix, from_state, n):
        cnt=matrix.get(from_state)
        if not cnt: return []
        total=sum(cnt.values())
        if total<CONFIG["MARKOV_MIN_COUNT"]: return []
        probs=[(k,v/total) for k,v in cnt.items()]
        probs.sort(key=lambda x:x[1], reverse=True)
        return probs[:n]

class SignalRegistry:
    RULES = ["markov_oc","markov_co","markov_oo","markov_cc","markov_oj","markov_cj",
             "due_theory","cycle_breaker","cold_regression","mirror","cut","sumcomp",
             "neighbor","fakeout_break","anti_herd","tom_level1","tom_level2",
             "aesthetic_ugly","aesthetic_beauty","open_time_cut","close_time_mirror","interdependence_bind",
             "post_trap","post_repeat","dow_pattern","deep_cold"]
    def __init__(self, state: Dict[str,Any]):
        self.hits={r:deque(maxlen=CONFIG["SIGNAL_ACCURACY_LOOKBACK"]) for r in self.RULES}
        self.boosts={r:1.0 for r in self.RULES}
        saved=state.get("signal_registry",{})
        for r in self.RULES:
            if r in saved:
                self.boosts[r]=saved[r].get("boost",1.0)
                self.hits[r].extend(saved[r].get("history",[])[-CONFIG["SIGNAL_ACCURACY_LOOKBACK"]:])
    def record(self, rule: str, hit: bool):
        self.hits[rule].append(1 if hit else 0); self._recalc(rule)
    def _recalc(self, rule: str):
        hist=list(self.hits[rule]); n=len(hist)
        if n<CONFIG["SIGNAL_MIN_SAMPLES"]: self.boosts[rule]=1.0; return
        acc=sum(hist)/n
        if acc<CONFIG["SIGNAL_MUTE_THRESHOLD"]: self.boosts[rule]=0.2
        elif acc>CONFIG["SIGNAL_BOOST_THRESHOLD"]: self.boosts[rule]=min(2.0,1.0+(acc-CONFIG["SIGNAL_BOOST_THRESHOLD"])*2.0)
        else: self.boosts[rule]=1.0
    def get(self, rule: str)->float: return self.boosts.get(rule,1.0)
    def to_dict(self): return {r:{"boost":self.boosts[r],"history":list(self.hits[r])} for r in self.RULES}

class HierarchicalPriority:
    HARD=2; SOFT=1; NOISE=0
    def __init__(self): self.stack: List[Tuple[int,str,float,Any]]=[]
    def add(self, level: int, rule: str, weight: float, meta: Any=None): self.stack.append((level,rule,weight,meta))
    def resolve(self, registry: SignalRegistry)->Tuple[Dict[str,float],Dict[str,List[str]]]:
        by_digit: Dict[str,List[Tuple[int,float,str]]]=defaultdict(list)
        for level,rule,weight,meta in self.stack:
            adj=registry.get(rule) if rule in registry.RULES else 1.0
            eff=weight*adj
            if isinstance(meta,str) and meta.isdigit() and len(meta)==1:
                by_digit[meta].append((level,eff,rule))
            elif isinstance(meta,(list,tuple)):
                for d in meta:
                    if isinstance(d,str) and d.isdigit() and len(d)==1:
                        by_digit[d].append((level,eff,rule))
        final: Dict[str,float]={}; attr: Dict[str,List[str]]={}
        for d,entries in by_digit.items():
            entries.sort(key=lambda x:(x[0],x[1]), reverse=True)
            top_level=entries[0][0]
            same=[e for e in entries if e[0]==top_level]
            # FIX: Use addition instead of multiplication to prevent score collapse
            w=1.0; rules_here=[]
            for _,weight,rule in same:
                w += (weight - 1.0)  # Add the delta from 1.0
                if rule not in rules_here: rules_here.append(rule)
            final[d]=max(0.0, w); attr[d]=rules_here
        return final,attr

class AestheticPsychology:
    @staticmethod
    def weight(panna: str)->float:
        if panna in BEAUTY_PANNAS: return CONFIG["BEAUTY_PENALTY"]
        if panna in UGLY_PANNAS: return CONFIG["UGLY_BOOST"]
        return 1.0

def build_base_freq(idx: MarketIndex, asof: datetime.date, n_days: int)->Dict[str,float]:
    last=idx.last_n_on_or_before(asof,n_days)
    cnt=Counter()
    for _,r in last:
        res=extract_day_result(r)
        if res["od"].isdigit(): cnt[res["od"]]+=1
        if res["cd"].isdigit(): cnt[res["cd"]]+=1
    return {str(i):(cnt.get(str(i),0)+1.0)*0.1 for i in range(10)}

def get_day_context(idx: MarketIndex, asof: datetime.date, offset_days: int=1)->Dict[str,str]:
    target_rows=idx.last_n_on_or_before(asof,offset_days)
    if len(target_rows)<offset_days: return {}
    _,r=target_rows[offset_days-1]
    return extract_day_result(r)

def compute_signals_v51(idx: MarketIndex, asof: datetime.date, yctx: Dict[str,str], mm: MarkovMachine, registry: SignalRegistry):
    stack=HierarchicalPriority()
    y_od,y_cd,y_j=yctx.get("od",""),yctx.get("cd",""),yctx.get("j","")
    msa=MultiScaleAnalyzer(idx,asof)
    due=DueTheory(idx,asof)
    tom=TheoryOfMind(idx,asof,yctx,msa)
    post=PostSequenceBias(idx,asof)
    dow=DoWPredictor(idx,asof)
    today_dow = asof.weekday()

    if y_od.isdigit():
        for d,prob in mm.top_n(mm.oc,y_od,2):
            if prob>=0.30: stack.add(HierarchicalPriority.HARD,"markov_oc",CONFIG["MARKOV_BOOST_BASE"]*prob,d)
    if y_cd.isdigit():
        for d,prob in mm.top_n(mm.co,y_cd,2):
            if prob>=0.30: stack.add(HierarchicalPriority.HARD,"markov_co",CONFIG["MARKOV_BOOST_BASE"]*prob,d)
    if y_od.isdigit():
        for d,prob in mm.top_n(mm.oo,y_od,2):
            if prob>=0.30: stack.add(HierarchicalPriority.HARD,"markov_oo",CONFIG["MARKOV_BOOST_BASE"]*prob,d)
    if y_cd.isdigit():
        for d,prob in mm.top_n(mm.cc,y_cd,2):
            if prob>=0.30: stack.add(HierarchicalPriority.HARD,"markov_cc",CONFIG["MARKOV_BOOST_BASE"]*prob,d)

    for d in due.due_digits: stack.add(HierarchicalPriority.HARD,"due_theory",CONFIG["DUE_BOOST_BASE"],d)
    for d in due.cycle_digits: stack.add(HierarchicalPriority.HARD,"cycle_breaker",CONFIG["CYCLE_BREAKER_BOOST"],d)
    for d in due.cold_digits: stack.add(HierarchicalPriority.HARD,"cold_regression",CONFIG["COLD_REGRESSION_BOOST"],d)
    for d in due.deep_cold_digits: stack.add(HierarchicalPriority.HARD,"deep_cold",CONFIG["COLD_REGRESSION_DEEP_BOOST"],d)
    for d in tom.anti_herd_digits: stack.add(HierarchicalPriority.HARD,"tom_level2",CONFIG["TOM_TRAP_BOOST"],d)
    for d,c in msa.macro["digits"].items():
        if c>=4: stack.add(HierarchicalPriority.HARD,"anti_herd",CONFIG["TOM_ANTIHERD_PENALTY"],d)

    for d in post.trap_boost_digits: stack.add(HierarchicalPriority.HARD,"post_trap",CONFIG["POST_TRAP_BOOST"],d)
    for d in post.repeat_boost_digits: stack.add(HierarchicalPriority.HARD,"post_repeat",CONFIG["POST_REPEAT_BOOST"],d)

    for d in map(str, range(10)):
        dow_b = dow.boost(today_dow, d)
        if dow_b > 1.0:
            stack.add(HierarchicalPriority.SOFT, "dow_pattern", dow_b, d)

    if y_od.isdigit():
        cut_o=str((int(y_od)+5)%10)
        stack.add(HierarchicalPriority.SOFT,"open_time_cut",CONFIG["OPEN_CUT_BOOST"],cut_o)
        for nb in [(int(y_od)+1)%10,(int(y_od)-1)%10]:
            stack.add(HierarchicalPriority.SOFT,"neighbor",CONFIG["OPEN_NEIGHBOR_BOOST"],str(nb))
        stack.add(HierarchicalPriority.SOFT,"break",CONFIG["BREAK_PENALTY"],y_od)
    if y_cd.isdigit():
        cut_c=str((int(y_cd)+5)%10)
        stack.add(HierarchicalPriority.SOFT,"close_time_mirror",CONFIG["CLOSE_MIRROR_BOOST"],cut_c)
        sum_c = None
        if y_od.isdigit(): sum_c=str((10-int(y_od))%10)
        elif y_cd.isdigit(): sum_c=str((10-int(y_cd))%10)
        if sum_c: stack.add(HierarchicalPriority.SOFT,"close_time_sumcomp",CONFIG["CLOSE_SUMCOMP_BOOST"],sum_c)
        stack.add(HierarchicalPriority.SOFT,"break",CONFIG["BREAK_PENALTY"],y_cd)
    if y_j and len(y_j)==2:
        stack.add(HierarchicalPriority.SOFT,"mirror",CONFIG["MIRROR_BOOST"],y_j[1])
        stack.add(HierarchicalPriority.SOFT,"mirror",CONFIG["MIRROR_BOOST"],y_j[0])

    last2=idx.last_n_on_or_before(asof,2)
    if len(last2)>=2:
        j0=extract_day_result(last2[0][1]).get("j","")
        j1=extract_day_result(last2[1][1]).get("j","")
        if j0 and j1 and (j0==j1 or j0==mirror_jodi(j1)):
            for ch in j0:
                if ch.isdigit(): stack.add(HierarchicalPriority.SOFT,"fakeout_break",0.25,ch)
    for d,c in msa.meso["digits"].items():
        if c>=3: stack.add(HierarchicalPriority.SOFT,"streak",CONFIG["STREAK_PENALTY"],d)

    base_freq=build_base_freq(idx,asof,CONFIG["FREQ_LOOKBACK_DAYS"])
    for d,w in base_freq.items(): stack.add(HierarchicalPriority.NOISE,"base_freq",w,d)

    digit_weights,digit_attr=stack.resolve(registry)
    return {"digit_weights":digit_weights,"digit_attribution":digit_attr,
            "y_od":y_od,"y_cd":y_cd,"y_j":y_j,"due":due,"tom":tom,"msa":msa,"mm":mm,"stack":stack,
            "post":post,"dow":dow}

def pannas_pool_for_digits(digs: List[str])->List[str]:
    raw=[]
    for d in digs:
        try: raw+=VALID_PANNAS_BY_LAST_DIGIT.get(int(d),[])
        except: continue
    seen=set(); out=[]
    for p in raw:
        if p not in seen: seen.add(p); out.append(p)
    if CONFIG["EXCLUDE_TRIPLE_PANNAS"]: out=[p for p in out if not(len(set(p))==1 and len(p)==3)]
    return out

def recent_pannas(idx: MarketIndex, asof: datetime.date, n_days: int)->Set[str]:
    last=idx.last_n_on_or_before(asof,n_days); out=set()
    for _,r in last:
        res=extract_day_result(r)
        if res["op"]: out.add(res["op"])
        if res["cp"]: out.add(res["cp"])
    return out

def select_digits(weights: Dict[str, float], k: int, veto: Set[str]) -> List[str]:
    items = [(d, w) for d, w in weights.items() if d not in veto]
    items.sort(key=lambda x: x[1], reverse=True)
    out = [d for d, _ in items[:k]]
    for d in map(str, range(10)):
        if d not in out and d not in veto:
            out.append(d)
        if len(out) >= k: break
    return out[:k]

def select_digits_role_aware(weights: Dict[str, float], k: int, veto: Set[str],
                              exclude: Set[str], exclude_penalty: float = None) -> List[str]:
    if exclude_penalty is None:
        exclude_penalty = CONFIG["ROLE_OVERLAP_DIGIT_PENALTY"]
    items = [(d, w * (exclude_penalty if d in exclude else 1.0))
             for d, w in weights.items() if d not in veto]
    items.sort(key=lambda x: x[1], reverse=True)
    out = [d for d, _ in items[:k]]
    for d in map(str, range(10)):
        if d not in out and d not in veto:
            out.append(d)
        if len(out) >= k: break
    return out[:k]

def sample_pannas_psych(candidate_pool: List[str], need_k: int, idx: MarketIndex, asof: datetime.date,
                        role: str, sig: Dict[str,Any], chosen_digits: List[str], registry: SignalRegistry,
                        exclude_pannas: Optional[Set[str]] = None)->Tuple[List[str],Dict[str,List[str]],Dict[str,float]]:
    if not candidate_pool: return [],{},{}
    math_valid=[p for p in candidate_pool if str(sum(int(x) for x in p)%10) in chosen_digits]
    if not math_valid:
        logger.warning(f"sample_pannas_psych: no math-valid pannas for {role} digits {chosen_digits}; falling back.")
        math_valid=list(candidate_pool)
        if not math_valid: return [],{},{}
    recent=recent_pannas(idx,asof,5)
    y_od,y_cd=sig["y_od"],sig["y_cd"]
    cut_digit=str((int(y_od if role=="open" else y_cd)+5)%10) if (y_od if role=="open" else y_cd).isdigit() else None
    comp_digit=str((10-int(y_cd if role=="open" else y_od))%10) if (y_cd if role=="open" else y_od).isdigit() else None
    items=list(math_valid)
    ws=[]; scores={}
    exclude_pannas = exclude_pannas or set()
    for p in items:
        w=1.0
        w*=AestheticPsychology.weight(p)
        if p in recent: w*=CONFIG["PANNA_RECENT_PENALTY"]
        mp=mirror_panna(p)
        if mp in recent and mp!=p: w*=CONFIG["PANNA_MIRROR_BONUS"]*registry.get("mirror")
        if cut_digit and cut_digit in p: w*=CONFIG["PANNA_CUT_BONUS"]*registry.get("cut")
        if comp_digit and comp_digit in p: w*=CONFIG["PANNA_SUMMOD_BONUS"]*registry.get("sumcomp")
        if p in exclude_pannas:
            w *= CONFIG["ROLE_OVERLAP_PANNA_PENALTY"]
        if len(set(p)) == 2:
            w *= CONFIG["DOUBLE_PANNA_BOOST"]
        mm=sig["mm"]
        if role=="open" and y_od.isdigit():
            for d,prob in mm.top_n(mm.oc,y_od,3):
                if d in p: w*=1.0+prob
        if role=="close" and y_cd.isdigit():
            for d,prob in mm.top_n(mm.co,y_cd,3):
                if d in p: w*=1.0+prob
        ws.append(max(w,1e-6)); scores[p]=w
    picks=weighted_choice(items,ws,need_k)
    out=[]; seen=set()
    for p in picks:
        if p not in seen:
            seen.add(p); out.append(p)
        if len(out)>=need_k: break
    for p in math_valid:
        if p not in seen:
            seen.add(p); out.append(p)
        if len(out)>=need_k: break
    attr={p:["psych"] for p in out}
    return out[:need_k],attr,scores

def build_jodis_psych(open_d: List[str], close_d: List[str], sig: Dict[str,Any], registry: SignalRegistry)->Tuple[List[str],Dict[str,List[str]],Dict[str,float]]:
    y_j=sig["y_j"]
    mm=sig["mm"]
    y_od,y_cd=sig["y_od"],sig["y_cd"]
    cut_jodi=""
    if y_j and len(y_j)==2:
        cut_jodi=f"{(int(y_j[0])+5)%10}{(int(y_j[1])+5)%10}"
    markov_oj={}
    if y_od.isdigit():
        for j,prob in mm.top_n(mm.oj,y_od,5):
            markov_oj[j]=prob
    cands=[]; scores={}
    for a in open_d:
        for b in close_d:
            j=f"{a}{b}"
            w=1.0
            if j==y_j: w*=0.2
            if a==b:
                if y_j and len(y_j)==2 and y_j[0]==y_j[1]:
                    w*=CONFIG["RED_JODI_PENALTY"]
                else:
                    w*=1.15
            if y_j and j==mirror_jodi(y_j):
                w*=1.5*registry.get("mirror")
            if (int(a)+int(b)) in (10,20):
                w*=1.35
            if cut_jodi and j==cut_jodi:
                w*=CONFIG["CUT_BOOST"]*registry.get("cut")
            if j in markov_oj:
                w*=1.0+markov_oj[j]*2.0
            cands.append((j,w)); scores[j]=w
    cands.sort(key=lambda x:x[1], reverse=True)
    out=[]; seen=set()
    for j,_ in cands:
        if j not in seen:
            seen.add(j); out.append(j)
        if len(out)>=CONFIG["NUM_JODIS"]: break
    attr={j:["psych"] for j in out}
    return out,attr,scores

def rolling_accuracy(entries: List[Dict[str,Any]], days: int)->Optional[Dict[str,Any]]:
    recent=[e for e in entries if e.get("actual") is not None][-days:]
    if not recent: return None
    hits=0; total=0; dow_hits=defaultdict(int); dow_total=defaultdict(int)
    per_signal_hits=defaultdict(int); per_signal_total=defaultdict(int)
    for e in recent:
        actual=e["actual"]; pred=e.get("predictions",{}); sig=e.get("v51_signals",{})
        hit=False
        if actual.get("od") in pred.get("OpenDigits",[]): hit=True; per_signal_hits["open_digit"]+=1
        if actual.get("cd") in pred.get("CloseDigits",[]): hit=True; per_signal_hits["close_digit"]+=1
        if actual.get("op") in pred.get("OpenPannas",[]): hit=True; per_signal_hits["open_panna"]+=1
        if actual.get("cp") in pred.get("ClosePannas",[]): hit=True; per_signal_hits["close_panna"]+=1
        if actual.get("j") in pred.get("Jodis",[]): hit=True; per_signal_hits["jodi"]+=1
        for k in ("due_digits","cycle_digits","cold_digits","deep_cold_digits","anti_herd","trap_digits","post_trap","post_repeat"):
            for d in sig.get(k,[]):
                per_signal_total[k]+=1
                if d in (actual.get("od",""),actual.get("cd","")): per_signal_hits[k]+=1
        if hit: hits+=1
        total+=1
        try:
            d_obj=datetime.date.fromisoformat(e["date"]); dow_total[d_obj.weekday()]+=1
            if hit: dow_hits[d_obj.weekday()]+=1
        except: pass
    dow_acc={int(k):dow_hits[k]/dow_total[k] for k in dow_total if dow_total[k]>0}
    sig_acc={k:per_signal_hits[k]/per_signal_total[k] for k in per_signal_total if per_signal_total[k]>0}
    coverage=sum(1 for e in recent if e.get("v51_signals"))
    return {"overall":hits/total if total else 0.0,"sample_size":total,
            "dow_acc":dow_acc,"per_signal_acc":sig_acc,"coverage":coverage/len(recent)}

def predict_market_psych(mkt: str, idx: MarketIndex, asof: datetime.date, state: Dict[str,Any]):
    registry=SignalRegistry(state)
    mm=MarkovMachine()
    mm.build(idx,asof,CONFIG["MARKOV_LOOKBACK_DAYS"])
    yctx=get_day_context(idx,asof,1)
    sig=compute_signals_v51(idx,asof,yctx,mm,registry)
    veto=set()
    for d,c in sig["msa"].macro["digits"].items():
        if c>=8: veto.add(d)
    open_d=select_digits(sig["digit_weights"],CONFIG["NUM_OPEN_DIGITS"],veto)
    close_d=select_digits_role_aware(sig["digit_weights"],CONFIG["NUM_CLOSE_DIGITS"],
                                      veto, set(open_d))
    if yctx.get("od") and open_d:
        best_close=max(close_d,key=lambda d: mm.prob(mm.oc,yctx["od"],d))
        if mm.prob(mm.oc,yctx["od"],best_close)<0.05 and yctx["od"].isdigit():
            top=mm.top_n(mm.oc,yctx["od"],1)
            if top and top[0][0] not in close_d:
                close_d=[top[0][0]]+close_d[:-1]
    open_pool=pannas_pool_for_digits(open_d)
    close_pool=pannas_pool_for_digits(close_d)
    open_p,open_attr,open_scores=sample_pannas_psych(open_pool,CONFIG["NUM_OPEN_PANNAS"],idx,asof,"open",sig,open_d,registry)
    close_p,close_attr,close_scores=sample_pannas_psych(close_pool,CONFIG["NUM_CLOSE_PANNAS"],idx,asof,"close",sig,close_d,registry, exclude_pannas=set(open_p))
    jodis,jodi_attr,jodi_scores=build_jodis_psych(open_d,close_d,sig,registry)
    rule_conf=CONFIG["CONFIDENCE_BASE"]
    if sig["tom"].anti_herd_digits: rule_conf+=0.10
    if sig["due"].due_digits or sig["due"].cycle_digits: rule_conf+=0.10
    if mm.built: rule_conf+=0.10
    if any(d in sig["tom"].anti_herd_digits for d in open_d+close_d): rule_conf+=0.10
    if sig["post"].trap_boost_digits or sig["post"].repeat_boost_digits: rule_conf+=0.05
    rule_conf=min(rule_conf,CONFIG["CONFIDENCE_CAP"])
    accuracy=rolling_accuracy(state.get("markets",{}).get(mkt,[]),CONFIG["ACCURACY_LOOKBACK_DAYS"])
    if accuracy:
        conf=CONFIG["RULE_WEIGHT_WHEN_BLENDED"]*rule_conf+(1.0-CONFIG["RULE_WEIGHT_WHEN_BLENDED"])*accuracy["overall"]
        basis=f"blended (rules + {accuracy['sample_size']}-day empirical)"
    else:
        conf=rule_conf
        basis="psych_rules_only"
    conf=max(CONFIG["CONFIDENCE_MIN"],min(conf,CONFIG["CONFIDENCE_CAP"]))
    return {
        "OpenDigits":open_d,"CloseDigits":close_d,
        "OpenPannas":open_p,"ClosePannas":close_p,
        "Jodis":jodis,
        "confidence":round(conf,2),"confidence_basis":basis,
        "digit_weights":sig["digit_weights"],
        "panna_attr":{**open_attr,**close_attr},
        "jodi_attr":jodi_attr,
        "panna_scores":{**open_scores,**close_scores},
        "jodi_scores":jodi_scores,
        "v51_signals":{
            "due_digits":list(sig["due"].due_digits),
            "cycle_digits":list(sig["due"].cycle_digits),
            "cold_digits":list(sig["due"].cold_digits),
            "deep_cold_digits":list(sig["due"].deep_cold_digits),
            "anti_herd":list(sig["tom"].anti_herd_digits),
            "trap_digits":list(sig["tom"].trap_digits),
            "post_trap":list(sig["post"].trap_boost_digits),
            "post_repeat":list(sig["post"].repeat_boost_digits),
            "veto_exclusions":list(veto),
        }
    }

# ------------------------------------------------------------------
# ML ENGINE (Panna-First) v51
# ------------------------------------------------------------------
class PannaFeatureBuilder:
    @staticmethod
    def build_context(rows: List[Dict[str,Any]], asof: datetime.date)->Dict[str,Any]:
        dated=[]
        for r in rows:
            d=parse_date(r.get("Date") or r.get("date"))
            if d and d<=asof:
                dated.append((d,extract_day_result(r)))
        dated.sort(key=lambda x:x[0], reverse=True)
        ctx={"n_history":len(dated),"dow":asof.weekday(),"dom":asof.day,"month":asof.month}
        for lag,(_,res) in enumerate(dated[:7],1):
            ctx[f"op_lag{lag}"]=res["op"]; ctx[f"cp_lag{lag}"]=res["cp"]
            ctx[f"od_lag{lag}"]=res["od"]; ctx[f"cd_lag{lag}"]=res["cd"]
            ctx[f"j_lag{lag}"]=res["j"]
        last15=dated[:15]
        digit_cnt=Counter(); panna_cnt=Counter()
        for _,res in last15:
            if res["od"]: digit_cnt[res["od"]]+=1
            if res["cd"]: digit_cnt[res["cd"]]+=1
            if res["op"]: panna_cnt[res["op"]]+=1
            if res["cp"]: panna_cnt[res["cp"]]+=1
        ctx["max_digit_freq_15d"]=max(digit_cnt.values()) if digit_cnt else 0
        ctx["unique_digits_15d"]=len(digit_cnt)
        j1=dated[0][1]["j"] if len(dated)>0 else ""
        j2=dated[1][1]["j"] if len(dated)>1 else ""
        ctx["fakeout_active"]=1 if (j1 and j2 and (j1==j2 or j1==mirror_jodi(j2))) else 0
        ctx["y_od"]=dated[0][1]["od"] if dated else ""
        ctx["y_cd"]=dated[0][1]["cd"] if dated else ""
        ctx["y_op"]=dated[0][1]["op"] if dated else ""
        ctx["y_cp"]=dated[0][1]["cp"] if dated else ""
        ctx["cut_od"]=str((int(ctx["y_od"])+5)%10) if ctx["y_od"].isdigit() else ""
        ctx["cut_cd"]=str((int(ctx["y_cd"])+5)%10) if ctx["y_cd"].isdigit() else ""
        last12_digits=set()
        for _,res in dated[:12]:
            if res["od"]: last12_digits.add(res["od"])
            if res["cd"]: last12_digits.add(res["cd"])
        ctx["cold_digits"]=[d for d in map(str,range(10)) if d not in last12_digits]
        panna_recency={}
        for days_back,(_,res) in enumerate(dated[:30],1):
            for p in (res["op"],res["cp"]):
                if p and p not in panna_recency:
                    panna_recency[p]=days_back
        ctx["panna_recency"]=panna_recency
        
        # FIX: Build separate markov chains for Open and Close roles to prevent data leak
        markov_op=defaultdict(Counter)
        markov_cp=defaultdict(Counter)
        for i in range(len(dated)-1):
            curr_op_digit=PANNA_TO_DIGIT.get(dated[i][1]["op"],"")
            next_op_digit=PANNA_TO_DIGIT.get(dated[i+1][1]["op"],"")
            if curr_op_digit and next_op_digit:
                markov_op[curr_op_digit][next_op_digit]+=1
                
            curr_cp_digit=PANNA_TO_DIGIT.get(dated[i][1]["cp"],"")
            next_cp_digit=PANNA_TO_DIGIT.get(dated[i+1][1]["cp"],"")
            if curr_cp_digit and next_cp_digit:
                markov_cp[curr_cp_digit][next_cp_digit]+=1
        ctx["markov_op_digit"]=markov_op
        ctx["markov_cp_digit"]=markov_cp
        return ctx

    @staticmethod
    def build_candidate_features(ctx: Dict[str,Any], panna: str, role: str="open")->List[float]:
        d=PANNA_TO_DIGIT.get(panna,"0")
        feats=[]
        y_digit=ctx.get("y_od","") if role=="open" else ctx.get("y_cd","")
        cut_digit=ctx.get("cut_od","") if role=="open" else ctx.get("cut_cd","")
        y_panna=ctx.get("y_op","") if role=="open" else ctx.get("y_cp","")
        feats.append(int(d))
        feats.append(1 if len(set(panna))==2 else 0)
        feats.append(1 if panna in BEAUTY_PANNAS else 0)
        feats.append(1 if panna in UGLY_PANNAS else 0)
        rec=ctx["panna_recency"].get(panna,30)
        feats.append(min(rec,30))
        feats.append(1 if panna==y_panna else 0)
        mirror_y=y_panna[::-1] if len(y_panna)==3 else ""
        feats.append(1 if panna==mirror_y else 0)
        feats.append(1 if y_digit and y_digit in panna else 0)
        feats.append(1 if cut_digit and cut_digit in panna else 0)
        feats.append(ctx.get("max_digit_freq_15d",0)/15.0)
        
        # FIX: Use the correct markov chain based on role
        markov_score=0.0
        if y_digit and d:
            mk = ctx.get("markov_op_digit", {}) if role=="open" else ctx.get("markov_cp_digit", {})
            mc = mk.get(y_digit, Counter())
            total=sum(mc.values())
            if total: markov_score=mc.get(d,0)/total
        feats.append(markov_score)
        feats.append(1 if d in ctx.get("cold_digits",[]) else 0)
        feats.append(ctx.get("fakeout_active",0))
        feats.append(0 if role=="open" else 1)
        feats.append(ctx.get("dow",0))
        feats.append(ctx.get("dom",1))
        feats.append(ctx.get("month",1))
        return feats

class PannaRanker:
    def __init__(self):
        self.model=None; self.fitted=False

    def _build_training_data(self, rows: List[Dict[str,Any]])->Tuple[List[List[float]],List[int]]:
        X,y=[],[]
        dated=[]
        for r in rows:
            d=parse_date(r.get("Date") or r.get("date"))
            if d: dated.append((d,extract_day_result(r)))
        dated.sort(key=lambda x:x[0])
        neg_count=CONFIG["NEGATIVE_SAMPLES_PER_POSITIVE"]
        for i in range(1,len(dated)):
            asof=dated[i][0]
            today_res=dated[i][1]
            ctx=PannaFeatureBuilder.build_context([r for _,r in dated[:i]],asof)
            for role,target_panna in (("open",today_res["op"]),("close",today_res["cp"])):
                if not target_panna or target_panna not in ALL_VALID_PANNAS: continue
                X.append(PannaFeatureBuilder.build_candidate_features(ctx,target_panna,role))
                y.append(1)
                pool=[p for p in ALL_VALID_PANNAS if p!=target_panna]
                negs=random.sample(pool,min(neg_count,len(pool)))
                for neg in negs:
                    X.append(PannaFeatureBuilder.build_candidate_features(ctx,neg,role))
                    y.append(0)
        return X,y

    def fit(self, rows: List[Dict[str,Any]], max_iter: int=100)->bool:
        if not ML_AVAILABLE:
            logger.warning("ML unavailable; using heuristic fallback.")
            return False
        X,y=self._build_training_data(rows)
        if len(X)<200 or len(set(y))<2:
            logger.info("Insufficient training data for ML.")
            return False
        try:
            self.model=HistGradientBoostingClassifier(
                max_iter=max_iter, learning_rate=0.1, max_depth=5, random_state=42, early_stopping=False
            )
            self.model.fit(np.array(X),np.array(y))
            self.fitted=True
            logger.info(f"Trained on {len(X)} samples ({sum(y)} positives).")
            return True
        except Exception as e:
            logger.warning(f"Training failed: {e}")
            return False

    def score_all(self, ctx: Dict[str,Any], role: str="open")->List[Tuple[str,float]]:
        if not self.fitted or self.model is None:
            return self._heuristic_score_all(ctx,role)
        candidates=ALL_VALID_PANNAS
        X=np.array([PannaFeatureBuilder.build_candidate_features(ctx,p,role) for p in candidates])
        try:
            probs=self.model.predict_proba(X)[:,1]
            return sorted(zip(candidates,probs),key=lambda x:x[1],reverse=True)
        except Exception as e:
            logger.warning(f"ML scoring failed: {e}")
            return self._heuristic_score_all(ctx,role)

    @staticmethod
    def _heuristic_score_all(ctx: Dict[str,Any], role: str="open")->List[Tuple[str,float]]:
        y_digit=ctx.get("y_od","") if role=="open" else ctx.get("y_cd","")
        cut_digit=ctx.get("cut_od","") if role=="open" else ctx.get("cut_cd","")
        y_panna=ctx.get("y_op","") if role=="open" else ctx.get("y_cp","")
        mirror_y=y_panna[::-1] if len(y_panna)==3 else ""
        recency=ctx.get("panna_recency",{})
        scored=[]
        for p in ALL_VALID_PANNAS:
            d=PANNA_TO_DIGIT[p]; w=1.0
            if p==y_panna: w*=0.2
            if p==mirror_y: w*=1.3
            if d==y_digit: w*=0.3
            if d==cut_digit: w*=1.4
            if d in ctx.get("cold_digits",[]): w*=1.2
            if p in BEAUTY_PANNAS: w*=0.6
            if p in UGLY_PANNAS: w*=1.3
            if p in recency and recency[p]<=3: w*=0.4
            if len(set(p))==2: w*=CONFIG["DOUBLE_PANNA_BOOST"]
            scored.append((p,w))
        return sorted(scored,key=lambda x:x[1],reverse=True)

def _hash_rows(rows: List[Dict[str,Any]])->str:
    if not rows: return "empty"
    h=hashlib.md5()
    h.update(str(len(rows)).encode())
    sample=json.dumps(rows[-30:],sort_keys=True,default=str)
    h.update(sample.encode())
    return h.hexdigest()[:16]

def get_cached_or_train_ranker(mkt: str, rows: List[Dict[str,Any]], cache_dir: Path, clear_cache: bool, max_iter: int=100)->PannaRanker:
    if not JOBLIB_AVAILABLE or not ML_AVAILABLE:
        ranker=PannaRanker(); ranker.fit(rows,max_iter=max_iter); return ranker
    data_hash=_hash_rows(rows)
    cache_file=cache_dir/f"{mkt}_{data_hash}.joblib"
    if not clear_cache and cache_file.exists():
        try:
            ranker=joblib.load(cache_file)
            logger.debug(f"Cache hit for {mkt}")
            return ranker
        except Exception:
            pass
    ranker=PannaRanker()
    ranker.fit(rows,max_iter=max_iter)
    if ranker.fitted:
        try:
            cache_dir.mkdir(parents=True,exist_ok=True)
            joblib.dump(ranker,cache_file)
            logger.debug(f"Cache saved for {mkt}")
        except Exception as e:
            logger.warning(f"Cache save failed for {mkt}: {e}")
    return ranker

# v51: role-aware panna/digit selection for ML
def select_pannas_and_digits_role_aware(scored: List[Tuple[str,float]], need_pannas: int, need_digits: int,
                                         exclude_digits: Optional[Set[str]]=None,
                                         exclude_pannas: Optional[Set[str]]=None)->Tuple[List[str],List[str]]:
    exclude_digits = exclude_digits or set()
    exclude_pannas = exclude_pannas or set()
    digit_best={}
    for p,score in scored:
        d=PANNA_TO_DIGIT[p]
        adj_score = score * (CONFIG["ROLE_OVERLAP_DIGIT_PENALTY"] if d in exclude_digits else 1.0)
        adj_score = adj_score * (CONFIG["ROLE_OVERLAP_PANNA_PENALTY"] if p in exclude_pannas else 1.0)
        if d not in digit_best or adj_score>digit_best[d][1]:
            digit_best[d]=(p,adj_score)
    top_digits=[d for d,_ in sorted(digit_best.items(),key=lambda x:x[1][1],reverse=True)]
    if len(top_digits)<need_digits:
        missing=[d for d in map(str,range(10)) if d not in top_digits]
        top_digits.extend(missing)
    selected_digits=top_digits[:need_digits]
    selected_pannas=[]; seen=set()
    for p,_ in scored:
        if p in seen: continue
        if PANNA_TO_DIGIT[p] in selected_digits and p not in exclude_pannas:
            selected_pannas.append(p); seen.add(p)
        if len(selected_pannas)>=need_pannas: break
    if len(selected_pannas)<need_pannas:
        for p,_ in scored:
            if p not in seen:
                selected_pannas.append(p); seen.add(p)
            if len(selected_pannas)>=need_pannas: break
    return selected_pannas,selected_digits

def build_jodis_ml(open_digits: List[str], close_digits: List[str], ctx: Dict[str,Any])->List[str]:
    y_j=ctx.get("j_lag1","")
    cut_jodi=""
    if y_j and len(y_j)==2:
        cut_jodi=f"{(int(y_j[0])+5)%10}{(int(y_j[1])+5)%10}"
    cands=[]
    for a in open_digits:
        for b in close_digits:
            j=f"{a}{b}"; w=1.0
            if j==y_j: w*=0.2
            if y_j and j==mirror_jodi(y_j): w*=1.5
            if (int(a)+int(b)) in (10,20): w*=1.35
            if cut_jodi and j==cut_jodi: w*=1.4
            cands.append((j,w))
    cands.sort(key=lambda x:x[1],reverse=True)
    seen=set(); out=[]
    for j,_ in cands:
        if j not in seen:
            seen.add(j); out.append(j)
        if len(out)>=CONFIG["NUM_JODIS"]: break
    return out

def predict_market_ml(mkt: str, rows: List[Dict[str,Any]], asof: datetime.date, cache_dir: Path, clear_cache: bool, max_iter: int=100)->Optional[Dict[str,Any]]:
    if len(rows)<CONFIG["MIN_HISTORY_FOR_ML"]:
        return None
    ctx=PannaFeatureBuilder.build_context(rows,asof)
    ranker=get_cached_or_train_ranker(mkt,rows,cache_dir,clear_cache,max_iter=max_iter)
    
    open_scored=ranker.score_all(ctx,role="open")
    open_pannas,open_digits=select_pannas_and_digits_role_aware(open_scored,CONFIG["NUM_OPEN_PANNAS"],CONFIG["NUM_OPEN_DIGITS"])
    
    close_scored=ranker.score_all(ctx,role="close")
    close_pannas,close_digits=select_pannas_and_digits_role_aware(
        close_scored,CONFIG["NUM_CLOSE_PANNAS"],CONFIG["NUM_CLOSE_DIGITS"],
        exclude_digits=set(open_digits), exclude_pannas=set(open_pannas))
        
    jodis=build_jodis_ml(open_digits,close_digits,ctx)
    
    # FIX: Calculate real ML confidence based on top prediction probabilities
    top_open_prob = open_scored[0][1] if open_scored else 0.0
    top_close_prob = close_scored[0][1] if close_scored else 0.0
    ml_conf = (top_open_prob + top_close_prob) / 2.0
    ml_conf = max(CONFIG["CONFIDENCE_MIN"], min(ml_conf, CONFIG["CONFIDENCE_CAP"]))
    
    open_panna_scores={p:s for p,s in open_scored}
    close_panna_scores={p:s for p,s in close_scored}
    digit_scores=defaultdict(float)
    for p,s in open_scored:
        d=PANNA_TO_DIGIT[p]
        digit_scores[d]=max(digit_scores[d],s)
    for p,s in close_scored:
        d=PANNA_TO_DIGIT[p]
        digit_scores[d]=max(digit_scores[d],s)
    jodi_scores={}
    for j in jodis:
        a,b=j[0],j[1]; w=1.0
        if j==ctx.get("j_lag1",""): w*=0.2
        if (int(a)+int(b)) in (10,20): w*=1.35
        jodi_scores[j]=w
    return {
        "OpenDigits":open_digits,"CloseDigits":close_digits,
        "OpenPannas":open_pannas,"ClosePannas":close_pannas,
        "Jodis":jodis,
        "digit_scores":dict(digit_scores),
        "panna_scores":{**open_panna_scores,**close_panna_scores},
        "jodi_scores":jodi_scores,
        "confidence":round(ml_conf, 2),"confidence_basis":"ml_panna_ranker"
    }

# ------------------------------------------------------------------
# ENSEMBLE MIXER & BACKTEST (v51)
# ------------------------------------------------------------------
def normalize_scores(scores: Dict[str,float])->Dict[str,float]:
    if not scores: return {}
    mx=max(scores.values())
    if mx<=0: return {k:0.0 for k in scores}
    return {k:v/mx for k,v in scores.items()}

def ensemble_blend(psych_pred: Dict[str,Any], ml_pred: Dict[str,Any])->Dict[str,Any]:
    psych_dw=normalize_scores(psych_pred.get("digit_weights",{}))
    ml_dw=normalize_scores(ml_pred.get("digit_scores",{}))
    all_digits=set(psych_dw)|set(ml_dw)
    psych_top=set(psych_pred.get("OpenDigits",[])+psych_pred.get("CloseDigits",[]))
    ml_top=set(ml_pred.get("OpenDigits",[])+ml_pred.get("CloseDigits",[]))
    ensemble_digit_scores={}
    for d in all_digits:
        s=CONFIG["ENSEMBLE_PSYCH_WEIGHT"]*psych_dw.get(d,0.0)+CONFIG["ENSEMBLE_ML_WEIGHT"]*ml_dw.get(d,0.0)
        if d in psych_top and d in ml_top: s+=CONFIG["ENSEMBLE_AGREEMENT_BONUS"]
        ensemble_digit_scores[d]=s

    open_candidates=list(dict.fromkeys(psych_pred.get("OpenDigits",[])+ml_pred.get("OpenDigits",[])))
    open_candidates.sort(key=lambda d: ensemble_digit_scores.get(d,0.0),reverse=True)
    final_open=open_candidates[:CONFIG["NUM_OPEN_DIGITS"]]
    
    close_candidates=list(dict.fromkeys(psych_pred.get("CloseDigits",[])+ml_pred.get("CloseDigits",[])))
    # v51: enforce role separation at ensemble layer too
    close_candidates=[d for d in close_candidates if d not in final_open]
    close_candidates.sort(key=lambda d: ensemble_digit_scores.get(d,0.0),reverse=True)
    final_close=close_candidates[:CONFIG["NUM_CLOSE_DIGITS"]]
    
    # FIX: Fallback to global pool if not enough close candidates
    if len(final_close)<CONFIG["NUM_CLOSE_DIGITS"]:
        all_digits_pool = [str(i) for i in range(10)]
        for d in all_digits_pool:
            if d not in final_open and d not in final_close:
                final_close.append(d)
            if len(final_close)>=CONFIG["NUM_CLOSE_DIGITS"]: break

    psych_pw=normalize_scores(psych_pred.get("panna_scores",{}))
    ml_pw=normalize_scores(ml_pred.get("panna_scores",{}))
    all_pannas=set(psych_pw)|set(ml_pw)
    psych_top_pannas=set(psych_pred.get("OpenPannas",[])+psych_pred.get("ClosePannas",[]))
    ml_top_pannas=set(ml_pred.get("OpenPannas",[])+ml_pred.get("ClosePannas",[]))
    ensemble_panna_scores={}
    for p in all_pannas:
        s=CONFIG["ENSEMBLE_PSYCH_WEIGHT"]*psych_pw.get(p,0.0)+CONFIG["ENSEMBLE_ML_WEIGHT"]*ml_pw.get(p,0.0)
        if p in psych_top_pannas and p in ml_top_pannas: s*=(1.0+CONFIG["ENSEMBLE_AGREEMENT_BONUS"])
        ensemble_panna_scores[p]=s

    final_open_pannas=[]; seen=set()
    for p,_ in sorted(ensemble_panna_scores.items(),key=lambda x:x[1],reverse=True):
        if PANNA_TO_DIGIT[p] in final_open and p not in seen:
            final_open_pannas.append(p); seen.add(p)
        if len(final_open_pannas)>=CONFIG["NUM_OPEN_PANNAS"]: break
    for p,_ in sorted(ensemble_panna_scores.items(),key=lambda x:x[1],reverse=True):
        if p not in seen: final_open_pannas.append(p); seen.add(p)
        if len(final_open_pannas)>=CONFIG["NUM_OPEN_PANNAS"]: break

    final_close_pannas=[]; seen_c=set()
    for p,_ in sorted(ensemble_panna_scores.items(),key=lambda x:x[1],reverse=True):
        if PANNA_TO_DIGIT[p] in final_close and p not in seen_c and p not in final_open_pannas:
            final_close_pannas.append(p); seen_c.add(p)
        if len(final_close_pannas)>=CONFIG["NUM_CLOSE_PANNAS"]: break
    for p,_ in sorted(ensemble_panna_scores.items(),key=lambda x:x[1],reverse=True):
        if p not in seen_c and p not in final_open_pannas: final_close_pannas.append(p); seen_c.add(p)
        if len(final_close_pannas)>=CONFIG["NUM_CLOSE_PANNAS"]: break

    psych_jw=normalize_scores(psych_pred.get("jodi_scores",{}))
    ml_jw=normalize_scores(ml_pred.get("jodi_scores",{}))
    all_jodis=set(psych_jw)|set(ml_jw)
    psych_top_jodis=set(psych_pred.get("Jodis",[]))
    ml_top_jodis=set(ml_pred.get("Jodis",[]))
    ensemble_jodi_scores={}
    for j in all_jodis:
        s=CONFIG["ENSEMBLE_PSYCH_WEIGHT"]*psych_jw.get(j,0.0)+CONFIG["ENSEMBLE_ML_WEIGHT"]*ml_jw.get(j,0.0)
        if j in psych_top_jodis and j in ml_top_jodis: s+=CONFIG["ENSEMBLE_AGREEMENT_BONUS"]
        ensemble_jodi_scores[j]=s
    cart=[]
    for a in final_open:
        for b in final_close:
            j=f"{a}{b}"; base=ensemble_jodi_scores.get(j,0.0)
            if (int(a)+int(b)) in (10,20): base+=0.1
            cart.append((j,base))
    cart.sort(key=lambda x:x[1],reverse=True)
    seen_j=set(); final_jodis=[]
    for j,_ in cart:
        if j not in seen_j: seen_j.add(j); final_jodis.append(j)
        if len(final_jodis)>=CONFIG["NUM_JODIS"]: break

    agreements={
        "digits":list(psych_top&ml_top),
        "pannas":list(psych_top_pannas&ml_top_pannas),
        "jodis":list(psych_top_jodis&ml_top_jodis),
    }
    pc=psych_pred.get("confidence",0.0); mc=ml_pred.get("confidence",0.0)
    conf=0.5*pc+0.5*mc
    if agreements["digits"]: conf+=0.05
    if agreements["pannas"]: conf+=0.05
    if agreements["jodis"]: conf+=0.05
    conf=min(conf,CONFIG["CONFIDENCE_CAP"])
    return {
        "OpenDigits":final_open,"CloseDigits":final_close,
        "OpenPannas":final_open_pannas,"ClosePannas":final_close_pannas,
        "Jodis":final_jodis,
        "confidence":round(max(CONFIG["CONFIDENCE_MIN"],conf),2),
        "confidence_basis":f"ensemble (psych={pc:.2f}, ml={mc:.2f}, agreements={len(agreements['digits'])}d/{len(agreements['pannas'])}p/{len(agreements['jodis'])}j)",
        "agreements":agreements,
        "v51_signals":psych_pred.get("v51_signals",{}),
        "psych_only":{k:psych_pred[k] for k in ["OpenDigits","CloseDigits","OpenPannas","ClosePannas","Jodis"]},
        "ml_only":{k:ml_pred.get(k,[]) for k in ["OpenDigits","CloseDigits","OpenPannas","ClosePannas","Jodis"]},
    }

def run_ensemble(psych_pred: Dict[str,Any], ml_pred: Optional[Dict[str,Any]])->Dict[str,Any]:
    if ml_pred is None:
        return {
            **psych_pred,
            "confidence_basis":psych_pred.get("confidence_basis","")+" | ML skipped (insufficient data)",
            "agreements":{"digits":[],"pannas":[],"jodis":[]},
            "psych_only":{k:psych_pred[k] for k in ["OpenDigits","CloseDigits","OpenPannas","ClosePannas","Jodis"]},
            "ml_only":{k:[] for k in ["OpenDigits","CloseDigits","OpenPannas","ClosePannas","Jodis"]},
        }
    return ensemble_blend(psych_pred,ml_pred)

def find_actual_for_date(idx: MarketIndex, target_date: datetime.date)->Optional[Dict[str,str]]:
    r=idx.row_for_date(target_date)
    if r is None: return None
    res=extract_day_result(r)
    if not (res["od"] or res["cd"] or res["j"] or res["op"] or res["cp"]): return None
    return res

def load_state(path: Path)->Dict[str,Any]:
    data=load_json(path)
    if isinstance(data,dict) and "markets" in data: return data
    return {"markets":{},"schema_version":2,"signal_registry":{},"first_run_done":{}}

def record_prediction(state: Dict[str,Any], mkt: str, pred: Dict[str,Any], asof: datetime.date)->None:
    markets=state.setdefault("markets",{})
    mkt_list=markets.setdefault(mkt,[])
    entry={"date":asof.isoformat(),
           "predictions":{k:v for k,v in pred.items() if k not in ("agreements","psych_only","ml_only","v51_signals","digit_weights","panna_scores","jodi_scores","panna_attr","jodi_attr")},
           "actual":None,
           "v51_signals":pred.get("v51_signals",{})}
    mkt_list.append(entry)
    cap=CONFIG["STATE_HISTORY_CAP_PER_MARKET"]
    if len(mkt_list)>cap: mkt_list[:len(mkt_list)-cap]=[]

def backtest_update(state: Dict[str,Any], indexes: Dict[str,MarketIndex])->Dict[str,Any]:
    scored_counts={}; markets=state.get("markets",{})
    for mkt,entries in markets.items():
        idx=indexes.get(mkt)
        if idx is None: continue
        n_scored=0
        for entry in entries:
            if entry.get("actual") is not None: continue
            d=datetime.date.fromisoformat(entry["date"]) if isinstance(entry.get("date"),str) else None
            if d is None: continue
            actual=find_actual_for_date(idx,d)
            if actual: entry["actual"]=actual; n_scored+=1
        scored_counts[mkt]=n_scored
    return scored_counts

def walk_forward_backtest(mkt: str, rows: List[Dict[str,Any]], idx: MarketIndex, days: int=15)->Dict[str,Any]:
    dated=[(parse_date(r.get("Date") or r.get("date")),r) for r in rows if parse_date(r.get("Date") or r.get("date"))]
    dated.sort(key=lambda x:x[0])
    if len(dated)<days+5: return {"error":"insufficient_history","days_available":len(dated)}
    od_hits=cd_hits=op_hits=cp_hits=j_hits=any_hits=n_tests=0
    dow_hits=defaultdict(int); dow_total=defaultdict(int)
    trend_hits={"up":0,"down":0,"flat":0}; trend_total={"up":0,"down":0,"flat":0}
    
    # FIX: Use a dedicated backtest cache directory instead of polluting root
    bt_cache_dir = Path("./ml_cache_backtest")
    bt_cache_dir.mkdir(exist_ok=True)
    
    for i in range(max(0,len(dated)-days),len(dated)):
        test_date=dated[i][0]
        prior_rows=[r for _,r in dated[:i]]
        if len(prior_rows)<10: continue
        prior_idx=MarketIndex(prior_rows)
        try:
            psych=predict_market_psych(mkt,prior_idx,test_date,{"markets":{},"signal_registry":{}})
        except Exception as e:
            logger.debug(f"Backtest psych error {test_date}: {e}"); continue
        ml=None
        if len(prior_rows)>=CONFIG["MIN_HISTORY_FOR_ML"]:
            try: ml=predict_market_ml(mkt,prior_rows,test_date,bt_cache_dir,False,max_iter=CONFIG["BACKTEST_ML_MAX_ITER"])
            except Exception as e: logger.debug(f"Backtest ML error {test_date}: {e}")
        pred=run_ensemble(psych,ml)
        actual=extract_day_result(dated[i][1]); n_tests+=1
        hit_any=False
        if actual["od"] in pred.get("OpenDigits",[]): od_hits+=1; hit_any=True
        if actual["cd"] in pred.get("CloseDigits",[]): cd_hits+=1; hit_any=True
        if actual["op"] in pred.get("OpenPannas",[]): op_hits+=1; hit_any=True
        if actual["cp"] in pred.get("ClosePannas",[]): cp_hits+=1; hit_any=True
        if actual["j"] in pred.get("Jodis",[]): j_hits+=1; hit_any=True
        if hit_any: any_hits+=1
        dow=test_date.weekday(); dow_total[dow]+=1
        if hit_any: dow_hits[dow]+=1
        if i>0:
            prev_od=extract_day_result(dated[i-1][1]).get("od","")
            if prev_od.isdigit() and actual["od"].isdigit():
                diff=int(actual["od"])-int(prev_od)
                bucket="up" if diff>0 else ("down" if diff<0 else "flat")
                trend_total[bucket]+=1
                if hit_any: trend_hits[bucket]+=1
    dow_acc={int(k):dow_hits[k]/dow_total[k] for k in dow_total if dow_total[k]>0}
    trend_acc={k:(trend_hits[k]/trend_total[k] if trend_total[k]>0 else 0.0) for k in trend_total}
    return {"days_tested":n_tests,
            "open_digit_acc":od_hits/n_tests if n_tests else 0.0,
            "close_digit_acc":cd_hits/n_tests if n_tests else 0.0,
            "open_panna_acc":op_hits/n_tests if n_tests else 0.0,
            "close_panna_acc":cp_hits/n_tests if n_tests else 0.0,
            "jodi_acc":j_hits/n_tests if n_tests else 0.0,
            "any_hit_acc":any_hits/n_tests if n_tests else 0.0,
            "dow_acc":dow_acc,"trend_acc":trend_acc}

def to_app_structure(mkt: str, pred: Dict[str,Any], asof: datetime.date)->Dict[str,Any]:
    return {
        "market":mkt,
        "predictions":{
            "Date":asof.strftime("%Y-%m-%d"),
            "Open Digits":pred.get("OpenDigits",[]),
            "Close Digits":pred.get("CloseDigits",[]),
            "Open Pannas":pred.get("OpenPannas",[]),
            "Close Pannas":pred.get("ClosePannas",[]),
            "Jodis":pred.get("Jodis",[]),
        },
        "confidence":pred.get("confidence",0.0),
        "confidence_basis":pred.get("confidence_basis",""),
        "ensemble":{
            "agreements":pred.get("agreements",{"digits":[],"pannas":[],"jodis":[]}),
            "psych_only":pred.get("psych_only",{}),
            "ml_only":pred.get("ml_only",{}),
        },
        "v51_signals":pred.get("v51_signals",{}),
    }

def run_once(args, all_hist: Dict[str,List[Dict[str,Any]]], state: Dict[str,Any], output_dir: Path)->Dict[str,Any]:
    asof=datetime.date.today()
    if args.test_date: asof=parse_date(args.test_date) or asof
    indexes={mkt:MarketIndex(rows) for mkt,rows in all_hist.items()}
    cache_dir=output_dir/"ml_cache"; cache_dir.mkdir(parents=True,exist_ok=True)
    scored_counts=backtest_update(state,indexes)
    if scored_counts: logger.info(f"Incremental backtest scored: {scored_counts}")
    first_run_stats={}
    if not args.skip_first_backtest:
        state.setdefault("first_run_done",{})
        for mkt,rows in all_hist.items():
            if state["first_run_done"].get(mkt): continue
            if len(rows)<CONFIG["BACKTEST_DAYS_FIRST_RUN"]+CONFIG["MIN_HISTORY_FOR_ML"]:
                logger.info(f"Skipping first-run backtest for {mkt}: insufficient history")
                state["first_run_done"][mkt]=True; continue
            logger.info(f"Running first-run backtest for {mkt}...")
            stats=walk_forward_backtest(mkt,rows,indexes[mkt],CONFIG["BACKTEST_DAYS_FIRST_RUN"])
            first_run_stats[mkt]=stats; state["first_run_done"][mkt]=True
            logger.info(f"First-run backtest {mkt}: {stats}")
    results=[]
    for mkt,rows in all_hist.items():
        idx=indexes.get(mkt)
        if idx is None or len(idx)==0: logger.warning(f"No index for {mkt}"); continue
        try: psych=predict_market_psych(mkt,idx,asof,state)
        except Exception as e: logger.error(f"Psych predict failed for {mkt}: {e}"); continue
        ml=None
        try: ml=predict_market_ml(mkt,rows,asof,cache_dir,args.clear_cache)
        except Exception as e: logger.warning(f"ML predict failed for {mkt}: {e}")
        pred=run_ensemble(psych,ml)
        record_prediction(state,mkt,pred,asof)
        results.append(to_app_structure(mkt,pred,asof))
    backtest_summary={}
    for mkt,entries in state.get("markets",{}).items():
        scored=[e for e in entries if e.get("actual") is not None]
        if not scored: continue
        acc=rolling_accuracy(entries,CONFIG["ACCURACY_LOOKBACK_DAYS"])
        backtest_summary[mkt]={"total_predictions":len(entries),"scored":len(scored),"rolling":acc}
    return {"meta":{"generated_at":iso_timestamp(),"engine":"v51_unified","schema_version":2,
                    "prediction_date":asof.isoformat(),"backtest_newly_scored":scored_counts,
                    "first_run_backtest":first_run_stats,"rolling_backtest_summary":backtest_summary},
            "markets":results}

def selftest():
    print("=== V51 Unified Self-Test ===")
    dummy_rows=[
        {"Date":"2024-01-01","Open Digit":"1","Close Digit":"2","Open Panna":"100","Close Panna":"110","Jodi":"12"},
        {"Date":"2024-01-02","Open Digit":"3","Close Digit":"4","Open Panna":"300","Close Panna":"400","Jodi":"34"},
        {"Date":"2024-01-03","Open Digit":"5","Close Digit":"6","Open Panna":"500","Close Panna":"600","Jodi":"56"},
        {"Date":"2024-01-04","Open Digit":"7","Close Digit":"8","Open Panna":"700","Close Panna":"800","Jodi":"78"},
        {"Date":"2024-01-05","Open Digit":"9","Close Digit":"0","Open Panna":"900","Close Panna":"190","Jodi":"90"},
    ]
    idx=MarketIndex(dummy_rows); state={"markets":{},"signal_registry":{}}
    pred=predict_market_psych("TEST",idx,datetime.date(2024,1,6),state)
    print("Psych:",json.dumps({k:pred[k] for k in ["OpenDigits","CloseDigits","OpenPannas","ClosePannas","Jodis","confidence"]},indent=2,default=str))
    ml=predict_market_ml("TEST",dummy_rows,datetime.date(2024,1,6),Path("."),False)
    print("ML:",json.dumps({k:ml[k] for k in ["OpenDigits","CloseDigits","OpenPannas","ClosePannas","Jodis"]} if ml else "None",indent=2,default=str))
    ens=run_ensemble(pred,ml)
    print("Ensemble role-separated:",
          "OPEN_DIGITS=",ens["OpenDigits"],"CLOSE_DIGITS=",ens["CloseDigits"],
          "overlap=",set(ens["OpenDigits"])&set(ens["CloseDigits"]))
    print("Self-test passed.")

def build_arg_parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="SattaMatkaAI v51 Unified (Psych + ML Ensemble, Role-Aware)")
    p.add_argument("--history",dest="history_file",default=None)
    p.add_argument("--output-dir",dest="output_dir",default=None)
    p.add_argument("--state",dest="state_file",default=None)
    p.add_argument("--prod",action="store_true")
    p.add_argument("--test",action="store_true")
    p.add_argument("--verbose",action="store_true")
    p.add_argument("--clear-cache",action="store_true")
    p.add_argument("--skip-first-backtest",action="store_true")
    p.add_argument("--test-date",default=None)
    p.add_argument("--selftest",action="store_true")
    return p

def main():
    p=build_arg_parser()
    # Backtest lab compatibility: accept --date as an alias for --test-date.
    known, remaining = p.parse_known_args()
    if remaining:
        p.add_argument("--date", dest="date_alias", default=None,
                       help="Alias for --test-date (used by run_backtest.py)")
        args = p.parse_args()
        if args.date_alias and not args.test_date:
            args.test_date = args.date_alias
    else:
        args = known
    if args.selftest: selftest(); return
    history_path=_resolve_path("HISTORY_FILE",args.history_file)
    output_dir=_resolve_path("OUTPUT_DIR",args.output_dir)
    state_path=_resolve_path("STATE_FILE",args.state_file)
    # FIX: Respect env vars for log file
    log_file=_resolve_path("LOG_FILE", None)
    setup_logging(log_file,CONFIG["ENABLE_LOGGING"],args.verbose)
    logger.info(f"Starting v51_unified | history={history_path} | state={state_path}")
    if CONFIG["TEST_MODE"]:
        random.seed(CONFIG["TEST_SEED"]); logger.info("TEST_MODE active with fixed seed.")
    all_hist=load_market_history(history_path)
    try: validate_history_schema(all_hist)
    except Exception as e: logger.error(f"History validation failed: {e}"); sys.exit(1)
    state=load_state(state_path)
    # Backtest lab: a past --test-date means "predict for that date only".
    # Skip the expensive first-run walk-forward backtest (it retrains ML per day
    # and can exceed 10 min); it is not needed to produce the dated prediction file.
    _asof = parse_date(args.test_date) if args.test_date else None
    if _asof is not None and _asof < datetime.date.today() and not args.skip_first_backtest:
        args.skip_first_backtest = True
        logger.info("Backtest mode: auto-skipping first-run walk-forward backtest for past date.")
    output=run_once(args,all_hist,state,output_dir)
    if args.prod: out_path=Path(_DEFAULTS["TODAY_PROD_FILE"])
    elif args.test: out_path=Path(_DEFAULTS["TODAY_TEST_FILE"])
    else: out_path=output_dir/f"todays_predictions_v51_{today_str()}.json"
    save_json_atomic(out_path,output)
    logger.info(f"Output written to {out_path} ({len(output['markets'])} markets)")
    save_json_atomic(state_path,state)
    logger.info(f"State saved to {state_path}")
    if args.verbose: print(json.dumps(output,indent=2,default=str))
    # ==== BACKTEST LAB 1-LINER: unique dated output -> D:\backtest\predictions_<YYYY-MM-DD>_v51_unified.json ====
    # NOTE: must run INSIDE main() — the module-level copy below never executes,
    # because __name__ is "__main__" when the script is run directly.
    try:
        _bt_write_dated(output, args)
    except Exception as _bt_e:
        print(f"[BACKTEST] skipped: {_bt_e}")


def _bt_write_dated(out_obj, args=None):
    import re as _bt_re
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
        _argv = list(sys.argv)
        for _i, _a in enumerate(_argv):
            if _a.startswith("--test-date") or _a.startswith("--date"):
                _v = _a.split("=", 1)[1] if "=" in _a else (_argv[_i+1] if _i+1 < len(_argv) else "")
                if _bt_re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(_v)):
                    ds = _v; break
    if not _bt_re.fullmatch(r"\d{4}-\d{2}-\d{2}", ds):
        _td = getattr(args, "test_date", None)
        if _td and _bt_re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(_td)):
            ds = str(_td)
    if not _bt_re.fullmatch(r"\d{4}-\d{2}-\d{2}", ds):
        ds = datetime.date.today().isoformat()
    _bt_dir = Path(os.environ.get("LAB_DIR") or ("D:\\backtest" if (os.name == "nt" or os.path.splitdrive("D:\\")[0]) else Path(__file__).resolve().parent))
    _bt_dir.mkdir(parents=True, exist_ok=True)
    _p = _bt_dir / f"predictions_{ds}_v51_unified.json"
    _t = _p.with_suffix(".json.tmp")
    _t.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(_t, _p)
    print(f"[BACKTEST] wrote {_p}")
# ==== END BACKTEST LAB 1-LINER (active copy, called from main) ====

if __name__=="__main__":
    main()

# ==== BACKTEST LAB 1-LINER: unique dated output -> D:\backtest\predictions_<YYYY-MM-DD>_v51_unified.json ====
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
        _p = _bt_dir / f"predictions_{ds}_v51_unified.json"
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
