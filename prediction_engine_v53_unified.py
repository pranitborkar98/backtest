"""Deterministic forecasts with chronological evaluation and an immutable issue ledger.

Python 3.10+, standard library only. Run --help for options. No uploading or betting.
Historical replay is a simulation; only archived, issued forecasts are live evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import hashlib
from itertools import combinations_with_replacement
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import zlib

ROOT = Path(__file__).resolve().parent
VERSION = "v53.0-unified"
IST = timezone(timedelta(hours=5, minutes=30))
FIELDS = {"od": "Open Digits", "cd": "Close Digits", "op": "Open Pannas",
          "cp": "Close Pannas", "j": "Jodis"}
ALIASES = {"od": ("opendigit", "od"), "cd": ("closedigit", "cd"),
           "op": ("openpanna", "op"), "cp": ("closepanna", "cp"), "j": ("jodi", "j")}
PANNAS = tuple(sorted("".join(p) for p in combinations_with_replacement("1234567890", 3)))
UNIVERSES = {"od": tuple(map(str, range(10))), "cd": tuple(map(str, range(10))),
             "op": PANNAS, "cp": PANNAS, "j": tuple(f"{i:02d}" for i in range(100))}
INDEX = {k: {v: i for i, v in enumerate(vals)} for k, vals in UNIVERSES.items()}
EXPERTS = ("uniform", "frequency", "recent_60", "weekday", "transition", "cognitive")
PICKS = {"od": 3, "cd": 3, "op": 8, "cp": 8, "j": 9}

# Ported, testable parts of v33/v50/v51. These are signal families, not claims
# about operator intent: every score is computed only from rows before asof.
MICRO_DAYS, MESO_DAYS, MACRO_DAYS = 2, 5, 15
MARKOV_DAYS, DUE_DAYS = 365, 20
FIBONACCI_GAPS = {5, 8, 13}
BEAUTY_PANNAS = {"111", "222", "333", "444", "555", "666", "777", "888", "999",
                 "123", "234", "345", "456", "567", "678", "789", "890", "901",
                 "135", "246", "357", "468", "579", "680", "791", "802", "913",
                 "147", "258", "369", "470", "581", "692", "703", "814", "925"}


def _normalize_probs(values):
    values = [max(float(v), 0.0) for v in values]
    total = sum(values)
    return [v / total for v in values] if total else [1.0 / len(values)] * len(values)


def _panna_digit(panna):
    return str(sum(map(int, panna)) % 10)


def cognitive_probabilities(rows, asof, category):
    """High-end rule expert from v33/v50, returned as a full probability vector."""
    vals = UNIVERSES[category]
    before = [r for r in rows if r["date"] < asof.isoformat()]
    recent = before[-MACRO_DAYS:]
    medium = before[-MESO_DAYS:]
    micro = before[-MICRO_DAYS:]
    weights = {v: 1.0 for v in vals}

    def add_counts(source, field, factor):
        counts = Counter(r.get(field) for r in source if r.get(field) in weights)
        for v, n in counts.items():
            weights[v] *= 1.0 + factor * n

    field = {"od": "od", "cd": "cd", "op": "op", "cp": "cp", "j": "j"}[category]
    add_counts(before[-45:], field, 0.10)
    add_counts(recent, field, 0.16)
    add_counts(medium, field, 0.24)
    add_counts(micro, field, 0.35)
    last = before[-1] if before else {}
    prev = before[-2] if len(before) > 1 else {}

    if category in ("od", "cd"):
        # Markov transitions: OC/CO plus self transitions, as in v33.
        source_field = "od" if category == "cd" else "cd"
        transition_counts = Counter()
        for a, b in zip(before[-MARKOV_DAYS:-1], before[-MARKOV_DAYS + 1:]):
            if a.get(source_field) in (None, "") or b.get(field) in (None, ""):
                continue
            transition_counts[b[field]] += int(a[source_field] == last.get(source_field))
        for v, n in transition_counts.items():
            weights[v] *= 1.0 + 0.35 * n
        # Due/cold regression and Fibonacci cycle candidates.
        seen = {r.get(field) for r in before[-DUE_DAYS:] if r.get(field) in weights}
        for v in vals:
            if v not in seen:
                weights[v] *= 1.30
        positions = [i for i, r in enumerate(before[-DUE_DAYS:]) if r.get(field) == v]
        for v in vals:
            pos = [i for i, r in enumerate(before[-DUE_DAYS:]) if r.get(field) == v]
            if len(pos) >= 2 and any((pos[i] - pos[i - 1]) in FIBONACCI_GAPS for i in range(1, len(pos))):
                weights[v] *= 1.25
        # Yesterday cut/neighbor/mirror and anti-herd suppression.
        if last.get(field) in vals:
            d = int(last[field])
            for v in vals:
                if v == str((d + 5) % 10): weights[v] *= 1.25
                if v in (str((d + 1) % 10), str((d - 1) % 10)): weights[v] *= 1.15
        hot = {v for v, n in Counter(r.get(field) for r in recent).items() if n >= 4}
        for v in hot:
            weights[v] *= 0.65
    elif category in ("op", "cp"):
        # Panna-level psychology and sum-mod consistency.
        y = last.get(field)
        for p in vals:
            d = _panna_digit(p)
            if last.get("od" if category == "op" else "cd") == d:
                weights[p] *= 0.72  # avoid blindly repeating yesterday's digit
            if y and p == y[::-1]:
                weights[p] *= 1.30
            if p in BEAUTY_PANNAS:
                weights[p] *= 0.70
            if len(set(p)) == 2:
                weights[p] *= 1.12
    else:
        # Jodi transition and cross-digit structure.
        if last.get("j"):
            mirror = f"{(int(last['j'][0]) + 5) % 10}{(int(last['j'][1]) + 5) % 10}"
            cut = f"{(int(last['j'][0]) + 5) % 10}{(int(last['j'][1]) + 5) % 10}"
            for j in vals:
                if j == mirror: weights[j] *= 1.25
                if j == cut: weights[j] *= 1.20
        for j in vals:
            if int(j[0]) + int(j[1]) in (10, 20):
                weights[j] *= 1.12
    return _normalize_probs([weights[v] for v in vals])


def key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def market_key(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def parse_date(value):
    if not value:
        return None
    value = str(value).strip()
    for fmt, part in (("%Y-%m-%d", value[:10]), ("%d/%m/%Y", value), ("%Y%m%d", value)):
        try:
            return datetime.strptime(part, fmt).date()
        except ValueError:
            pass
    return None


def number(value, width):
    if isinstance(value, bool) or value is None:
        return None
    s = str(value).strip()
    if not re.fullmatch(r"[0-9]{1," + str(width) + r"}", s):
        return None
    return s.zfill(width)


def normalize_result(raw, stats):
    raw = {key(k): v for k, v in raw.items()}
    result = {}
    for cat, aliases in ALIASES.items():
        value = next((raw[a] for a in aliases if a in raw and raw[a] is not None), None)
        width = 3 if cat in ("op", "cp") else 2 if cat == "j" else 1
        val = number(value, width)
        if val is not None and val not in INDEX[cat]:
            stats["invalid_" + cat] += 1
            val = None
        result[cat] = val
    return result


def reconcile(result, stats):
    result = dict(result)
    blocked = set()
    for p, d in (("op", "od"), ("cp", "cd")):
        if result[p]:
            derived = str(sum(map(int, result[p])) % 10)
            if result[d] is not None and result[d] != derived:
                stats["panna_digit_conflicts"] += 1
                result[p] = result[d] = result["j"] = None
                blocked.add(d)
            elif result[d] is None:
                result[d] = derived
    if result["j"]:
        if any(result[d] is not None and result[d] != result["j"][i]
               for i, d in enumerate(("od", "cd"))):
            stats["jodi_digit_conflicts"] += 1
            result["j"] = None
        else:
            for i, d in enumerate(("od", "cd")):
                if d not in blocked and result[d] is None:
                    result[d] = result["j"][i]
    if result["od"] is not None and result["cd"] is not None:
        result["j"] = result["od"] + result["cd"]
    return result


def load_history(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("History must be a nonempty market -> records mapping.")
    stats = Counter()
    grouped = defaultdict(lambda: defaultdict(list))
    for market, rows in raw.items():
        if not isinstance(rows, list):
            stats["non_list_markets"] += 1
            continue
        m = market_key(market)
        if not m:
            continue
        for row in rows:
            stats["raw_rows"] += 1
            if not isinstance(row, dict):
                stats["invalid_rows"] += 1
                continue
            dt = parse_date(row.get("Date", row.get("date")))
            if dt is None:
                stats["invalid_dates"] += 1
                continue
            grouped[m][dt].append(normalize_result(row, stats))
    markets = {}
    for market, days in sorted(grouped.items()):
        normalized = []
        for dt, entries in sorted(days.items()):
            stats["duplicate_rows"] += len(entries) - 1
            values = {}
            conflicting = set()
            for cat in FIELDS:
                found = {r[cat] for r in entries if r[cat] is not None}
                values[cat] = next(iter(found)) if len(found) == 1 else None
                if len(found) > 1:
                    conflicting.add(cat)
                    stats["conflicting_duplicate_fields"] += 1
            # Quarantine dependent fields too; never derive back a disputed value.
            for p, d in (("op", "od"), ("cp", "cd")):
                if p in conflicting or d in conflicting:
                    values[p] = values[d] = values["j"] = None
            if "j" in conflicting:
                values = dict.fromkeys(FIELDS)
            values = reconcile(values, stats)
            if any(v is not None for v in values.values()):
                normalized.append({"date": dt.isoformat(), **values})
            else:
                stats["empty_or_quarantined_days"] += 1
        if normalized:
            markets[market] = normalized
    if not markets:
        raise ValueError("No usable dated history rows.")
    stats["markets"] = len(markets)
    stats["usable_days"] = sum(map(len, markets.values()))
    return markets, dict(stats)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Learner:
    """Per-market, per-target categorical experts; feedback uses proper log loss."""

    def __init__(self, category):
        self.category = category
        self.size = len(UNIVERSES[category])
        self.counts = Counter()
        self.recent = deque()
        self.recent_counts = Counter()
        self.weekdays = defaultdict(Counter)
        self.transitions = defaultdict(Counter)
        self.previous = None
        self.losses = [0.0] * len(EXPERTS)
        self.updates = 0

    def observe(self, value, weekday):
        if value is None:
            # Do not infer transitions across a known missing result.
            self.previous = None
            return
        self.counts[value] += 1
        self.weekdays[weekday][value] += 1
        if self.previous is not None:
            self.transitions[self.previous][value] += 1
        self.previous = value
        self.recent.append(value)
        self.recent_counts[value] += 1
        if len(self.recent) > 60:
            self.recent_counts[self.recent.popleft()] -= 1

    def probabilities(self, weekday):
        vals = UNIVERSES[self.category]
        uniform = [1.0 / self.size] * self.size

        def smooth(counts, prior, strength):
            denom = sum(counts.values()) + strength
            return [(counts[v] + strength * prior[i]) / denom for i, v in enumerate(vals)]

        freq = smooth(self.counts, uniform, 20.0)
        return [uniform, freq, smooth(self.recent_counts, uniform, 20.0),
                smooth(self.weekdays[weekday], freq, 30.0),
                smooth(self.transitions[self.previous], freq, 30.0)]

    def weights(self):
        minimum = min(self.losses)
        raw = [math.exp(-0.2 * (v - minimum)) for v in self.losses]
        total = sum(raw)
        return [0.98 * v / total + 0.02 / len(raw) for v in raw]

    def blend(self, probabilities):
        weights = self.weights()
        return [sum(w * p[i] for w, p in zip(weights, probabilities)) for i in range(self.size)]

    def update(self, probabilities, actual):
        if actual is None:
            return
        # v52 ledger rows contain five experts. Preserve compatibility by
        # reconstructing the new cognitive slot instead of shrinking losses.
        probabilities = list(probabilities)
        if len(probabilities) < len(EXPERTS):
            fallback = self.probabilities(0)
            probabilities.extend(fallback[len(probabilities):len(EXPERTS)])
        probabilities = probabilities[:len(EXPERTS)]
        ix = INDEX[self.category][actual]
        self.losses = [0.99 * loss - math.log(max(probs[ix], 1e-15))
                       for loss, probs in zip(self.losses, probabilities)]
        self.updates += 1


def top(category, probabilities, allowed=None):
    vals = UNIVERSES[category]
    eligible = [i for i, v in enumerate(vals) if allowed is None or allowed(v)]
    ranked = sorted(eligible, key=lambda i: (-probabilities[i], vals[i]))
    return [vals[i] for i in ranked[:PICKS[category]]]


def select(probabilities):
    picks = {cat: top(cat, probabilities[cat]) for cat in ("od", "cd")}
    for cat, digit in (("op", "od"), ("cp", "cd")):
        picks[cat] = top(cat, probabilities[cat],
                         lambda p: str(sum(map(int, p)) % 10) in picks[digit])
    picks["j"] = top("j", probabilities["j"], lambda j: j[0] in picks["od"] and j[1] in picks["cd"])
    return picks


class Metrics:
    def __init__(self):
        self.data = {cat: Counter() for cat in FIELDS}

    def add(self, picks, actual, probs=None):
        for cat in FIELDS:
            value = actual.get(cat)
            if value is None:
                continue
            d = self.data[cat]
            choices = list(dict.fromkeys(picks.get(cat, [])))
            d["n"] += 1
            d["hits"] += int(value in choices)
            d["uniform_expected_hits"] += len(choices) / len(UNIVERSES[cat])
            if probs:
                d["loss_sum"] -= math.log(max(probs[cat][INDEX[cat][value]], 1e-15))
                d["loss_n"] += 1

    def report(self):
        out = {}
        for cat, d in self.data.items():
            n = d["n"]
            if not n:
                out[cat] = {"samples": 0, "hit_rate": None}
                continue
            rate = d["hits"] / n
            z = 1.96
            denom = 1 + z * z / n
            center = (rate + z * z / (2 * n)) / denom
            radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
            out[cat] = {"samples": n, "hits": d["hits"], "hit_rate": rate,
                        "uniform_pick_coverage": d["uniform_expected_hits"] / n,
                        "wilson_95_interval": [max(0.0, center - radius), min(1.0, center + radius)],
                        "mean_log_loss": d["loss_sum"] / d["loss_n"] if d["loss_n"] else None}
        return out


@contextmanager
def ledger(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS forecasts (market TEXT, date TEXT, payload BLOB NOT NULL, PRIMARY KEY(market,date))")
        db.execute("CREATE TABLE IF NOT EXISTS info (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def saved_forecasts(db, market):
    return {dt: json.loads(zlib.decompress(blob)) for dt, blob in
            db.execute("SELECT date,payload FROM forecasts WHERE market=? ORDER BY date", (market,))}


def insert_forecast(db, market, dt, payload):
    blob = zlib.compress(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), 6)
    db.execute("INSERT INTO forecasts(market,date,payload) VALUES (?,?,?)", (market, dt, blob))


def train_market(rows, asof, archived, validation_days=180, warmup=60):
    # Every model observation, weight update and metric is strictly before asof.
    prior = [r for r in rows if r["date"] < asof.isoformat()]
    learners = {cat: Learner(cat) for cat in FIELDS}
    evaluation = {name: Metrics() for name in ("adaptive", "frequency", "recent_60")}
    eval_start = max(warmup, len(prior) - validation_days)
    archived_feedback = Counter()
    for i, row in enumerate(prior):
        dow = date.fromisoformat(row["date"]).weekday()
        if i >= eval_start:
            expert_probs = {cat: learner.probabilities(dow) for cat, learner in learners.items()}
            for cat in FIELDS:
                expert_probs[cat].append(cognitive_probabilities(prior[:i], date.fromisoformat(row["date"]), cat))
            blended = {cat: learners[cat].blend(p) for cat, p in expert_probs.items()}
            for name, probs in (("adaptive", blended),
                                ("frequency", {cat: p[1] for cat, p in expert_probs.items()}),
                                ("recent_60", {cat: p[2] for cat, p in expert_probs.items()})):
                evaluation[name].add(select(probs), row, probs)
            issued = archived.get(row["date"])
            for cat, learner in learners.items():
                # Score the exact probabilities saved at issue time when available.
                # This replaces replay feedback for that date; it never adds a second update.
                feedback = issued["expert_probabilities"][cat] if issued else expert_probs[cat]
                learner.update(feedback, row[cat])
                if issued and row[cat] is not None:
                    archived_feedback[cat] += 1
        for cat, learner in learners.items():
            learner.observe(row[cat], dow)
    expert_probs = {cat: learner.probabilities(asof.weekday()) for cat, learner in learners.items()}
    for cat in FIELDS:
        expert_probs[cat].append(cognitive_probabilities(prior, asof, cat))
    blended = {cat: learners[cat].blend(p) for cat, p in expert_probs.items()}
    return {"picks": select(blended), "probabilities": blended, "expert_probabilities": expert_probs,
            "weights": {cat: dict(zip(EXPERTS, learner.weights())) for cat, learner in learners.items()},
            "weight_updates": {cat: learner.updates for cat, learner in learners.items()},
            "archived_feedback_fields": dict(archived_feedback),
            "history_rows": len(prior), "history_last_date": prior[-1]["date"] if prior else None,
            "training_fingerprint": digest(prior),
            "evaluation_first_date": prior[eval_start]["date"] if eval_start < len(prior) else None,
            "evaluation_last_date": prior[-1]["date"] if eval_start < len(prior) else None,
            "evaluation": {name: value.report() for name, value in evaluation.items()}}


def score_issued(archived, rows, asof):
    # Current-day settlements may be reported; training still excludes that day.
    by_date = {r["date"]: r for r in rows if r["date"] <= asof.isoformat()}
    metrics = Metrics()
    settled = {}
    records = []
    for dt, issued in sorted(archived.items()):
        actual = by_date.get(dt)
        if actual is None:
            continue
        metrics.add(issued["picks"], actual, issued["probabilities"])
        for cat in FIELDS:
            if actual[cat] is not None:
                settled[dt + "/" + cat] = actual[cat]
        records.append({"date": dt, "issued_at": issued["issued_at"], "actual": actual,
                        "picks": issued["picks"],
                        "hits": {cat: actual[cat] in issued["picks"][cat] if actual[cat] is not None else None
                                 for cat in FIELDS}})
    return metrics.report(), settled, records


def prediction_entries(obj):
    """Read known app JSON and state-ledger shapes; never execute old engines."""
    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
    items = obj.get("markets", []) if isinstance(obj, dict) else obj
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                yield item.get("market", item.get("Market", "")), item, meta
    elif isinstance(items, dict):
        for market, entries in items.items():
            for item in entries if isinstance(entries, list) else [entries]:
                if isinstance(item, dict):
                    yield market, item, meta


def load_prediction_picks(path):
    """Load one legacy prediction JSON without importing or executing its engine."""
    obj = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    found = {}
    for market, item, meta in prediction_entries(obj):
        pred = item.get("predicted", item.get("predictions", item))
        if not isinstance(pred, dict):
            continue
        dt = parse_date(item.get("date") or pred.get("Date") or meta.get("prediction_date"))
        if not dt:
            continue
        normalized = {key(k): v for k, v in pred.items()}
        picks = {}
        for cat, label in FIELDS.items():
            raw = normalized.get(key(label), [])
            if isinstance(raw, list) and len(raw) == 2 and isinstance(raw[0], list) and isinstance(raw[1], dict):
                raw = raw[0]
            if not isinstance(raw, list):
                raw = []
            width = 3 if cat in ("op", "cp") else 2 if cat == "j" else 1
            picks[cat] = sorted({v for val in raw if (v := number(val, width)) in INDEX[cat]})
        if any(picks.values()):
            found[(market_key(market), dt.isoformat())] = picks
    return found


def load_legacy_v10_archive(root):
    """Read dated v10 files for measurement only; do not execute v10."""
    found = {}
    paths = sorted((Path(root) / "output").glob("predictions_*_v10.json"))
    paths += sorted((Path(root) / "assets").glob("predictions_*_v10.json"))
    for path in paths:
        try:
            found.update(load_prediction_picks(path))
        except (ValueError, OSError, TypeError, json.JSONDecodeError):
            continue
    return found


def benchmark_choice(market, rows, asof, archived, adaptive_picks, adaptive_evaluation,
                     legacy_archive, candidate):
    """Choose a legacy category only when settled evidence beats adaptive evidence.

    This is deliberately conservative: two or more settled observations are required,
    and the legacy hit rate must lead by five percentage points after Laplace smoothing.
    The selected picks are still stored in the v52 ledger, so later runs score the
    selected forecast instead of silently discarding it.
    """
    if not candidate:
        return {"picks": adaptive_picks, "source": {cat: "adaptive" for cat in FIELDS}, "evidence": {}}
    prior = {r["date"]: r for r in rows if r["date"] < asof.isoformat()}
    evidence = {}
    selected = {cat: list(adaptive_picks.get(cat, [])) for cat in FIELDS}
    source = {cat: "adaptive" for cat in FIELDS}
    for cat in FIELDS:
        adaptive_hits = adaptive_n = legacy_hits = legacy_n = 0
        for dt, issued in archived.items():
            actual = prior.get(dt)
            if not actual or actual.get(cat) is None:
                continue
            adaptive_n += 1
            adaptive_hits += int(actual[cat] in issued.get("picks", {}).get(cat, []))
        for (m, dt), picks in legacy_archive.items():
            if m != market or dt >= asof.isoformat():
                continue
            actual = prior.get(dt)
            if not actual or actual.get(cat) is None:
                continue
            legacy_n += 1
            legacy_hits += int(actual[cat] in picks.get(cat, []))
        legacy_picks = candidate.get(cat, [])
        # A fresh benchmark ledger has no archived v52 forecasts yet. In that case,
        # use the engine's strict chronological replay as the adaptive baseline.
        # This prevents a missing ledger from making every challenger look better.
        if adaptive_n < 2:
            replay = adaptive_evaluation.get(cat, {}) if adaptive_evaluation else {}
            adaptive_hits = int(replay.get("hits", 0) or 0)
            adaptive_n = int(replay.get("samples", 0) or 0)
        # No switch without enough evidence and a valid candidate list.
        adaptive_rate = (adaptive_hits + 1) / (adaptive_n + 2) if adaptive_n else 0.0
        legacy_rate = (legacy_hits + 1) / (legacy_n + 2) if legacy_n else 0.0
        use_legacy = bool(legacy_picks and legacy_n >= 2 and legacy_rate >= adaptive_rate + 0.05)
        if use_legacy:
            selected[cat] = list(legacy_picks)
            source[cat] = "v10_challenger"
        evidence[cat] = {"adaptive_hits": adaptive_hits, "adaptive_samples": adaptive_n,
                         "adaptive_rate": adaptive_rate, "v10_hits": legacy_hits,
                         "v10_samples": legacy_n, "v10_rate": legacy_rate,
                         "selected": source[cat]}
    # Preserve structural constraints after category-level selection. If a
    # challenger digit breaks an adaptive panna/jodi, give the digit back to
    # adaptive before falling back to a generic universe ordering.
    def revert(cat, reason):
        selected[cat] = list(adaptive_picks[cat])
        source[cat] = "adaptive"
        evidence[cat]["constraint_fallback"] = reason
        evidence[cat]["selected"] = "adaptive"

    for panna, digit in (("op", "od"), ("cp", "cd")):
        valid = [p for p in selected[panna] if str(sum(map(int, p)) % 10) in selected[digit]]
        if not valid and source[digit] == "v10_challenger":
            revert(digit, "digit_reverted_for_panna")
            valid = [p for p in selected[panna] if str(sum(map(int, p)) % 10) in selected[digit]]
        if not valid:
            valid = [p for p in adaptive_picks[panna] if str(sum(map(int, p)) % 10) in selected[digit]]
        if not valid:
            valid = [p for p in PANNAS if str(sum(map(int, p)) % 10) in selected[digit]][:PICKS[panna]]
        selected[panna] = valid

    valid_jodis = [j for j in selected["j"] if j[0] in selected["od"] and j[1] in selected["cd"]]
    if not valid_jodis and source["j"] == "v10_challenger":
        revert("j", "jodi_reverted_for_digits")
        valid_jodis = [j for j in selected["j"] if j[0] in selected["od"] and j[1] in selected["cd"]]
    if not valid_jodis and source["od"] == "v10_challenger":
        revert("od", "digit_reverted_for_jodi")
        valid_jodis = [j for j in selected["j"] if j[0] in selected["od"] and j[1] in selected["cd"]]
    if not valid_jodis and source["cd"] == "v10_challenger":
        revert("cd", "digit_reverted_for_jodi")
        valid_jodis = [j for j in selected["j"] if j[0] in selected["od"] and j[1] in selected["cd"]]
    if not valid_jodis:
        valid_jodis = [j for j in adaptive_picks["j"] if j[0] in selected["od"] and j[1] in selected["cd"]]
    if not valid_jodis:
        valid_jodis = [a + b for a in selected["od"] for b in selected["cd"]][:PICKS["j"]]
    selected["j"] = valid_jodis
    # A jodi repair may have reverted a digit after the first panna pass;
    # repair pannas once more against the final digits.
    for panna, digit in (("op", "od"), ("cp", "cd")):
        valid = [p for p in selected[panna] if str(sum(map(int, p)) % 10) in selected[digit]]
        if not valid:
            valid = [p for p in adaptive_picks[panna] if str(sum(map(int, p)) % 10) in selected[digit]]
            if valid and source[panna] == "v10_challenger":
                source[panna] = "adaptive"
                evidence[panna]["constraint_fallback"] = "panna_reverted_for_final_digit"
                evidence[panna]["selected"] = "adaptive"
        if not valid:
            valid = [p for p in PANNAS if str(sum(map(int, p)) % 10) in selected[digit]][:PICKS[panna]]
        selected[panna] = valid
    return {"picks": selected, "source": source, "evidence": evidence}


def archive_audit(root, histories, asof):
    paths = set()
    for folder in (root / "output", root / "assets", root / "engine_snapshots", root / "output/history"):
        paths.update(folder.glob("predictions_*.json"))
        paths.update(folder.glob("prediction_engine_*.json"))
        if (folder / "todays_predictions.json").exists():
            paths.add(folder / "todays_predictions.json")
    paths.update((root / "state").glob("engine*_state.json"))
    for p in (root / "todays_predictions.json", root / "v50_todays_predictions.json"):
        if p.exists():
            paths.add(p)
    by_date = {m: {r["date"]: r for r in rows} for m, rows in histories.items()}
    observations = defaultdict(dict)
    files = []
    errors = []
    recovered_nested_fields = 0
    for path in sorted(paths):
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            count = 0
            for market, item, meta in prediction_entries(obj):
                pred = item.get("predicted", item.get("predictions", item))
                if not isinstance(pred, dict):
                    continue
                dt = parse_date(item.get("date") or pred.get("Date") or meta.get("prediction_date"))
                if not dt:
                    continue
                normalized = {key(k): v for k, v in pred.items()}
                picks = {}
                for cat, label in FIELDS.items():
                    raw = normalized.get(key(label), [])
                    # v33 accidentally serialized (picks, attribution) tuples as arrays.
                    if isinstance(raw, list) and len(raw) == 2 and isinstance(raw[0], list) and isinstance(raw[1], dict):
                        raw = raw[0]
                        recovered_nested_fields += 1
                    if not isinstance(raw, list):
                        raw = []
                    width = 3 if cat in ("op", "cp") else 2 if cat == "j" else 1
                    picks[cat] = sorted({v for val in raw if (v := number(val, width)) in INDEX[cat]})
                if not any(picks.values()):
                    continue
                family = str(meta.get("engine") or path.stem)
                m = market_key(market)
                identity = family + "/" + m + "/" + dt.isoformat()
                signature = digest(picks)
                observations[identity].setdefault(signature, {
                    "engine_label": family, "market": m, "date": dt.isoformat(), "picks": picks,
                    "source": str(path), "generated_at": meta.get("generated_at"),
                    "actual": by_date.get(m, {}).get(dt.isoformat()) if dt < asof else None})
                count += 1
            files.append({"path": str(path), "read_predictions": count})
        except (ValueError, OSError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    totals = defaultdict(Metrics)
    conflicts = 0
    matched = 0
    for variants in observations.values():
        if len(variants) != 1:
            conflicts += 1
            continue
        obs = next(iter(variants.values()))
        if obs["actual"]:
            totals[obs["engine_label"]].add(obs["picks"], obs["actual"])
            matched += 1
    return {"status": "retrospective_unverified_timing",
            "limitations": ["Issue-before-result timing is not proven for legacy files.",
                            "Engine labels and output paths are shared by several scripts; labels cannot identify an exact source version.",
                            "Conflicting picks for the same engine label, market and date are excluded.",
                            "State copies may overlap exported files under different labels; do not pool groups.",
                            "Different date coverage prevents a fair winner ranking.",
                            "Legacy picks have no saved expert probabilities and are scored for audit only, not attributed to new experts."],
            "files": files, "errors": errors, "unique_engine_market_dates": len(observations),
            "nested_prediction_fields_recovered": recovered_nested_fields,
            "conflicting_engine_market_dates_excluded": conflicts, "matched_engine_market_dates": matched,
            "by_engine_label": {name: metric.report() for name, metric in sorted(totals.items())}}


def default_history():
    for path in (ROOT / "sattaboss-data/data/all_markets_history.json", ROOT / "output/all_markets_history.json",
                 ROOT / "all_markets_history.json"):
        if path.exists():
            return path
    raise FileNotFoundError("Supply --history with a history JSON path.")


def aggregate_validation(markets):
    result = {}
    for model in ("adaptive", "frequency", "recent_60"):
        result[model] = {}
        for cat in FIELDS:
            entries = [m["chronological_evaluation"][model][cat] for m in markets.values()]
            count = sum(e["samples"] for e in entries)
            hits = sum(e.get("hits", 0) for e in entries)
            result[model][cat] = {"samples": count, "hits": hits,
                "hit_rate": hits / count if count else None,
                "mean_log_loss": sum(e["mean_log_loss"] * e["samples"] for e in entries if e["samples"]) / count if count else None}
    return result


def run(args):
    now = datetime.now(IST)
    asof = parse_date(args.date) if args.date else now.date()
    if asof is None:
        raise ValueError("Invalid --date; use YYYY-MM-DD.")
    histories, quality = load_history(args.history or default_history())
    selected = [market_key(m) for m in args.markets.split(",")] if args.markets else sorted(histories)
    if any(m not in histories for m in selected):
        raise ValueError("Unknown market requested; use names present in history.")
    selected = sorted(set(selected))
    simulated = asof < now.date()
    out = Path(args.output_dir).resolve()
    if simulated:
        out = out / "replay" / asof.isoformat()
    candidate_path = getattr(args, "legacy_candidate", None)
    candidate_archive = load_prediction_picks(candidate_path) if candidate_path else {}
    legacy_archive = load_legacy_v10_archive(ROOT) if candidate_path else {}
    ledger_path = out / "forecast_ledger.sqlite3"
    report = {"engine": VERSION, "date": asof.isoformat(),
              "mode": "historical_simulation" if simulated else "issued_forecasts",
              "data_quality": quality, "markets": {},
              "validation": {"window_observations": args.validation_days, "minimum_warmup": args.warmup,
                             "training_cutoff": "date strictly earlier than prediction date",
                             "method": "predict, score, update weights, then observe actual",
                             "baseline_note": "Frequency and recent models use identical dates and pick constraints. Uniform coverage assumes equally likely categories; it is not an established payout or real-world probability.",
                             "interval_note": "Wilson intervals assume independent observations. Shared markets/dates can violate this; intervals are descriptive, not proof of an edge."}}
    results = []
    newly_settled = corrected = removed = 0
    with ledger(ledger_path) as db:
        settings = digest({"version": VERSION, "window": args.validation_days, "warmup": args.warmup, "picks": PICKS})
        existing = db.execute("SELECT value FROM info WHERE name='settings'").fetchone()
        if existing and existing[0] != settings:
            raise ValueError("Settings/version changed. Use a new --output-dir to keep evaluation comparable.")
        db.execute("INSERT OR IGNORE INTO info VALUES ('settings',?)", (settings,))
        for number_market, market in enumerate(selected, 1):
            rows = histories[market]
            archived = saved_forecasts(db, market)
            trained = train_market(rows, asof, archived, args.validation_days, args.warmup)
            live_metrics, settlements, records = score_issued(archived, rows, asof)
            info_key = "settled/" + market
            previous = db.execute("SELECT value FROM info WHERE name=?", (info_key,)).fetchone()
            previous = json.loads(previous[0]) if previous else {}
            newly_settled += sum(k not in previous for k in settlements)
            corrected += sum(k in previous and previous[k] != v for k, v in settlements.items())
            removed += sum(k not in settlements for k in previous)
            db.execute("INSERT OR REPLACE INTO info VALUES (?,?)", (info_key, json.dumps(settlements)))
            issued = archived.get(asof.isoformat())
            result_already_available = any(r["date"] == asof.isoformat() for r in rows)
            candidate = candidate_archive.get((market, asof.isoformat()), {})
            benchmark = benchmark_choice(market, rows, asof, archived, trained["picks"],
                                         trained["evaluation"].get("adaptive", {}),
                                         legacy_archive, candidate)
            if not issued and (simulated or not result_already_available) and trained["history_rows"] >= args.warmup:
                issued = {k: trained[k] for k in ("picks", "probabilities", "expert_probabilities", "weights",
                                                  "training_fingerprint", "history_rows", "history_last_date")}
                issued.update({"issued_at": now.isoformat(), "version": VERSION,
                               "mode": "historical_simulation" if simulated else "issued_forecast",
                               "adaptive_picks": dict(trained["picks"]),
                               "benchmark_source": benchmark["source"],
                               "benchmark_evidence": benchmark["evidence"]})
                issued["picks"] = benchmark["picks"]
                insert_forecast(db, market, asof.isoformat(), issued)
            picks = issued["picks"] if issued else {cat: [] for cat in FIELDS}
            last = trained["history_last_date"]
            age = (asof - date.fromisoformat(last)).days if last else None
            status = ("result_already_available" if not issued and result_already_available
                      else "insufficient_history" if not issued else "stale_history" if age > 7 else "ready")
            results.append({"market": market, "predictions": {"Date": asof.isoformat(),
                            **{label: picks[cat] for cat, label in FIELDS.items()}},
                            "status": status, "issued_at": issued["issued_at"] if issued else None,
                            "confidence_basis": "See per-category chronological and issued-forecast evaluation; no invented overall confidence.",
                            "history_last_date": last, "history_age_days": age,
                            "weights_at_issue": issued["weights"] if issued else None,
                            "training_fingerprint_at_issue": issued["training_fingerprint"] if issued else None,
                            "history_changed_since_issue": bool(issued and issued["training_fingerprint"] != trained["training_fingerprint"])})
            report["markets"][market] = {"history_rows": trained["history_rows"], "history_last_date": last,
                "history_age_days": age, "chronological_evaluation": trained["evaluation"],
                "evaluation_first_date": trained["evaluation_first_date"],
                "evaluation_last_date": trained["evaluation_last_date"],
                "issued_forecast_evaluation": live_metrics, "issued_forecast_records": records,
                "weights_for_next_issue": trained["weights"], "weight_updates": trained["weight_updates"],
                "archived_feedback_fields": trained["archived_feedback_fields"],
                "benchmark": benchmark}
            if number_market % 12 == 0 or number_market == len(selected):
                print(f"Validated {number_market}/{len(selected)} markets", flush=True)
        report["feedback_changes"] = {"newly_settled_fields": newly_settled,
                                      "corrected_fields": corrected, "removed_fields": removed}
    # Ledger commits first. Interrupted exports can always be regenerated from it.
    if not args.skip_legacy_audit:
        report["legacy_archive_audit"] = archive_audit(ROOT, histories, asof)
    report["aggregate_chronological_evaluation"] = aggregate_validation(report["markets"])
    prediction_path = out / f"predictions_{asof:%Y%m%d}_v53.json"
    report_path = out / "validation_report.json"
    output = {"meta": {"engine": VERSION, "prediction_date": asof.isoformat(), "generated_at": now.isoformat(),
              "mode": report["mode"], "history_file": str(Path(args.history or default_history()).resolve()),
              "ledger_file": str(ledger_path), "validation_report": str(report_path),
              "feedback_changes": report["feedback_changes"], "deterministic": True,
              "benchmark_mode": bool(candidate_path),
              "benchmark_candidate_file": str(Path(candidate_path).resolve()) if candidate_path else None}, "markets": results}
    atomic_json(prediction_path, output)
    atomic_json(out / "todays_predictions.json", output)
    atomic_json(report_path, report)
    print(json.dumps({"predictions": str(prediction_path), "report": str(report_path),
                      "markets": len(results), **report["feedback_changes"]}, indent=2))
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--date", help="Target date YYYY-MM-DD; past dates use an isolated replay ledger")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/adaptive_v53")
    parser.add_argument("--markets", help="Comma-separated market names; default all markets")
    parser.add_argument("--validation-days", type=int, default=180, help="Recent observations used for chronological evaluation and weight adaptation")
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--skip-legacy-audit", action="store_true")
    parser.add_argument("--legacy-candidate", type=Path,
                        help="Optional v10 JSON for the same target date; used only as a measured challenger.")
    args = parser.parse_args()
    if args.validation_days < 1 or args.warmup < 1:
        parser.error("--validation-days and --warmup must be positive")
    try:
        run(args)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"Engine failed: {exc}\n")


if __name__ == "__main__":
    main()
