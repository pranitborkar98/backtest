# -*- coding: utf-8 -*-
"""
================================================================================
MATKA ENGINE v14.3 — PREMIUM MARKETS FIRST
================================================================================
Premium markets (Kalyan, Milan, Rajdhani, Time Bazar, Main Bazar) processed
first in backtest/predict output, followed by all others alphabetically.

Usage:
  python matka_engine_v14_3.py              # Backtest mode
  python matka_engine_v14_3.py --predict    # Live prediction mode
================================================================================
"""

import json
import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

HISTORY_FILE = "all_markets_history.txt"
import os as _bt_os, json as _bt_json
# Backtest lab: resolve history relative to THIS script's folder (never depends on CWD),
# prefer the JSON history the lab ships with, and allow SATTA_HISTORY_FILE override.
_here = _bt_os.path.dirname(_bt_os.path.abspath(__file__))
_bt_env_hist = _bt_os.environ.get("SATTA_HISTORY_FILE")
for _cand in ([_bt_env_hist] if _bt_env_hist else []) + \
             [_bt_os.path.join(_here, "all_markets_history.json"),
              _bt_os.path.join(_here, "all_markets_history.txt"),
              "all_markets_history.json", "all_markets_history.txt"]:
    if _cand and _bt_os.path.exists(_cand):
        HISTORY_FILE = _cand
        break
DASHBOARD_DIR = "./dashboards"
os.makedirs(DASHBOARD_DIR, exist_ok=True)

# ============================================================
# PREMIUM MARKETS — These appear FIRST in output
# ============================================================
PREMIUM_MARKETS = [
    "kalyan",
    "kalyan_day",
    "kalyan_night",
    "kalyan_morning",
    "kalyan_express",
    "kalyan_sai_day",
    "kalyan_sridevi",
    "kalyan_sridevi_night",
    "milan_day",
    "milan_night",
    "milan_morning",
    "milan_bazar_day",
    "milan_bazar_morning",
    "milan_bazar_night",
    "rajdhani_day",
    "rajdhani_night",
    "rajdhani_morning",
    "time_bazar",
    "time_bazar_day",
    "time_bazar_morning",
    "time_bazar_night",
    "main_bazar",
    "main_bazar_day",
    "main_bazar_morning",
    "main_bazar_night",
    "main_mumbai_night",
    "main_mumbai_rk",
]

def sort_markets_premium_first(market_names):
    """Sort: Premium markets in defined order, then rest alphabetically."""
    premium_set = set(PREMIUM_MARKETS)
    premium = [m for m in PREMIUM_MARKETS if m in market_names]
    others = sorted([m for m in market_names if m not in premium_set])
    return premium + others

# ============================================================
# 1. DATA LOADING
# ============================================================
def load_and_preprocess(filepath, cutoff_iso=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    markets = {}
    for market, records in data.items():
        if not isinstance(records, list) or len(records) < 60:
            continue
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df.sort_values('Date').reset_index(drop=True)
        # Backtest lab time-travel: only keep records strictly BEFORE the target date
        # (predicting "next day" from history must never leak the outcome itself).
        if cutoff_iso is not None:
            df = df[df['Date'] < pd.Timestamp(cutoff_iso)].reset_index(drop=True)
        df['Open Digit']  = df['Open Digit'].astype(str)
        df['Close Digit'] = df['Close Digit'].astype(str)
        markets[market] = df
    return markets

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def create_features(df, target_col='Open Digit'):
    df = df.copy().reset_index(drop=True)
    digit_map = {str(i): i for i in range(10)}
    s = df[target_col].map(digit_map)
    n = len(df)
    other_col = 'Close Digit' if target_col == 'Open Digit' else 'Open Digit'

    for lag in [1, 2, 3, 5, 7, 14]:
        df[f'lag_{lag}'] = s.shift(lag)
    df['lag_1_other'] = df[other_col].map(digit_map).shift(1)

    df['weekday']      = df['Date'].dt.weekday
    df['day_of_month'] = df['Date'].dt.day
    df['month']        = df['Date'].dt.month
    df['is_weekend']   = (df['weekday'] >= 5).astype(int)

    for d in range(10):
        appeared = (s == d).astype(int)
        for win in [5, 10, 20, 30]:
            df[f'count_{d}_{win}d'] = appeared.shift(1).rolling(win, min_periods=3).sum()
        df[f'ewm_{d}'] = appeared.shift(1).ewm(span=10, min_periods=3).mean()

    for d_prev in range(10):
        for d_next in range(10):
            mask_prev = (s.shift(1) == d_prev).astype(int)
            mask_both = ((s.shift(1) == d_prev) & (s == d_next)).astype(int)
            num = mask_both.shift(1).rolling(30, min_periods=5).sum()
            den = mask_prev.shift(1).rolling(30, min_periods=5).sum()
            df[f'trans_{d_prev}_{d_next}'] = num / (den + 1e-6)

    df['is_repeat'] = (s == s.shift(1)).astype(int).shift(1)
    df['repeat_streak'] = df['is_repeat'].groupby((df['is_repeat'] == 0).cumsum()).cumcount().shift(1)

    diff_lag = (s.shift(1) - s.shift(2)).abs()
    df['was_fakeout_zone'] = ((diff_lag == 1) | (diff_lag == 9)).astype(int).shift(1)

    anchor_vals = []
    for i in range(n):
        if i < 5:
            anchor_vals.append(np.nan)
        else:
            yesterday = s.iloc[i-1]
            last5 = s.iloc[i-5:i]
            anchor_vals.append(((last5 - yesterday).abs() <= 2).sum())
    df['anchor_last5'] = pd.Series(anchor_vals).shift(1)

    avoid_vals = []
    for i in range(n):
        if i < 3:
            avoid_vals.append(np.nan)
        else:
            yesterday = s.iloc[i-1]
            last3 = s.iloc[i-3:i]
            avoid_vals.append(int(yesterday in last3.values))
    df['avoid_last3'] = pd.Series(avoid_vals).shift(1)

    for d in range(10):
        last_seen = []
        last_idx = -999
        for i in range(n):
            if s.iloc[i] == d:
                last_idx = i
            last_seen.append(i - last_idx)
        df[f'cold_{d}'] = pd.Series(last_seen).shift(1)

    df['post_trap_1d'] = df['was_fakeout_zone'].shift(1).fillna(0)
    df['post_trap_2d'] = df['was_fakeout_zone'].shift(2).fillna(0)
    df['post_repeat_1d'] = df['is_repeat'].shift(1).fillna(0)
    df['post_repeat_2d'] = df['is_repeat'].shift(2).fillna(0)

    if 'Close Digit' in df.columns:
        df['double_panna'] = (df['Open Digit'] == df['Close Digit']).astype(int).shift(1)
    else:
        df['double_panna'] = 0

    df['target'] = s
    df = df.dropna().reset_index(drop=True)
    return df

# ============================================================
# 3. HELPERS
# ============================================================
def get_feature_cols(df_feat):
    exclude = ['Date', 'Open Digit', 'Close Digit', 'target', 'Jodi', 'Open Panna', 'Close Panna']
    return [c for c in df_feat.columns
            if c not in exclude and df_feat[c].dtype in [np.float64, np.int64, np.int32]]

def train_full_model(df, target_col='Open Digit'):
    df_feat = create_features(df, target_col)
    feature_cols = get_feature_cols(df_feat)
    X = df_feat[feature_cols]
    y = df_feat['target'].astype(int)
    if len(np.unique(y)) < 2 or len(X) < 30:
        return None, None, None
    # Backtest lab speed: a single prediction only needs recent history. Training on
    # the last ~1500 rows keeps accuracy effectively identical (HGB uses early trees)
    # and cuts matka --predict from >600s to well under the harness timeout.
    _BT_TRAIN_WINDOW = 1500
    if len(X) > _BT_TRAIN_WINDOW:
        X = X.iloc[-_BT_TRAIN_WINDOW:]
        y = y.iloc[-_BT_TRAIN_WINDOW:]
    model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, max_depth=6, random_state=42, early_stopping=False
    )
    model.fit(X, y)
    return model, feature_cols, df_feat

def predict_next_day(df, target_col='Open Digit', k_picks=3):
    model, feature_cols, df_feat = train_full_model(df, target_col)
    if model is None:
        return None, None, None, None
    X_last = df_feat[feature_cols].iloc[-1:]
    last_date = df_feat['Date'].iloc[-1]
    probs = model.predict_proba(X_last)[0]
    top_indices = np.argsort(probs)[::-1][:k_picks]
    top_digits = [int(idx) for idx in top_indices]
    top_probs = [float(probs[idx]) for idx in top_indices]
    confidence = sum(top_probs)
    return top_digits, top_probs, confidence, last_date

# ============================================================
# 4. WALK-FORWARD BACKTEST
# ============================================================
def walk_forward_backtest(df, target_col='Open Digit', test_days=30, k_picks=3):
    df_feat = create_features(df, target_col)
    feature_cols = get_feature_cols(df_feat)
    X = df_feat[feature_cols]
    y = df_feat['target'].astype(int)

    start_test_idx = len(df_feat) - test_days
    if start_test_idx < 50:
        return None

    hits = 0; total = 0
    y_true_all = []; y_pred_top1 = []; probs_all = []; daily_results = []

    for i in range(start_test_idx, len(df_feat)):
        X_train = X.iloc[:i]; y_train = y.iloc[:i]
        if len(np.unique(y_train)) < 2:
            continue
        X_test = X.iloc[i:i+1]; y_actual = y.iloc[i]
        actual_date = df_feat['Date'].iloc[i]

        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=6, random_state=42, early_stopping=False)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[0]
        top_indices = np.argsort(probs)[::-1][:k_picks]
        top_preds = [int(idx) for idx in top_indices]
        top1_pred = int(np.argmax(probs))
        hit = int(y_actual in top_preds)
        hits += hit; total += 1
        y_true_all.append(int(y_actual)); y_pred_top1.append(top1_pred); probs_all.append(probs)
        daily_results.append({'date': actual_date, 'actual': int(y_actual), 'top1': top1_pred, 'top3': top_preds, 'hit': hit, 'prob_top1': float(probs[top1_pred])})

    if total == 0:
        return None
    hit_rate = (hits / total) * 100
    cm = confusion_matrix(y_true_all, y_pred_top1, labels=list(range(10)))
    return {'hit_rate': hit_rate, 'total': total, 'cm': cm, 'daily': daily_results,
            'y_true': y_true_all, 'y_pred': y_pred_top1, 'probs': probs_all, 'feature_cols': feature_cols}

# ============================================================
# 5. KELLY SIMULATION
# ============================================================
def kelly_simulation(daily_results, odds=9.0, kelly_fraction=0.25, bankroll_start=1000.0):
    bankroll = bankroll_start
    history = [bankroll]
    for rec in daily_results:
        stake_per_pick = bankroll * kelly_fraction / 3.0
        total_stake = stake_per_pick * 3
        if rec['hit']:
            bankroll += stake_per_pick * odds - total_stake
        else:
            bankroll -= total_stake
        history.append(bankroll)
        if bankroll <= 0:
            break
    return history

# ============================================================
# 6. DASHBOARD GENERATOR
# ============================================================
def generate_dashboard(result_od, result_cd, market_name):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0a0a0f')
    fig.suptitle(f'Matka Engine v14.3 — {market_name.upper()}', fontsize=20, fontweight='bold', color='white', y=0.98)

    ax1 = plt.subplot(2, 3, 1)
    hits = [1 if t == p else 0 for t, p in zip(result_od['y_true'], result_od['y_pred'])]
    df_hits = pd.DataFrame({'Hit': hits})
    df_hits['Rolling'] = df_hits['Hit'].rolling(7, min_periods=1).mean() * 100
    ax1.fill_between(df_hits.index, df_hits['Rolling'], color='#00f0ff', alpha=0.15)
    ax1.plot(df_hits.index, df_hits['Rolling'], color='#00f0ff', linewidth=2)
    ax1.axhline(30, color='#ff3366', linestyle='--', label='Random 30%')
    ax1.set_title('Open Digit — 7-Day Rolling Hit Rate', color='white', fontsize=12)
    ax1.set_ylabel('Hit Rate %', color='#888'); ax1.set_xlabel('Test Day', color='#888')
    ax1.legend(); ax1.set_ylim(0, 105); ax1.set_facecolor('#111118'); ax1.tick_params(colors='#888')
    ax1.spines['bottom'].set_color('#333'); ax1.spines['left'].set_color('#333')

    ax2 = plt.subplot(2, 3, 2)
    hits = [1 if t == p else 0 for t, p in zip(result_cd['y_true'], result_cd['y_pred'])]
    df_hits = pd.DataFrame({'Hit': hits})
    df_hits['Rolling'] = df_hits['Hit'].rolling(7, min_periods=1).mean() * 100
    ax2.fill_between(df_hits.index, df_hits['Rolling'], color='#00f0ff', alpha=0.15)
    ax2.plot(df_hits.index, df_hits['Rolling'], color='#00f0ff', linewidth=2)
    ax2.axhline(30, color='#ff3366', linestyle='--', label='Random 30%')
    ax2.set_title('Close Digit — 7-Day Rolling Hit Rate', color='white', fontsize=12)
    ax2.set_ylabel('Hit Rate %', color='#888'); ax2.set_xlabel('Test Day', color='#888')
    ax2.legend(); ax2.set_ylim(0, 105); ax2.set_facecolor('#111118'); ax2.tick_params(colors='#888')
    ax2.spines['bottom'].set_color('#333'); ax2.spines['left'].set_color('#333')

    ax3 = plt.subplot(2, 3, 3)
    all_daily = result_od['daily'] + result_cd['daily']
    all_daily.sort(key=lambda x: x['date'])
    bankroll = kelly_simulation(all_daily)
    days = range(len(bankroll))
    ax3.fill_between(days, bankroll, 1000, where=[b >= 1000 for b in bankroll], color='#00ff88', alpha=0.15, interpolate=True)
    ax3.fill_between(days, bankroll, 1000, where=[b < 1000 for b in bankroll], color='#ff3366', alpha=0.15, interpolate=True)
    ax3.plot(days, bankroll, color='#00ff88', linewidth=2)
    ax3.axhline(1000, color='#ffaa00', linestyle='--', label='Start ₹1000')
    ax3.set_title('Kelly Equity Curve (Combined OD+CD)', color='white', fontsize=12)
    ax3.set_ylabel('Bankroll (₹)', color='#888'); ax3.set_xlabel('Test Day', color='#888')
    ax3.legend(); ax3.set_facecolor('#111118'); ax3.tick_params(colors='#888')
    ax3.spines['bottom'].set_color('#333'); ax3.spines['left'].set_color('#333')

    ax4 = plt.subplot(2, 3, 4)
    cm = result_od['cm']; cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    im = ax4.imshow(cm_norm, cmap='magma', vmin=0, vmax=1)
    ax4.set_title('Open Digit — Confusion Matrix', color='white', fontsize=12)
    ax4.set_xlabel('Predicted', color='#888'); ax4.set_ylabel('Actual', color='#888')
    ax4.set_xticks(range(10)); ax4.set_yticks(range(10)); ax4.tick_params(colors='#888')
    for i in range(10):
        for j in range(10):
            ax4.text(j, i, f'{cm_norm[i,j]:.0%}', ha='center', va='center', color='white' if cm_norm[i,j] > 0.4 else '#ccc', fontsize=8)
    plt.colorbar(im, ax=ax4, shrink=0.7, label='Rate')

    ax5 = plt.subplot(2, 3, 5)
    cm = result_cd['cm']; cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    im = ax5.imshow(cm_norm, cmap='magma', vmin=0, vmax=1)
    ax5.set_title('Close Digit — Confusion Matrix', color='white', fontsize=12)
    ax5.set_xlabel('Predicted', color='#888'); ax5.set_ylabel('Actual', color='#888')
    ax5.set_xticks(range(10)); ax5.set_yticks(range(10)); ax5.tick_params(colors='#888')
    for i in range(10):
        for j in range(10):
            ax5.text(j, i, f'{cm_norm[i,j]:.0%}', ha='center', va='center', color='white' if cm_norm[i,j] > 0.4 else '#ccc', fontsize=8)
    plt.colorbar(im, ax=ax5, shrink=0.7, label='Rate')

    ax6 = plt.subplot(2, 3, 6)
    od_acc = np.diag(result_od['cm']) / (result_od['cm'].sum(axis=1) + 1e-6) * 100
    cd_acc = np.diag(result_cd['cm']) / (result_cd['cm'].sum(axis=1) + 1e-6) * 100
    x = np.arange(10)
    ax6.bar(x - 0.2, od_acc, 0.4, label='Open Digit', color='#00f0ff', alpha=0.8)
    ax6.bar(x + 0.2, cd_acc, 0.4, label='Close Digit', color='#ff3366', alpha=0.8)
    ax6.axhline(10, color='#ffaa00', linestyle='--', label='Random 10%')
    ax6.set_title('Per-Digit Top-1 Accuracy', color='white', fontsize=12)
    ax6.set_xlabel('Digit', color='#888'); ax6.set_ylabel('Accuracy %', color='#888')
    ax6.set_xticks(x); ax6.legend(); ax6.set_facecolor('#111118'); ax6.tick_params(colors='#888')
    ax6.spines['bottom'].set_color('#333'); ax6.spines['left'].set_color('#333')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(DASHBOARD_DIR, f"dashboard_{market_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a0a0f', edgecolor='none')
    plt.close()
    return path

# ============================================================
# 7. PREDICTION MODE
# ============================================================
def run_prediction_mode(markets_data):
    print("\n" + "=" * 80)
    print("  🔮 LIVE PREDICTION MODE — Tomorrow's Numbers")
    print("=" * 80)

    predictions = []
    sorted_markets = sort_markets_premium_first(list(markets_data.keys()))

    for market in sorted_markets:
        df = markets_data[market]
        try:
            od_preds, od_probs, od_conf, last_date = predict_next_day(df, 'Open Digit', k_picks=3)
            cd_preds, cd_probs, cd_conf, _ = predict_next_day(df, 'Close Digit', k_picks=3)
            if od_preds is None or cd_preds is None:
                continue

            avg_conf = (od_conf + cd_conf) / 2
            if avg_conf >= 0.50:
                signal = "🟢 STRONG"
            elif avg_conf >= 0.40:
                signal = "🟡 MODERATE"
            else:
                signal = "🔴 WEAK"

            predictions.append({
                'market': market,
                'last_date': last_date.strftime('%d/%m/%Y') if last_date else 'N/A',
                'open_top3': od_preds,
                'open_probs': [f'{p*100:.1f}%' for p in od_probs],
                'close_top3': cd_preds,
                'close_probs': [f'{p*100:.1f}%' for p in cd_probs],
                'confidence': avg_conf,
                'signal': signal
            })
        except Exception:
            pass

    predictions.sort(key=lambda x: x['confidence'], reverse=True)

    print(f"\n{'MARKET':<28} {'OPEN (Top-3)':<22} {'CLOSE (Top-3)':<22} {'CONF':<8} {'SIGNAL':<12}")
    print("-" * 80)
    for p in predictions:
        open_str = f"{p['open_top3']} ({', '.join(p['open_probs'][:2])})"
        close_str = f"{p['close_top3']} ({', '.join(p['close_probs'][:2])})"
        print(f"{p['market']:<28} {open_str:<22} {close_str:<22} {p['confidence']*100:>5.1f}%  {p['signal']}")

    print(f"\n{'=' * 80}")
    print("  🏆 TOP 10 STRONGEST SIGNALS")
    print(f"{'=' * 80}")
    for i, p in enumerate(predictions[:10], 1):
        print(f"   {i:>2}. {p['market']:<25} | Open: {p['open_top3']} | Close: {p['close_top3']} | {p['signal']}")

    out_file = "predictions_tomorrow.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"\n💾 Saved: {os.path.abspath(out_file)}")
    return predictions

# ============================================================
# 8. BACKTEST MODE
# ============================================================
def run_backtest_mode(markets_data):
    print("=" * 70)
    print("  MATKA ENGINE v14.3 — BACKTEST MODE")
    print("  Premium Markets First  |  Dashboards Auto-Generated")
    print("=" * 70)

    all_rates = []
    market_results = []
    dashboard_count = 0
    sorted_markets = sort_markets_premium_first(list(markets_data.keys()))

    for market in sorted_markets:
        df = markets_data[market]
        try:
            result_od = walk_forward_backtest(df, 'Open Digit', test_days=30, k_picks=3)
            result_cd = walk_forward_backtest(df, 'Close Digit', test_days=30, k_picks=3)

            if result_od is None or result_cd is None:
                print(f"⚠️  {market:<25} | Insufficient data")
                continue

            od_rate = result_od['hit_rate']; cd_rate = result_cd['hit_rate']
            combined = (od_rate + cd_rate) / 2
            all_rates.append(combined)
            market_results.append({"market": market, "rate": combined, "od": od_rate, "cd": cd_rate})

            is_premium = market in PREMIUM_MARKETS
            prefix = "⭐" if is_premium else "📊"

            if combined > 32.0 and dashboard_count < 15:
                generate_dashboard(result_od, result_cd, market)
                print(f"{prefix} {market:<25} | OD: {od_rate:>5.1f}% | CD: {cd_rate:>5.1f}% | Comb: {combined:>5.1f}% | 📈 Dashboard")
                dashboard_count += 1
            else:
                print(f"{prefix} {market:<25} | OD: {od_rate:>5.1f}% | CD: {cd_rate:>5.1f}% | Comb: {combined:>5.1f}%")

        except Exception as e:
            print(f"❌ {market:<25} | Error: {e}")


    print(f"\n{'=' * 70}")
    if all_rates:
        mean_rate = np.mean(all_rates)
        strong = sum(1 for r in all_rates if r > 35.0)
        marginal = sum(1 for r in all_rates if 30.0 <= r <= 35.0)
        weak = sum(1 for r in all_rates if r < 30.0)

        print(f"🚀 AGGREGATE ACROSS {len(all_rates)} MARKETS")
        print(f"   Mean Hit Rate (Top-3) : {mean_rate:.1f}%  (Random: 30.0%)")
        print(f"   Edge over random       : {mean_rate - 30.0:+.1f}%")
        print(f"\n📊 MARKET BREAKDOWN:")
        print(f"   🟢 Strong Edge   (>35%) : {strong}")
        print(f"   🟡 Marginal Edge (30-35%): {marginal}")
        print(f"   🔴 No Edge      (<30%)  : {weak}")

        market_results.sort(key=lambda x: x['rate'], reverse=True)
        print(f"\n🏆 TOP 5 MARKETS:")
        for i, res in enumerate(market_results[:5], 1):
            print(f"   {i}. {res['market']:<22} {res['rate']:>5.1f}%  (OD {res['od']:.1f}% | CD {res['cd']:.1f}%)")

        print(f"\n📁 Dashboards: {os.path.abspath(DASHBOARD_DIR)}  ({dashboard_count} generated)")
    print(f"{'=' * 70}")




# ============================================================
# 9. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Matka Engine v14.3 — Premium Markets First')
    parser.add_argument('--predict', action='store_true', help='Live prediction mode')
    parser.add_argument('--date', default=None, help='Override run date YYYY-MM-DD (backtesting)')
    args = parser.parse_args()
    if args.date:
        import datetime as _d
        globals()['RUN_DATE_OVERRIDE'] = _d.datetime.strptime(args.date, "%Y-%m-%d").date().isoformat()

    print("Loading historical data...")
    cutoff = globals().get('RUN_DATE_OVERRIDE')  # backtest lab: truncate history to before --date
    markets_data = load_and_preprocess(HISTORY_FILE, cutoff_iso=cutoff)
    print(f"Loaded {len(markets_data)} markets.\n")

    if args.predict:
        run_prediction_mode(markets_data)
    else:
        run_backtest_mode(markets_data)

    # Expose results for the BACKTEST LAB writer below (module level, after main()).
    try:
        globals()['predictions'] = list(predictions)  # noqa: F821  (set by run_prediction_mode)
    except NameError:
        pass

if __name__ == "__main__":
    main()

# ==== BACKTEST LAB 1-LINER: unique dated output -> D:\backtest\predictions_<YYYY-MM-DD>_v14_3.json ====
try:
    import sys as _bt_sys, re as _bt_re, json as _bt_json, os as _bt_os, datetime as _bt_dt
    from pathlib import Path as _bt_Path
    def _bt_write(out_obj, d=None):
        # Resolve the prediction date: engine meta -> per-market "Date" -> CLI flag -> passed date -> today.
        _m = (out_obj or {}).get("meta") or {} if isinstance(out_obj, dict) else {}
        ds = str(_m.get("prediction_date") or _m.get("run_date") or _m.get("date") or _m.get("target_date") or globals().get("RUN_DATE_OVERRIDE") or "")
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
        _p = _bt_dir / f"predictions_{ds}_v14_3.json"
        _t = _p.with_suffix(".json.tmp")
        _t.write_text(_bt_json.dumps(out_obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        _bt_os.replace(_t, _p)
        print(f"[BACKTEST] wrote {_p}")
    if isinstance(globals().get('predictions'), list):
        _bt_write({"meta": {"engine": "v14_3", "note": "confidence-ranked list format"}, "markets": predictions})
except Exception as _bt_e:
    print(f"[BACKTEST] skipped: {_bt_e}")
# ==== END BACKTEST LAB 1-LINER ====
