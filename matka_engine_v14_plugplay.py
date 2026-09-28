# -*- coding: utf-8 -*-
"""
================================================================================
MATKA ENGINE v14.1 — PLUG & PLAY EDITION
================================================================================
Fully self-contained. No external ML libs needed (uses native sklearn).
Includes: Psychological features, Walk-Forward Backtest, Kelly Simulation,
          Confusion Matrix, Feature Importance, Auto-generated Dashboard PNG.

Just run:  python matka_engine_v14_plugplay.py
Output:    Console results + dashboard PNGs saved to ./dashboards/
================================================================================
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

# Native sklearn — no pip install needed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

HISTORY_FILE = "all_markets_history.txt"
DASHBOARD_DIR = "./dashboards"
os.makedirs(DASHBOARD_DIR, exist_ok=True)

# ============================================================
# 1. DATA LOADING
# ============================================================
def load_and_preprocess(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    markets = {}
    for market, records in data.items():
        if not isinstance(records, list) or len(records) < 80:
            continue
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df.sort_values('Date').reset_index(drop=True)
        df['Open Digit']  = df['Open Digit'].astype(str)
        df['Close Digit'] = df['Close Digit'].astype(str)
        markets[market] = df
    return markets

# ============================================================
# 2. FEATURE ENGINEERING (Leakage-Free)
# ============================================================
def create_features(df, target_col='Open Digit'):
    df = df.copy().reset_index(drop=True)
    digit_map = {str(i): i for i in range(10)}
    s = df[target_col].map(digit_map)
    n = len(df)
    other_col = 'Close Digit' if target_col == 'Open Digit' else 'Open Digit'

    # --- 2.1 Basic Lags ---
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f'lag_{lag}'] = s.shift(lag)

    # Cross-target lag
    df['lag_1_other'] = df[other_col].map(digit_map).shift(1)

    # --- 2.2 Date Features ---
    df['weekday']        = df['Date'].dt.weekday
    df['day_of_month']   = df['Date'].dt.day
    df['month']          = df['Date'].dt.month
    df['is_weekend']     = (df['weekday'] >= 5).astype(int)

    # --- 2.3 Rolling Frequencies (strictly shifted) ---
    for d in range(10):
        appeared = (s == d).astype(int)
        for win in [5, 10, 20, 30]:
            df[f'count_{d}_{win}d'] = appeared.shift(1).rolling(win, min_periods=3).sum()

    # --- 2.4 Exponential Decay Frequency ---
    for d in range(10):
        appeared = (s == d).astype(int)
        df[f'ewm_{d}'] = appeared.shift(1).ewm(span=10, min_periods=3).mean()

    # --- 2.5 Transition Probabilities ---
    for d_prev in range(10):
        for d_next in range(10):
            mask_prev = (s.shift(1) == d_prev).astype(int)
            mask_both = ((s.shift(1) == d_prev) & (s == d_next)).astype(int)
            num = mask_both.shift(1).rolling(30, min_periods=5).sum()
            den = mask_prev.shift(1).rolling(30, min_periods=5).sum()
            df[f'trans_{d_prev}_{d_next}'] = num / (den + 1e-6)

    # --- 2.6 Psychological Features ---
    # Repeat streak
    df['is_repeat'] = (s == s.shift(1)).astype(int).shift(1)
    df['repeat_streak'] = df['is_repeat'].groupby((df['is_repeat'] == 0).cumsum()).cumcount().shift(1)

    # Fakeout trap
    diff_lag = (s.shift(1) - s.shift(2)).abs()
    df['was_fakeout_zone'] = ((diff_lag == 1) | (diff_lag == 9)).astype(int).shift(1)

    # Anchoring
    anchor_vals = []
    for i in range(n):
        if i < 5:
            anchor_vals.append(np.nan)
        else:
            yesterday = s.iloc[i-1]
            last5 = s.iloc[i-5:i]
            anchor_vals.append(((last5 - yesterday).abs() <= 2).sum())
    df['anchor_last5'] = pd.Series(anchor_vals).shift(1)

    # Avoidance
    avoid_vals = []
    for i in range(n):
        if i < 3:
            avoid_vals.append(np.nan)
        else:
            yesterday = s.iloc[i-1]
            last3 = s.iloc[i-3:i]
            avoid_vals.append(int(yesterday in last3.values))
    df['avoid_last3'] = pd.Series(avoid_vals).shift(1)

    # Cold numbers (days since each digit appeared)
    for d in range(10):
        last_seen = []
        last_idx = -999
        for i in range(n):
            if s.iloc[i] == d:
                last_idx = i
            last_seen.append(i - last_idx)
        df[f'cold_{d}'] = pd.Series(last_seen).shift(1)

    # Post-trap / post-repeat signals
    df['post_trap_1d'] = df['was_fakeout_zone'].shift(1).fillna(0)
    df['post_trap_2d'] = df['was_fakeout_zone'].shift(2).fillna(0)
    df['post_repeat_1d'] = df['is_repeat'].shift(1).fillna(0)
    df['post_repeat_2d'] = df['is_repeat'].shift(2).fillna(0)

    # Double panna
    if 'Close Digit' in df.columns:
        df['double_panna'] = (df['Open Digit'] == df['Close Digit']).astype(int).shift(1)
    else:
        df['double_panna'] = 0

    # --- 2.7 Target ---
    df['target'] = s

    # Drop NaNs
    df = df.dropna().reset_index(drop=True)
    return df

# ============================================================
# 3. WALK-FORWARD BACKTEST
# ============================================================
def walk_forward_backtest(df, target_col='Open Digit', test_days=30, k_picks=3):
    df_feat = create_features(df, target_col)

    exclude = ['Date', 'Open Digit', 'Close Digit', 'target',
               'Jodi', 'Open Panna', 'Close Panna']
    feature_cols = [c for c in df_feat.columns
                    if c not in exclude and df_feat[c].dtype in [np.float64, np.int64, np.int32]]

    X = df_feat[feature_cols]
    y = df_feat['target'].astype(int)

    start_test_idx = len(df_feat) - test_days
    if start_test_idx < 50:
        return None

    hits = 0
    total = 0
    y_true_all = []
    y_pred_top1 = []
    probs_all = []
    daily_results = []

    for i in range(start_test_idx, len(df_feat)):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]

        if len(np.unique(y_train)) < 2:
            continue

        X_test = X.iloc[i:i+1]
        y_actual = y.iloc[i]
        actual_date = df_feat['Date'].iloc[i]

        # HistGradientBoosting — fast, native, multiclass
        model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=42,
            early_stopping=False
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[0]
        top_indices = np.argsort(probs)[::-1][:k_picks]
        top_preds = [int(idx) for idx in top_indices]
        top1_pred = int(np.argmax(probs))

        hit = int(y_actual in top_preds)
        hits += hit
        total += 1

        y_true_all.append(int(y_actual))
        y_pred_top1.append(top1_pred)
        probs_all.append(probs)
        daily_results.append({
            'date': actual_date,
            'actual': int(y_actual),
            'top1': top1_pred,
            'top3': top_preds,
            'hit': hit,
            'prob_top1': float(probs[top1_pred])
        })

    if total == 0:
        return None

    hit_rate = (hits / total) * 100
    cm = confusion_matrix(y_true_all, y_pred_top1, labels=list(range(10)))

    return {
        'hit_rate': hit_rate,
        'total': total,
        'cm': cm,
        'daily': daily_results,
        'y_true': y_true_all,
        'y_pred': y_pred_top1,
        'probs': probs_all,
        'feature_cols': feature_cols
    }

# ============================================================
# 4. KELLY SIMULATION
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
# 5. DASHBOARD GENERATOR
# ============================================================
def generate_dashboard(result_od, result_cd, market_name, feature_cols):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0a0a0f')
    fig.suptitle(f'Matka Engine v14.1 — {market_name.upper()}',
                 fontsize=20, fontweight='bold', color='white', y=0.98)

    # --- Panel 1: Rolling Hit Rate (Open Digit) ---
    ax1 = plt.subplot(2, 3, 1)
    hits = [1 if t == p else 0 for t, p in zip(result_od['y_true'], result_od['y_pred'])]
    df_hits = pd.DataFrame({'Hit': hits})
    df_hits['Rolling'] = df_hits['Hit'].rolling(7, min_periods=1).mean() * 100
    ax1.fill_between(df_hits.index, df_hits['Rolling'], color='#00f0ff', alpha=0.15)
    ax1.plot(df_hits.index, df_hits['Rolling'], color='#00f0ff', linewidth=2)
    ax1.axhline(30, color='#ff3366', linestyle='--', label='Random 30%')
    ax1.set_title('Open Digit — 7-Day Rolling Hit Rate', color='white', fontsize=12)
    ax1.set_ylabel('Hit Rate %', color='#888')
    ax1.set_xlabel('Test Day', color='#888')
    ax1.legend()
    ax1.set_ylim(0, 105)
    ax1.set_facecolor('#111118')
    ax1.tick_params(colors='#888')
    ax1.spines['bottom'].set_color('#333')
    ax1.spines['left'].set_color('#333')

    # --- Panel 2: Rolling Hit Rate (Close Digit) ---
    ax2 = plt.subplot(2, 3, 2)
    hits = [1 if t == p else 0 for t, p in zip(result_cd['y_true'], result_cd['y_pred'])]
    df_hits = pd.DataFrame({'Hit': hits})
    df_hits['Rolling'] = df_hits['Hit'].rolling(7, min_periods=1).mean() * 100
    ax2.fill_between(df_hits.index, df_hits['Rolling'], color='#00f0ff', alpha=0.15)
    ax2.plot(df_hits.index, df_hits['Rolling'], color='#00f0ff', linewidth=2)
    ax2.axhline(30, color='#ff3366', linestyle='--', label='Random 30%')
    ax2.set_title('Close Digit — 7-Day Rolling Hit Rate', color='white', fontsize=12)
    ax2.set_ylabel('Hit Rate %', color='#888')
    ax2.set_xlabel('Test Day', color='#888')
    ax2.legend()
    ax2.set_ylim(0, 105)
    ax2.set_facecolor('#111118')
    ax2.tick_params(colors='#888')
    ax2.spines['bottom'].set_color('#333')
    ax2.spines['left'].set_color('#333')

    # --- Panel 3: Kelly Equity (Combined) ---
    ax3 = plt.subplot(2, 3, 3)
    all_daily = result_od['daily'] + result_cd['daily']
    all_daily.sort(key=lambda x: x['date'])
    bankroll = kelly_simulation(all_daily)
    days = range(len(bankroll))
    ax3.fill_between(days, bankroll, 1000, where=[b >= 1000 for b in bankroll],
                     color='#00ff88', alpha=0.15, interpolate=True)
    ax3.fill_between(days, bankroll, 1000, where=[b < 1000 for b in bankroll],
                     color='#ff3366', alpha=0.15, interpolate=True)
    ax3.plot(days, bankroll, color='#00ff88', linewidth=2)
    ax3.axhline(1000, color='#ffaa00', linestyle='--', label='Start ₹1000')
    ax3.set_title('Kelly Equity Curve (Combined OD+CD)', color='white', fontsize=12)
    ax3.set_ylabel('Bankroll (₹)', color='#888')
    ax3.set_xlabel('Test Day', color='#888')
    ax3.legend()
    ax3.set_facecolor('#111118')
    ax3.tick_params(colors='#888')
    ax3.spines['bottom'].set_color('#333')
    ax3.spines['left'].set_color('#333')

    # --- Panel 4: Confusion Matrix OD ---
    ax4 = plt.subplot(2, 3, 4)
    cm = result_od['cm']
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    im = ax4.imshow(cm_norm, cmap='magma', vmin=0, vmax=1)
    ax4.set_title('Open Digit — Confusion Matrix', color='white', fontsize=12)
    ax4.set_xlabel('Predicted', color='#888')
    ax4.set_ylabel('Actual', color='#888')
    ax4.set_xticks(range(10))
    ax4.set_yticks(range(10))
    ax4.tick_params(colors='#888')
    for i in range(10):
        for j in range(10):
            ax4.text(j, i, f'{cm_norm[i,j]:.0%}', ha='center', va='center',
                     color='white' if cm_norm[i,j] > 0.4 else '#ccc', fontsize=8)
    plt.colorbar(im, ax=ax4, shrink=0.7, label='Rate')

    # --- Panel 5: Confusion Matrix CD ---
    ax5 = plt.subplot(2, 3, 5)
    cm = result_cd['cm']
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    im = ax5.imshow(cm_norm, cmap='magma', vmin=0, vmax=1)
    ax5.set_title('Close Digit — Confusion Matrix', color='white', fontsize=12)
    ax5.set_xlabel('Predicted', color='#888')
    ax5.set_ylabel('Actual', color='#888')
    ax5.set_xticks(range(10))
    ax5.set_yticks(range(10))
    ax5.tick_params(colors='#888')
    for i in range(10):
        for j in range(10):
            ax5.text(j, i, f'{cm_norm[i,j]:.0%}', ha='center', va='center',
                     color='white' if cm_norm[i,j] > 0.4 else '#ccc', fontsize=8)
    plt.colorbar(im, ax=ax5, shrink=0.7, label='Rate')

    # --- Panel 6: Per-Digit Accuracy ---
    ax6 = plt.subplot(2, 3, 6)
    od_acc = np.diag(result_od['cm']) / (result_od['cm'].sum(axis=1) + 1e-6) * 100
    cd_acc = np.diag(result_cd['cm']) / (result_cd['cm'].sum(axis=1) + 1e-6) * 100
    x = np.arange(10)
    width = 0.35
    bars1 = ax6.bar(x - width/2, od_acc, width, label='Open Digit', color='#00f0ff', alpha=0.8)
    bars2 = ax6.bar(x + width/2, cd_acc, width, label='Close Digit', color='#ff3366', alpha=0.8)
    ax6.axhline(10, color='#ffaa00', linestyle='--', label='Random 10%')
    ax6.set_title('Per-Digit Top-1 Accuracy', color='white', fontsize=12)
    ax6.set_xlabel('Digit', color='#888')
    ax6.set_ylabel('Accuracy %', color='#888')
    ax6.set_xticks(x)
    ax6.legend()
    ax6.set_facecolor('#111118')
    ax6.tick_params(colors='#888')
    ax6.spines['bottom'].set_color('#333')
    ax6.spines['left'].set_color('#333')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(DASHBOARD_DIR, f"dashboard_{market_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a0a0f', edgecolor='none')
    plt.close()
    return path

# ============================================================
# 6. MAIN RUNNER
# ============================================================
def run_engine():
    print("=" * 70)
    print("  MATKA ENGINE v14.1 — PLUG & PLAY")
    print("  Model: HistGradientBoosting  |  Features: 100+  |  Dashboard: Auto")
    print("=" * 70)

    print("\nLoading historical data...")
    markets_data = load_and_preprocess(HISTORY_FILE)
    print(f"Loaded {len(markets_data)} markets.\n")

    all_rates = []
    market_results = []
    dashboard_count = 0

    for market, df in markets_data.items():
        try:
            result_od = walk_forward_backtest(df, 'Open Digit', test_days=30, k_picks=3)
            result_cd = walk_forward_backtest(df, 'Close Digit', test_days=30, k_picks=3)

            if result_od is None or result_cd is None:
                print(f"⚠️  {market:<25} | Insufficient data for walk-forward")
                continue

            od_rate = result_od['hit_rate']
            cd_rate = result_cd['hit_rate']
            combined = (od_rate + cd_rate) / 2
            all_rates.append(combined)
            market_results.append({"market": market, "rate": combined,
                                   "od": od_rate, "cd": cd_rate})

            # Auto-generate dashboard for top-performing markets
            if combined > 32.0 and dashboard_count < 10:
                path = generate_dashboard(result_od, result_cd, market, result_od['feature_cols'])
                print(f"📊 {market:<25} | OD: {od_rate:>5.1f}% | CD: {cd_rate:>5.1f}% | Comb: {combined:>5.1f}% | 📈 Dashboard saved")
                dashboard_count += 1
            else:
                print(f"📊 {market:<25} | OD: {od_rate:>5.1f}% | CD: {cd_rate:>5.1f}% | Comb: {combined:>5.1f}%")

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

        print(f"\n📁 Dashboards saved to: {os.path.abspath(DASHBOARD_DIR)}")
        print(f"   ({dashboard_count} dashboards generated for markets with >32% hit rate)")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    run_engine()
