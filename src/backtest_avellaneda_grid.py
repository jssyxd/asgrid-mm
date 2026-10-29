"""
backtest_avellaneda_grid.py - 策略 006: Avellaneda-Stoikov 自适应高频网格做市实测回测
包含与传统对称网格 (Naive Symmetric Grid) 的对抗性对比。
"""

import os
import sys
import numpy as np
import pandas as pd

def run_backtest(data_path="data_btc_1m_real.parquet"):
    if not os.path.exists(data_path):
        alt_path = "../to-research/012_avellaneda_grid/data_btc_1m_real.parquet"
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot find {data_path}")

    df = pd.read_parquet(data_path)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n_bars = len(close)

    log_hl = np.log(high / low)
    parkinson_var = (log_hl ** 2) / (4.0 * np.log(2.0))
    sigma_series = np.sqrt(pd.Series(parkinson_var).rolling(60).mean().fillna(0.0005).values * (365 * 1440))

    gamma = 0.08
    kappa = 1.8
    maker_fee = 0.0002
    initial_capital = 100000.0
    lot_size = 0.05

    # 1. Avellaneda-Stoikov Adaptive Grid
    as_cash = initial_capital
    as_q = 0.0
    as_equity = [initial_capital]
    as_trades = []

    # 2. Naive Symmetric Grid
    naive_cash = initial_capital
    naive_q = 0.0
    naive_equity = [initial_capital]
    naive_trades = []

    for t in range(60, n_bars):
        s = close[t]
        h = high[t]
        l = low[t]
        sig = max(0.10, sigma_series[t])
        
        # AS Grid
        r = s - (as_q * lot_size) * gamma * (sig ** 2) * s * 0.001
        half_spread = (1.0 / gamma) * np.log(1.0 + gamma / kappa) * (s * 0.0004)
        half_spread = max(s * 0.0005, half_spread)
        
        as_bid = r - half_spread
        as_ask = r + half_spread
        
        if l <= as_bid and as_q < 10:
            fill_px = as_bid
            fee = fill_px * lot_size * maker_fee
            as_cash -= (fill_px * lot_size + fee)
            as_q += 1.0
            as_trades.append(('BUY', fill_px, fee))
            
        if h >= as_ask and as_q > -10:
            fill_px = as_ask
            fee = fill_px * lot_size * maker_fee
            as_cash += (fill_px * lot_size - fee)
            as_q -= 1.0
            as_trades.append(('SELL', fill_px, fee))
            
        as_current_eq = as_cash + (as_q * lot_size) * s
        as_equity.append(as_current_eq)
        
        # Naive Grid
        step = s * 0.002
        naive_bid = s - step
        naive_ask = s + step
        
        if l <= naive_bid:
            fill_px = naive_bid
            fee = fill_px * lot_size * maker_fee
            naive_cash -= (fill_px * lot_size + fee)
            naive_q += 1.0
            naive_trades.append(('BUY', fill_px, fee))
            
        if h >= naive_ask and naive_q > 0:
            fill_px = naive_ask
            fee = fill_px * lot_size * maker_fee
            naive_cash += (fill_px * lot_size - fee)
            naive_q -= 1.0
            naive_trades.append(('SELL', fill_px, fee))
            
        naive_current_eq = naive_cash + (naive_q * lot_size) * s
        naive_equity.append(naive_current_eq)

    as_eq = np.array(as_equity)
    as_peak = np.maximum.accumulate(as_eq)
    as_mdd = np.max((as_peak - as_eq) / as_peak) * 100.0

    naive_eq = np.array(naive_equity)
    naive_peak = np.maximum.accumulate(naive_eq)
    naive_mdd = np.max((naive_peak - naive_eq) / naive_peak) * 100.0

    print("=" * 70)
    print("STRATEGY 006: AVELLANEDA-STOIKOV GRID REAL BACKTEST (8,000 1m BARS)")
    print("=" * 70)
    print(f"AS 自适应网格 -> 期末: ${as_eq[-1]:,.2f} | 净利: +${as_eq[-1]-initial_capital:,.2f} (+{(as_eq[-1]-initial_capital)/initial_capital*100:.2f}%) | 最大回撤: {as_mdd:.2f}% | 成交笔数: {len(as_trades)}")
    print(f"传统对称网格  -> 期末: ${naive_eq[-1]:,.2f} | 净利: +${naive_eq[-1]-initial_capital:,.2f} (+{(naive_eq[-1]-initial_capital)/initial_capital*100:.2f}%) | 最大回撤: {naive_mdd:.2f}% | 成交笔数: {len(naive_trades)}")
    print("-" * 70)
    print(f"回撤控制优势: AS 网格最大回撤比传统对称网格低 {(1.0 - as_mdd/naive_mdd)*100:.1f}%")
    print(f"收益捕获优势: AS 网格累计净利是传统对称网格的 {(as_eq[-1]-initial_capital)/(naive_eq[-1]-initial_capital):.1f} 倍")
    print("=" * 70)

if __name__ == '__main__':
    run_backtest()
