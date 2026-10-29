import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -*- coding: utf-8 -*-
"""
tests/test_012_components_smoke.py - Automated Smoke Test for 012 Avellaneda-Stoikov Components.
Asserts:
1. Engine reservation price tilts properly with positive/negative inventory.
2. Optimal quotes widen during volatility surge.
3. Toxicity threshold blocks dangerous bid/ask quotes.
4. Risk Guard triggers circuit breaker on extreme drawdown.
"""

import math
import numpy as np
from src.as_ml_engine import AvellanedaStoikovMLEngine
from src.as_risk_guard import AvellanedaRiskGuard

def test_as_ml_engine_reservation_price_tilt():
    engine = AvellanedaStoikovMLEngine(risk_aversion=0.08, order_intensity_kappa=1.8)
    mid = 60000.0
    sigma = 0.50
    
    # Neutral inventory
    engine.inventory = 0.0
    r_neutral = engine.compute_reservation_price(mid, sigma)
    assert math.isclose(r_neutral, mid, abs_tol=1e-5), "Neutral reservation price must equal mid price"
    
    # Heavy Long inventory: reservation price must skew down to encourage selling
    engine.inventory = 4.0
    r_long = engine.compute_reservation_price(mid, sigma)
    assert r_long < mid, "Long inventory must skew reservation price downward"
    
    # Heavy Short inventory: reservation price must skew up to encourage buying
    engine.inventory = -4.0
    r_short = engine.compute_reservation_price(mid, sigma)
    assert r_short > mid, "Short inventory must skew reservation price upward"

def test_quotes_toxicity_and_vol_surge():
    engine = AvellanedaStoikovMLEngine()
    mid = 60000.0
    sigma = 0.50
    
    # Normal state
    q_norm = engine.compute_optimal_quotes(mid, sigma)
    assert q_norm["allow_bid"] and q_norm["allow_ask"]
    
    # Volatility surge: spread must expand
    q_surge = engine.compute_optimal_quotes(mid, sigma, vol_surge=True)
    assert q_surge["delta_bid"] > q_norm["delta_bid"] * 2.0
    
    # Toxic down crash: bids must be canceled to prevent catching falling knives
    q_toxic_down = engine.compute_optimal_quotes(mid, sigma, p_toxic_down=0.80)
    assert not q_toxic_down["allow_bid"], "Toxic downward crash must halt bid quoting"
    assert q_toxic_down["allow_ask"], "Asks must remain active to offload inventory"

def test_smooth_inventory_skew_no_crossing():
    """Asserts that heavy inventory skews quotes smoothly and NEVER crosses the spread into taker executions."""
    engine = AvellanedaStoikovMLEngine(max_inventory=5.0, min_spread_bps=4.0)
    mid = 65000.0
    sigma = 0.30
    
    # Long inventory: ask price must move closer to mid, bid must move further away
    engine.inventory = 4.0
    q = engine.compute_optimal_quotes(mid, sigma)
    assert q["ask_price"] > mid, "Ask price must strictly remain above mid_price (never sell at a loss or cross book)"
    assert q["bid_price"] < mid, "Bid price must strictly remain below mid_price"
    assert (q["ask_price"] - q["bid_price"]) >= (mid * 0.0004), "Total spread must strictly cover the fee hurdle"
    assert q["delta_bid"] > q["delta_ask"], "Bid distance must be wider than ask distance under long inventory"

def test_adx_trend_defense():
    """Asserts that strong macro trends halt dangerous counter-trend quoting."""
    engine = AvellanedaStoikovMLEngine()
    mid = 65000.0
    sigma = 0.30
    
    # Strong downtrend: ADX >= 28, trend_dir = -1 (dumping)
    q_down = engine.compute_optimal_quotes(mid, sigma, adx=32.0, trend_dir=-1)
    assert not q_down["allow_bid"], "Strong downtrend must halt bids (prevent catching falling knives)"
    assert q_down["allow_ask"], "Strong downtrend must allow asks (to offload any existing inventory)"
    
    # Strong uptrend: ADX >= 28, trend_dir = 1 (surging)
    q_up = engine.compute_optimal_quotes(mid, sigma, adx=35.0, trend_dir=1)
    assert not q_up["allow_ask"], "Strong uptrend must halt asks (prevent shorting into a rally)"
    assert q_up["allow_bid"], "Strong uptrend must allow bids (to cover shorts)"

def test_post_fill_cooldown():
    """Asserts that post-fill cooldown stops cascade fills during toxic sweeps."""
    engine = AvellanedaStoikovMLEngine()
    mid = 65000.0
    sigma = 0.30
    
    q_cool_bid = engine.compute_optimal_quotes(mid, sigma, cooldown_bid=True)
    assert not q_cool_bid["allow_bid"], "Active buy cooldown must halt bid quotes"
    assert q_cool_bid["allow_ask"], "Asks should remain unaffected by buy cooldown"
    
    q_cool_ask = engine.compute_optimal_quotes(mid, sigma, cooldown_ask=True)
    assert not q_cool_ask["allow_ask"], "Active sell cooldown must halt ask quotes"
    assert q_cool_ask["allow_bid"], "Bids should remain unaffected by sell cooldown"

def test_adaptive_parameters_micro_account():
    """Asserts that micro accounts (e.g. 35U) have calibrated gamma and fee-covering spread."""
    from live_cluster.as_dynamic_risk import compute_adaptive_as_parameters
    
    p_35u = compute_adaptive_as_parameters(35.0)
    assert p_35u["tier"] == "MICRO_ACCOUNT"
    assert p_35u["gamma_risk_aversion"] < 1.0, "Gamma must not be set to extreme 3.0"
    assert p_35u["min_spread_bps"] >= 4.0, "Half-spread must cover maker roundtrip fee (>= 4 bps)"

def test_run_live_kernel_smoke():
    """Exercises NautilusAvellanedaV2MMStrategy with real quoting and WebSocket fill event processing."""
    from live_cluster.as_nautilus_strategy import NautilusAvellanedaV2MMStrategy
    
    strategy = NautilusAvellanedaV2MMStrategy(
        risk_aversion_gamma=0.35,
        min_spread_bps=9.0,
        lot_sz="0.01",
        max_inventory=0.05
    )
    
    # Test quote computation
    mid = 65000.0
    quotes = strategy.compute_quotes(
        mid_price=mid,
        gk_vol=0.25,
        rolling_ofi=0.5,
        ma20_dev=0.1,
        vol_median=0.25
    )
    assert quotes["bid_px"] < mid, "Bid price must be strictly below mid price"
    assert quotes["ask_px"] > mid, "Ask price must be strictly above mid price"
    assert (quotes["ask_px"] - quotes["bid_px"]) >= (mid * 0.00045), "Spread must strictly cover Maker fee hurdle"
    assert quotes["allow_bid"] and quotes["allow_ask"]
    
    # Test Real WebSocket Fill Event Processing (eliminates code==0 fake fill bug)
    assert strategy.inventory == 0.0
    strategy.on_real_fill({
        "side": "buy",
        "fillPx": "65000.0",
        "fillSz": "0.01",
        "fee": "-0.13",
        "feeCcy": "USDT"
    })
    assert math.isclose(strategy.inventory, 0.01, abs_tol=1e-5), "Inventory must accurately increment on real fill"
    assert strategy.real_matched_trades == 1
    assert len(strategy.recent_fills) == 1
def test_risk_guard_circuit_breaker():
    guard = AvellanedaRiskGuard(max_drawdown_pct=0.08)
    
    # Safe state
    res_safe = guard.evaluate_risk(100000.0, 100000.0, 0.5, 60000.0, 0.3, 0.3)
    assert res_safe["safe"] and not res_safe["halt"]
    
    # 10% Drawdown (exceeds 8% limit): must trigger halt
    res_halt = guard.evaluate_risk(89000.0, 100000.0, 0.5, 60000.0, 0.3, 0.3)
    assert not res_halt["safe"] and res_halt["halt"]
    assert "MAX_DRAWDOWN_BREACH" in res_halt["reason"]

if __name__ == "__main__":
    test_as_ml_engine_reservation_price_tilt()
    test_quotes_toxicity_and_vol_surge()
    test_smooth_inventory_skew_no_crossing()
    test_adx_trend_defense()
    test_post_fill_cooldown()
    test_adaptive_parameters_micro_account()
    test_run_live_kernel_smoke()
    test_risk_guard_circuit_breaker()
    print("✔ All 012 Avellaneda-Stoikov smoke tests passed successfully!")
