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
    test_risk_guard_circuit_breaker()
    print("✔ All 012 Avellaneda-Stoikov smoke tests passed successfully!")
