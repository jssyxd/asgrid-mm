# -*- coding: utf-8 -*-
"""
as_dynamic_risk.py - Automated Equity-Based Risk Scaling & Adaptive Avellaneda-Stoikov Parameters.
Dynamically scales Risk Aversion (gamma), Max Inventory, and Lot Size based on real-time account equity (USDT).
"""

def compute_adaptive_as_parameters(equity_usd: float) -> dict:
    """
     Automatically adjusts Avellaneda-Stoikov market making parameters 
    based on account equity to protect micro accounts from OOM/liquidation 
    while maximizing institutional efficiency on large accounts.
    """
    if equity_usd <= 100.0:
        # Micro Account (e.g. 35 USDT live account)
        return {
            "tier": "MICRO_ACCOUNT",
            "gamma_risk_aversion": 3.00,      # Extremely high risk aversion
            "kappa_order_intensity": 2.1346,
            "drift_alpha": 0.5563,
            "toxic_beta": 0.9544,
            "max_inventory": 0.05,            # Max 0.05 contracts (~$40 notional)
            "lot_sz": "0.01",                 # Minimum contract size
            "min_spread_bps": 3.0
        }
    elif equity_usd <= 1000.0:
        # Small Account (100 ~ 1,000 USDT)
        return {
            "tier": "SMALL_ACCOUNT",
            "gamma_risk_aversion": 1.50,
            "kappa_order_intensity": 2.1346,
            "drift_alpha": 0.5563,
            "toxic_beta": 0.9544,
            "max_inventory": 0.20,
            "lot_sz": "0.01",
            "min_spread_bps": 2.5
        }
    elif equity_usd <= 10000.0:
        # Medium Account (1,000 ~ 10,000 USDT)
        return {
            "tier": "MEDIUM_ACCOUNT",
            "gamma_risk_aversion": 0.50,
            "kappa_order_intensity": 2.1346,
            "drift_alpha": 0.5563,
            "toxic_beta": 0.9544,
            "max_inventory": 1.00,
            "lot_sz": "0.05",
            "min_spread_bps": 2.0
        }
    else:
        # Large / Institutional Account (> 10,000 USDT)
        return {
            "tier": "INSTITUTIONAL_ACCOUNT",
            "gamma_risk_aversion": 0.0451,    # Certified Kaggle baseline
            "kappa_order_intensity": 2.1346,
            "drift_alpha": 0.5563,
            "toxic_beta": 0.9544,
            "max_inventory": 5.00,
            "lot_sz": "0.10",
            "min_spread_bps": 2.0
        }
