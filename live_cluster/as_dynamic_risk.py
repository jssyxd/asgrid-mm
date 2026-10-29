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
        # γ must NOT be set to extreme 3.0 (which destroys spread and triggers self-harm pricing)
        # min_spread_bps set to 4.5 bps (total spread >= 9.0 bps, strictly covering 4.0 bps roundtrip fee)
        return {
            "tier": "MICRO_ACCOUNT",
            "gamma_risk_aversion": 0.35,      # Calibrated risk aversion for micro accounts
            "kappa_order_intensity": 2.1346,
            "drift_alpha": 0.5563,
            "toxic_beta": 0.9544,
            "max_inventory": 0.03,            # Max 0.03 contracts (~$20 notional on 35U)
            "lot_sz": "0.01",                 # Minimum contract size
            "min_spread_bps": 4.5             # Half-spread >= 4.5 bps (total spread >= 9.0 bps)
        }
    elif equity_usd <= 1000.0:
        # Small Account (100 ~ 1,000 USDT)
        return {
            "tier": "SMALL_ACCOUNT",
            "gamma_risk_aversion": 0.20,
            "kappa_order_intensity": 2.1346,
            "drift_alpha": 0.5563,
            "toxic_beta": 0.9544,
            "max_inventory": 0.15,
            "lot_sz": "0.01",
            "min_spread_bps": 4.0             # Half-spread >= 4.0 bps (total spread >= 8.0 bps)
        }
    elif equity_usd <= 10000.0:
        # Medium Account (1,000 ~ 10,000 USDT)
        return {
            "tier": "MEDIUM_ACCOUNT",
            "gamma_risk_aversion": 0.10,
            "kappa_order_intensity": 2.1346,
            "drift_alpha": 0.5563,
            "toxic_beta": 0.9544,
            "max_inventory": 0.80,
            "lot_sz": "0.05",
            "min_spread_bps": 3.5             # Half-spread >= 3.5 bps (total spread >= 7.0 bps)
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
            "min_spread_bps": 3.0             # Half-spread >= 3.0 bps (total spread >= 6.0 bps)
        }
