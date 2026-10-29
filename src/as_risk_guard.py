# -*- coding: utf-8 -*-
"""
as_risk_guard.py - High-Frequency LOB Market Making Risk Controller & Circuit Breaker.
"""

from typing import Dict, Any


class AvellanedaRiskGuard:
    """高频做市多重硬风控与黑天鹅断路器."""

    def __init__(
        self,
        max_drawdown_pct: float = 0.08,
        vol_surge_threshold: float = 2.2,
        max_inventory_notional: float = 250000.0,
    ):
        self.max_dd_pct = max_drawdown_pct
        self.vol_surge_th = vol_surge_threshold
        self.max_notional = max_inventory_notional
        self.is_halted = False
        self.halt_reason = ""

    def evaluate_risk(
        self,
        current_equity: float,
        peak_equity: float,
        inventory_btc: float,
        mid_price: float,
        current_vol: float,
        median_vol: float,
    ) -> Dict[str, Any]:
        """评估当前做市状态是否处于安全防御区."""
        drawdown = (peak_equity - current_equity) / (peak_equity + 1e-9)
        if drawdown >= self.max_dd_pct:
            self.is_halted = True
            self.halt_reason = f"MAX_DRAWDOWN_BREACH: 回撤达到 {drawdown*100:.2f}% (限制: {self.max_dd_pct*100:.1f}%)"
            return {"safe": False, "halt": True, "reason": self.halt_reason, "vol_surge": False}

        notional_exp = abs(inventory_btc * mid_price)
        if notional_exp >= self.max_notional:
            return {"safe": False, "halt": False, "reason": "MAX_NOTIONAL_REACHED", "vol_surge": False}

        vol_surge = bool(current_vol >= median_vol * self.vol_surge_th)
        return {
            "safe": True,
            "halt": False,
            "drawdown_pct": drawdown * 100.0,
            "vol_surge": vol_surge,
            "reason": "OK" if not vol_surge else "VOLATILITY_EXPANSION"
        }
