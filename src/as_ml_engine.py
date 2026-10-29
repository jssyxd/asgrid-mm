# -*- coding: utf-8 -*-
"""
as_ml_engine.py - Avellaneda-Stoikov Market Making Engine with ML Toxicity & Adverse Selection Gate.
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, Optional


class AvellanedaStoikovMLEngine:
    """受约束最优控制与机器学习毒性门控微观做市引擎."""

    def __init__(
        self,
        risk_aversion: float = 0.08,
        order_intensity_kappa: float = 1.8,
        terminal_horizon: float = 1.0,
        max_inventory: float = 5.0,
        lot_size: float = 0.05,
        min_spread_bps: float = 2.0,
        drift_alpha: float = 0.35,
        toxic_beta: float = 1.20,
    ):
        self.gamma = risk_aversion
        self.kappa = order_intensity_kappa
        self.T = terminal_horizon
        self.max_inventory = max_inventory
        self.lot_size = lot_size
        self.min_spread_ratio = min_spread_bps / 10000.0
        self.alpha_drift = drift_alpha
        self.beta_toxic = toxic_beta

        self.inventory: float = 0.0
        self.cash: float = 100000.0
        self.defensive_mode: bool = False

    def compute_reservation_price(
        self, mid_price: float, sigma: float, mu_pred: float = 0.0, t_left: float = 1.0
    ) -> float:
        """计算结合机器学习价格漂移预测的内部保留价格."""
        inventory_skew = self.inventory * self.gamma * (sigma ** 2) * mid_price * 0.001 * t_left
        drift_adjustment = self.alpha_drift * mu_pred * (mid_price * 0.0005)
        return mid_price - inventory_skew + drift_adjustment

    def compute_optimal_quotes(
        self,
        mid_price: float,
        sigma: float,
        mu_pred: float = 0.0,
        p_toxic_down: float = 0.0,
        p_toxic_up: float = 0.0,
        vol_surge: bool = False,
    ) -> Dict[str, Any]:
        """计算买卖双侧最优挂单价格与挂单许可状态机."""
        r = self.compute_reservation_price(mid_price, sigma, mu_pred)

        base_half_spread = (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa) * (mid_price * 0.0004)
        min_half_spread = mid_price * self.min_spread_ratio
        base_half_spread = max(min_half_spread, base_half_spread)

        if vol_surge:
            base_half_spread *= 2.5

        delta_ask = base_half_spread * (1.0 + self.beta_toxic * p_toxic_up)
        delta_bid = base_half_spread * (1.0 + self.beta_toxic * p_toxic_down)

        bid_price = r - delta_bid
        ask_price = r + delta_ask

        allow_bid = True
        allow_ask = True

        if p_toxic_down >= 0.65:
            allow_bid = False
            ask_price = min(ask_price, mid_price + base_half_spread * 0.6)

        if p_toxic_up >= 0.65:
            allow_ask = False
            bid_price = max(bid_price, mid_price - base_half_spread * 0.6)

        if self.inventory >= self.max_inventory:
            allow_bid = False
        elif self.inventory <= -self.max_inventory:
            allow_ask = False

        return {
            "reservation_price": r,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "allow_bid": allow_bid,
            "allow_ask": allow_ask,
            "delta_bid": delta_bid,
            "delta_ask": delta_ask,
            "vol_surge": vol_surge,
        }
