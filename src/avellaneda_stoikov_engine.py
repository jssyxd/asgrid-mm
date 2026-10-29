"""avellaneda_stoikov_engine.py - Avellaneda-Stoikov 做市商模型与微观自适应非对称网格系统.

核心思想（还原自 教学素材视频《传统网格马丁为什么会爆仓？金融数学如何拯救这个“印钞机”》）：
1. 传统网格/马丁爆仓本质：假定价格一定会均值回归，无保护被动接库存；赚取微小的震荡收益，承担单边大趋势漂移的无限尾部风险。
2. 金融数学拯救方案：将网格交易重构为“受约束的最优控制问题” (Constrained Optimal Control)：
   - 基于 Avellaneda-Stoikov (2008) 经典做市商随机控制偏微分方程 (HJB 方程)；
   - 内部保留价格 (Reservation Price / Mid-Price Tilt):
     r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
     其中:
       s: 当前市场中间价
       q: 内部持有净库存 (Inventory, 多头为正, 空头为负)
       gamma: 风险厌恶参数 (Risk Aversion)
       sigma: 瞬时波动率 (Volatility)
       (T - t): 剩余控制时域
   - 最优非对称挂单间距 (Optimal Asymmetric Quotes):
     delta_a (卖单距离) 与 delta_b (买单距离) 随库存 q 动态偏置。
     当多头库存积压 (q >> 0) 时，保留价格向下修正，使得买单挂得更低更远（甚至拒绝挂买单），卖单挂得更近更激进，主动促成去库存！
3. 动态波动率网格缩放与防御模式：
   - 波动率低时，收窄网格提高周转率；
   - 波动率飙升时，拉宽网格避开流动性黑洞；
   - 触发最大库存上限或保证金占用阈值时，强制进入防御状态（停止逆向开仓，只允许平仓）。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


class AvellanedaStoikovGridEngine:
    """受约束最优控制做市商网格交易引擎."""

    def __init__(
        self,
        risk_aversion: float = 0.1,     # gamma: 风险厌恶程度
        order_intensity_kappa: float = 1.5, # kappa: 订单簿流动性密度参数
        terminal_horizon: float = 1.0,   # T - t: 归一化时间窗口
        max_inventory: int = 10,         # 最大允许单向持仓手数
        base_grid_step: float = 0.002,   # 基础网格步长 (0.2%)
        max_drawdown_pct: float = 0.10,  # 组合硬性止损回撤阈值 (10%)
    ):
        self.gamma = risk_aversion
        self.kappa = order_intensity_kappa
        self.T = terminal_horizon
        self.max_inventory = max_inventory
        self.base_step = base_grid_step
        self.max_dd = max_drawdown_pct
        
        # 运行时状态
        self.inventory: int = 0
        self.cash: float = 100000.0
        self.peak_portfolio_value: float = 100000.0
        self.defensive_mode: bool = False

    def compute_reservation_price(self, mid_price: float, sigma: float, t_left: float = 1.0) -> float:
        """计算保留价格 (Reservation Price)."""
        # r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
        skew = self.inventory * self.gamma * (sigma ** 2) * t_left
        return mid_price - skew

    def compute_optimal_spreads(self, sigma: float, t_left: float = 1.0) -> Tuple[float, float]:
        """计算最优不对称买卖价差 (delta_bid, delta_ask)."""
        # 基础最优半价差: delta_half = 0.5 * gamma * sigma^2 + (1 / gamma) * ln(1 + gamma / kappa)
        base_half_spread = (1.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
        vol_adjusted_half = base_half_spread * (1.0 + 3.0 * sigma)

        # 库存不对称偏置项
        inventory_bias = (2.0 * self.inventory + 1.0) * self.gamma * (sigma ** 2) * t_left / 2.0
        
        delta_ask = vol_adjusted_half + inventory_bias
        delta_bid = vol_adjusted_half - inventory_bias

        return max(0.0005, delta_bid), max(0.0005, delta_ask)

    def step(self, mid_price: float, sigma: float, trend_signal: int = 0) -> Dict[str, Any]:
        """按时间步生成最优买卖挂单指令与风控状态.
        
        参数:
        - mid_price: 实时中间价
        - sigma: 当前波动率
        - trend_signal: 中长周期趋势方向 (1: 顺势多, -1: 顺势空, 0: 震荡)
        """
        current_value = self.cash + self.inventory * mid_price
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value

        # 1. 硬性风控检查：回撤与库存上限
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        if drawdown >= self.max_dd or abs(self.inventory) >= self.max_inventory:
            self.defensive_mode = True
        else:
            self.defensive_mode = False

        # 2. 计算保留中枢与非对称价差
        r = self.compute_reservation_price(mid_price, sigma)
        delta_bid, delta_ask = self.compute_optimal_spreads(sigma)

        # 3. 结合趋势信号形成非对称网格
        # 如果大趋势强力向下 (trend_signal == -1)，绝不盲目在逆势方向加仓
        allow_buy = True
        allow_sell = True

        if self.defensive_mode:
            # 防御模式下：多头超载则禁止再买入，只允许以激进卖单减仓
            if self.inventory > 0:
                allow_buy = False
            elif self.inventory < 0:
                allow_sell = False
        else:
            if trend_signal < 0 and self.inventory >= 2:
                allow_buy = False
            elif trend_signal > 0 and self.inventory <= -2:
                allow_sell = False

        bid_price = r - delta_bid if allow_buy else 0.0
        ask_price = r + delta_ask if allow_sell else 0.0

        return {
            "mid_price": mid_price,
            "reservation_price": r,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "delta_bid": delta_bid,
            "delta_ask": delta_ask,
            "inventory": self.inventory,
            "portfolio_value": current_value,
            "drawdown": drawdown,
            "defensive_mode": self.defensive_mode,
            "allow_buy": allow_buy,
            "allow_sell": allow_sell,
        }
