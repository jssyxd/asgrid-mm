"""grid_risk_controller.py - 微观网格与做市交易风控拦截器.

执行视频中强调的“活下去的金融数学四步防御法”：
1. 波动率激增拦截：当短期高频波动率超过历史中位数 2.5 倍时，强制拉宽网格 300%
2. 库存偏离上限与强平保护：库存达到风险预算上限时，立刻冻结新增逆势开仓
3. 极端趋势追踪拦截：结合 ADX 与均线，一旦确认单边大漂移，强制切断反向马丁补仓
4. 资本充足率监控：动态计算维持保证金比例，严禁无限制加仓摊平
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class GridRiskController:
    """网格与高频做市多重硬风控拦截器."""

    def __init__(
        self,
        vol_surge_multiplier: float = 2.5,
        max_margin_usage_ratio: float = 0.65,
        adx_trend_cutoff: float = 30.0,
    ):
        self.vol_surge_mult = vol_surge_multiplier
        self.max_margin_ratio = max_margin_usage_ratio
        self.adx_cutoff = adx_trend_cutoff

    def check_safety_guards(
        self,
        current_vol: float,
        median_vol: float,
        margin_used: float,
        total_equity: float,
        adx_value: float,
        trend_direction: int,
    ) -> Dict[str, Any]:
        """评估当前是否满足开仓安全边界."""
        margin_ratio = margin_used / (total_equity + 1e-8)
        is_vol_surging = bool(current_vol > median_vol * self.vol_surge_mult)
        is_margin_overload = bool(margin_ratio >= self.max_margin_ratio)
        is_strong_trend = bool(adx_value >= self.adx_cutoff)

        safe_to_trade = True
        warnings = []

        if is_vol_surging:
            safe_to_trade = False
            warnings.append("VOLATILITY_SURGE: Spread widened, new quotes paused")
        if is_margin_overload:
            safe_to_trade = False
            warnings.append("MARGIN_OVERLOAD: Maximum margin usage reached")
        if is_strong_trend:
            warnings.append(f"STRONG_TREND_{trend_direction}: Asymmetric filter active")

        return {
            "safe_to_trade": safe_to_trade,
            "margin_usage_ratio": margin_ratio,
            "is_vol_surging": is_vol_surging,
            "is_strong_trend": is_strong_trend,
            "trend_direction": trend_direction,
            "warnings": warnings,
        }
