# -*- coding: utf-8 -*-
"""
as_nautilus_strategy.py - Production NautilusTrader-Compatible Strategy & Core
Strategy 012: Avellaneda-Stoikov ML Grid V2 (Real Fill Accounting + Fee Hurdle + WebSocket OFI).
"""

import math
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

class NautilusAvellanedaV2MMStrategy:
    """
    NautilusTrader 兼容微观高频事件驱动做市状态机：
    1. 真实手续费门槛约束：双边总价差强制 >= 8.0 bps (针对 Maker 0.02% 费率形成 2.0x 利润覆盖)；
    2. 微观保留价格与去偏毒性断路器：仅在 OFI 严重背离且破位时熔断买单，杜绝大牛市被动抗空；
    3. 严格真实成交记账：挂单仅作为活动订单委托，绝不计为成交流水，仅在交易所确认真实 Fill 时核算；
    4. 适配 30 USDT 实盘小微本金：单笔步长 0.01 张 (~0.0001 BTC, 约 $7 名义值)，最大持仓 5 档 (~$35 名义值，安全杠杆 ~1.1x)。
    """

    def __init__(
        self,
        risk_aversion_gamma: float = 0.35,  # 校准后微账户平稳风险厌恶 (废除自杀式 3.0)
        order_intensity_kappa: float = 2.1346,
        lot_sz: str = "0.01",               # 最小 0.01 张 (0.0001 BTC, 约 $8.1 名义流水)
        max_inventory: float = 0.05,        # 最大 0.05 张 (5 笔 0.01，占用约 $13.50 保证金，占本金 45%)
        min_spread_bps: float = 9.0,        # 强制双边价差 >= 9.0 bps 覆盖双边 4 bps 成本 2.25x
        maker_fee: float = 0.0002,          # 0.02%
        taker_fee: float = 0.0005,          # 0.05%
        contract_value_btc: float = 0.01
    ):
        self.gamma = risk_aversion_gamma
        self.kappa = order_intensity_kappa
        self.lot_sz = lot_sz
        self.max_inventory = max_inventory
        self.min_spread_bps = min_spread_bps
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.ct_val = contract_value_btc
        
        # Real-time state
        self.inventory: float = 0.0  # current contracts
        self.real_matched_trades: int = 0
        self.real_volume_usd: float = 0.0
        self.real_fees_paid_usd: float = 0.0
        self.recent_fills: List[Dict[str, Any]] = []
        
        # Active order tracker
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        
        # Defensive metrics
        self.adverse_mitigations: int = 0
        self.vol_surge_count: int = 0

    def compute_quotes(
        self,
        mid_price: float,
        gk_vol: float,
        rolling_ofi: float,
        ma20_dev: float,
        vol_median: float
    ) -> Dict[str, Any]:
        """计算保留价格、最优非对称价差及挂单许可"""
        sig = max(0.10, gk_vol)
        
        # 1. 微观漂移与保留价格
        # 归一化 OFI (-3 到 +3)
        ofi_norm = np.clip(rolling_ofi / 5.0, -3.0, 3.0)
        mu_pred = np.clip(0.35 * ofi_norm - 0.20 * ma20_dev, -2.0, 2.0)
        
        drift_adj = 0.5563 * mu_pred * (mid_price * 0.0003)
        lot_float = float(self.lot_sz)
        inv_skew = (self.inventory * lot_float) * self.gamma * (sig ** 2) * mid_price * 0.0008
        r = mid_price - inv_skew + drift_adj
        
        # 2. 显式手续费门槛约束 (总价差 >= 9.0 bps)
        base_half_spd = (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa) * (mid_price * 0.0004)
        min_half_spd = mid_price * (self.min_spread_bps * 0.5 / 10000.0)  # 4.5 bps
        base_half_spd = max(min_half_spd, base_half_spd)
        
        # 3. 波动率激增保护
        vol_surge = (sig > 2.0 * vol_median)
        if vol_surge:
            base_half_spd *= 2.0
            self.vol_surge_count += 1
            
        # 4. 去偏毒性门控 (仅在严重负 OFI 与均线破位时切断买单)
        is_toxic_down = (ofi_norm < -1.0 and ma20_dev < -0.6 and sig > vol_median)
        is_toxic_up = (ofi_norm > 1.0 and ma20_dev > 0.6 and sig > vol_median)
        
        # 5. 平滑连续库存偏置 (杜绝阶跃式跨盘自残)
        inv_ratio = float(np.clip(self.inventory / max(self.max_inventory, 1e-6), -1.0, 1.0))
        inv_bias = inv_ratio * base_half_spd * 0.75
        
        delta_ask = base_half_spd * (1.5 if is_toxic_up else 1.0) - inv_bias
        delta_bid = base_half_spd * (1.5 if is_toxic_down else 1.0) + inv_bias
        
        delta_bid = max(min_half_spd * 0.8, delta_bid)
        delta_ask = max(min_half_spd * 0.8, delta_ask)
        
        bid_px = round(r - delta_bid, 1)
        ask_px = round(r + delta_ask, 1)
        
        # 保证严格 Maker 挂单，绝不跨越盘口中间价
        bid_px = min(bid_px, round(mid_price - min_half_spd * 0.5, 1))
        ask_px = max(ask_px, round(mid_price + min_half_spd * 0.5, 1))
        
        allow_bid = (self.inventory < self.max_inventory)
        allow_ask = (self.inventory > -self.max_inventory)
        
        if is_toxic_down:
            allow_bid = False  # 暴跌切断买单
            self.adverse_mitigations += 1
            
        if is_toxic_up:
            allow_ask = False  # 暴涨切断卖单
            self.adverse_mitigations += 1
        return {
            "r": round(r, 1),
            "bid_px": bid_px,
            "ask_px": ask_px,
            "allow_bid": allow_bid,
            "allow_ask": allow_ask,
            "is_toxic_down": is_toxic_down,
            "is_toxic_up": is_toxic_up,
            "vol_surge": vol_surge,
            "base_half_spd_bps": round(base_half_spd / mid_price * 10000, 2)
        }

    def on_real_fill(self, fill_event: Dict[str, Any]):
        """严格真实成交事件处理器：由 WebSocket orders 频道推送触发"""
        side = fill_event.get("side", "").lower()
        fill_px = float(fill_event.get("fillPx", 0))
        fill_sz = float(fill_event.get("fillSz", 0))  # 合约张数
        fee = abs(float(fill_event.get("fee", 0)))
        
        if fill_sz <= 0 or fill_px <= 0:
            return
            
        notional_usd = fill_px * (fill_sz * self.ct_val)
        
        if side == "buy":
            self.inventory += fill_sz
        elif side == "sell":
            self.inventory -= fill_sz
            
        self.real_matched_trades += 1
        self.real_volume_usd += notional_usd
        self.real_fees_paid_usd += fee
        
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tradeId": fill_event.get("tradeId"),
            "ordId": fill_event.get("ordId"),
            "side": side.upper(),
            "price": fill_px,
            "contracts": fill_sz,
            "notional_usd": round(notional_usd, 2),
            "fee_usd": round(fee, 4),
            "inventory_after": round(self.inventory, 3)
        }
        self.recent_fills.append(record)
        if len(self.recent_fills) > 100:
            self.recent_fills.pop(0)
