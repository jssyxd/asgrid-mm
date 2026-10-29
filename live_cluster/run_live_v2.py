# -*- coding: utf-8 -*-
"""
run_live.py - Production Live Service for Strategy 012 (Account: accountbot3)
Avellaneda-Stoikov ML Grid V2:
- Real-time WebSocket Orderbook & Trades
- Private WebSocket Order Fill Tracking
- NautilusTrader-compatible State Machine
- Strict Real Trade Volume & PnL Accounting
"""

import socket
# Force IPv4 socket resolution globally for OKX requests
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(*args, **kwargs):
    responses = _orig_getaddrinfo(*args, **kwargs)
    ipv4_res = [r for r in responses if r[0] == socket.AF_INET]
    return ipv4_res if ipv4_res else responses
socket.getaddrinfo = _ipv4_getaddrinfo

import os
import sys
import time
import json
import signal
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from gateway_okx_client import OKXGatewayClient
from as_nautilus_strategy import NautilusAvellanedaV2MMStrategy
from okx_ws_manager import OKXWebSocketManager

SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SERVICE_DIR, "config/accountbot3_credentials.env")
LOG_DIR = os.path.join(SERVICE_DIR, "logs")
STATUS_FILE = os.path.join(LOG_DIR, "07_as_grid_status.json")
FILLS_FILE = os.path.join(LOG_DIR, "07_as_grid_fills.json")
LOG_FILE = os.path.join(LOG_DIR, "07_as_grid_runtime.log")

os.makedirs(LOG_DIR, exist_ok=True)

def load_credentials():
    creds = {
        "OKX_API_KEY": os.environ.get("OKX_API_KEY", ""),
        "OKX_API_SECRET": os.environ.get("OKX_API_SECRET", ""),
        "OKX_API_PASSPHRASE": os.environ.get("OKX_API_PASSPHRASE", ""),
        "OKX_SIMULATED_TRADING": os.environ.get("OKX_SIMULATED_TRADING", "0"),
        "ACCOUNT_LABEL": os.environ.get("ACCOUNT_LABEL", "accountbot3")
    }
    if not creds.get("OKX_API_KEY") and os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    creds[k.strip()] = v.strip().strip('"').strip("'")
    return creds

def log_msg(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] [07_as_grid_accountbot3] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

running = True
def handle_shutdown(signum, frame):
    global running
    log_msg("Received shutdown signal. Stopping gracefully...")
    running = False

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

def compute_gk_vol(candles):
    """Causal Garman-Klass Volatility, MA20 Deviation, and ADX(14) Trend State."""
    arr = np.array(candles, dtype=object)
    arr = arr[::-1]  # oldest first
    open_ = arr[:, 1].astype(float)
    high = arr[:, 2].astype(float)
    low = arr[:, 3].astype(float)
    close = arr[:, 4].astype(float)
    n = len(close)
    
    log_hl = np.log(np.maximum(high, 1e-6) / np.maximum(low, 1e-6))
    log_co = np.log(np.maximum(close, 1e-6) / np.maximum(open_, 1e-6))
    gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    
    # 20-bar causal mean
    gk_rolling = pd.Series(gk_var).rolling(20, min_periods=1).mean().values
    annual_gk = np.sqrt(np.maximum(gk_rolling[-1], 1e-9) * (365.25 * 1440))
    
    s_close = pd.Series(close)
    ma20 = s_close.rolling(20, min_periods=1).mean().values[-1]
    cur_c = close[-1]
    atr_approx = np.mean(high[-14:] - low[-14:]) if len(high) >= 14 else (cur_c * 0.002)
    ma20_dev = (cur_c - ma20) / max(atr_approx, 0.05)
    
    # Causal ADX(14) calculation
    adx14 = 15.0
    trend_dir = 0
    if n >= 15:
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        delta_h = high[1:] - high[:-1]
        delta_l = low[:-1] - low[1:]
        plus_dm = np.where((delta_h > delta_l) & (delta_h > 0), delta_h, 0.0)
        minus_dm = np.where((delta_l > delta_h) & (delta_l > 0), delta_l, 0.0)
        
        smooth_tr = pd.Series(tr).rolling(min(n-1, 14), min_periods=3).mean().bfill()
        smooth_plus = pd.Series(plus_dm).rolling(min(n-1, 14), min_periods=3).mean().bfill()
        smooth_minus = pd.Series(minus_dm).rolling(min(n-1, 14), min_periods=3).mean().bfill()
        
        plus_di = 100.0 * (smooth_plus / (smooth_tr + 1e-9))
        minus_di = 100.0 * (smooth_minus / (smooth_tr + 1e-9))
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        adx14 = float(dx.rolling(min(n-1, 14), min_periods=3).mean().bfill().iloc[-1])
        curr_plus_di = float(plus_di.iloc[-1])
        curr_minus_di = float(minus_di.iloc[-1])
        
        if curr_plus_di > curr_minus_di and cur_c > ma20:
            trend_dir = 1
        elif curr_minus_di > curr_plus_di and cur_c < ma20:
            trend_dir = -1
            
    return float(annual_gk), float(ma20_dev), cur_c, float(adx14), int(trend_dir)
def main():
    log_msg("=== Starting NautilusTrader Avellaneda-Stoikov V2 Live Service ===")
    creds = load_credentials()
    simulated = (creds.get("OKX_SIMULATED_TRADING", "0") == "1")
    
    client = OKXGatewayClient(
        api_key=creds["OKX_API_KEY"],
        api_secret=creds["OKX_API_SECRET"],
        passphrase=creds["OKX_API_PASSPHRASE"],
        simulated=simulated
    )
    
    # Verify account & initial capital
    bal = client.get_balance("USDT")
    avail_bal = float(bal.get("availBal", bal.get("cashBal", 0)))
    log_msg(f"Verified connection to OKX {'Demo' if simulated else 'Real'}. Available Capital: {avail_bal} USDT")
    
    # Start WebSocket Manager
    ws_mgr = OKXWebSocketManager(
        api_key=creds["OKX_API_KEY"],
        api_secret=creds["OKX_API_SECRET"],
        passphrase=creds["OKX_API_PASSPHRASE"],
        inst_id="BTC-USDT-SWAP"
    )
    log_msg("Starting OKX WebSocket streams (books5, trades, orders, account)...")
    ws_mgr.start()
    
    strategy = NautilusAvellanedaV2MMStrategy(
        lot_sz="0.01",  # 0.01 张 (0.0001 BTC, 约 $8.1 名义流水)
        max_inventory=0.05,  # 最大 0.05 张，占用约 $13.50 保证金，占 30U 本金 45%
        min_spread_bps=9.0,  # 强制总价差 >= 9.0 bps (半价差 >= 4.5 bps 覆盖双向 0.04% 成本)
        risk_aversion_gamma=0.35,  # 校准后微账户平稳风险厌恶 (废除 3.00)
    )
    
    # Sync initial position from exchange
    init_positions = client.get_positions(inst_type="SWAP", inst_id="BTC-USDT-SWAP")
    if init_positions:
        strategy.inventory = float(init_positions[0].get("pos", 0))
        log_msg(f"Synced initial live position: {strategy.inventory} contracts")
        
    loop_count = 0
    last_candle_fetch = 0.0
    cached_gk_vol = 0.25
    cached_ma20_dev = 0.0
    cached_adx14 = 15.0
    cached_trend_dir = 0
    vol_median = 0.25
    recent_telemetry_history = []
    
    # Post-fill cooldown state
    last_pos = strategy.inventory
    last_fill_time_bid = 0.0
    last_fill_time_ask = 0.0
    FILL_COOLDOWN_SEC = 5.0
    
    time.sleep(2)  # Warm up WebSocket
    
    while running:
        loop_count += 1
        t_start = time.time()
        try:
            # 1. Process REAL fill events delivered by private WebSocket
            new_fills = ws_mgr.pop_new_fills()
            for f in new_fills:
                strategy.on_real_fill(f)
                log_msg(f"⭐ [REAL FILL CONFIRMED] {f['side']} {f['fillSz']} contracts @ {f['fillPx']} | Fee: {f['fee']} {f['feeCcy']}")
                
            # Post-fill cooldown trigger on inventory position change
            if strategy.inventory > last_pos:
                last_fill_time_bid = time.time()
                log_msg(f"⚡ Fill detected on BUY side ({last_pos} -> {strategy.inventory}), activating {FILL_COOLDOWN_SEC}s cooldown")
            elif strategy.inventory < last_pos:
                last_fill_time_ask = time.time()
                log_msg(f"⚡ Fill detected on SELL side ({last_pos} -> {strategy.inventory}), activating {FILL_COOLDOWN_SEC}s cooldown")
            last_pos = strategy.inventory
            
            cooldown_bid = (time.time() - last_fill_time_bid) < FILL_COOLDOWN_SEC
            cooldown_ask = (time.time() - last_fill_time_ask) < FILL_COOLDOWN_SEC
            
            # 2. Update K-line features periodically (every 15s)
            if time.time() - last_candle_fetch >= 15.0:
                candles = client.get_candles("BTC-USDT-SWAP", bar="1m", limit=30)
                if candles:
                    cached_gk_vol, cached_ma20_dev, last_close, cached_adx14, cached_trend_dir = compute_gk_vol(candles)
                    last_candle_fetch = time.time()
                    
            # 3. Retrieve latest order book & OFI from WebSocket
            book = ws_mgr.get_latest_book()
            mid_px = book["mid"]
            if mid_px <= 0:
                mid_px = client.get_ticker("BTC-USDT-SWAP").get("last", 0)
                mid_px = float(mid_px) if mid_px else 69000.0
                
            rolling_ofi = ws_mgr.get_rolling_ofi()
            
            # 4. Strategy pricing & quote calculation (with ADX and Cooldown)
            quote_res = strategy.compute_quotes(
                mid_price=mid_px,
                gk_vol=cached_gk_vol,
                rolling_ofi=rolling_ofi,
                ma20_dev=cached_ma20_dev,
                vol_median=vol_median,
                adx14=cached_adx14,
                trend_dir=cached_trend_dir,
                cooldown_bid=cooldown_bid,
                cooldown_ask=cooldown_ask
            )
            
            bid_px = quote_res["bid_px"]
            ask_px = quote_res["ask_px"]
            allow_bid = quote_res["allow_bid"]
            allow_ask = quote_res["allow_ask"]
            
            # 5. 队列感知订单生命周期管理 (Queue Priority Preservation with dynamic tolerance)
            PRICE_TOLERANCE_USDT = max(1.5, mid_px * 0.0001)
            pending = client.get_pending_orders(inst_type="SWAP", inst_id="BTC-USDT-SWAP")
            existing_buy = None
            existing_sell = None
            cancel_list = []
            
            for p in pending:
                side_p = p.get("side", "").lower()
                px_p = float(p.get("px", 0))
                ord_id = p.get("ordId")
                
                # 检查现有买单：若在允许报价且偏离 <= PRICE_TOLERANCE_USDT，保留排队优势！
                if side_p == "buy":
                    if allow_bid and abs(px_p - bid_px) <= PRICE_TOLERANCE_USDT:
                        existing_buy = p
                    else:
                        cancel_list.append({"instId": "BTC-USDT-SWAP", "ordId": ord_id})
                elif side_p == "sell":
                    if allow_ask and abs(px_p - ask_px) <= PRICE_TOLERANCE_USDT:
                        existing_sell = p
                    else:
                        cancel_list.append({"instId": "BTC-USDT-SWAP", "ordId": ord_id})
                try:
                    client.cancel_batch_orders(cancel_list)
                except Exception:
                    pass
                    
            now_ms = int(time.time() * 1000)
            actions = []
            
            # 买单：仅在无排队中买单时才 Post-Only 挂单
            if allow_bid:
                if existing_buy:
                    actions.append(f"BID_QUEUE_HOLD({existing_buy.get('px')})")
                else:
                    cl_id_bid = f"asb{now_ms}"[-32:]
                    res_b = client.place_order(
                        inst_id="BTC-USDT-SWAP",
                        td_mode="cross",
                        side="buy",
                        ord_type="post_only",
                        px=str(bid_px),
                        sz=strategy.lot_sz,
                        cl_ord_id=cl_id_bid
                    )
                    if res_b.get("code") == "0":
                        actions.append(f"BID_POST({bid_px})")
                    else:
                        actions.append(f"BID_REJ({res_b.get('msg')})")
            else:
                actions.append("BID_CUT")
                
            # 卖单：仅在无排队中卖单时才 Post-Only 挂单
            if allow_ask:
                if existing_sell:
                    actions.append(f"ASK_QUEUE_HOLD({existing_sell.get('px')})")
                else:
                    cl_id_ask = f"asa{now_ms}"[-32:]
                    res_a = client.place_order(
                        inst_id="BTC-USDT-SWAP",
                        td_mode="cross",
                        side="sell",
                        ord_type="post_only",
                        px=str(ask_px),
                        sz=strategy.lot_sz,
                        cl_ord_id=cl_id_ask
                    )
                    if res_a.get("code") == "0":
                        actions.append(f"ASK_POST({ask_px})")
                    else:
                        actions.append(f"ASK_REJ({res_a.get('msg')})")
            else:
                actions.append("ASK_CUT")
            # 6. Status and fills persistence
            status_data = {
                "service_id": "07_as_grid_accountbot3",
                "strategy": "Avellaneda-Stoikov ML Grid V2",
                "account": creds.get("ACCOUNT_LABEL", "accountbot3"),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "loop_count": loop_count,
                "current_mid": mid_px,
                "bid_quote": bid_px,
                "ask_quote": ask_px,
                "inventory_contracts": round(strategy.inventory, 3),
                "real_matched_trades": strategy.real_matched_trades,
                "real_volume_usd": round(strategy.real_volume_usd, 2),
                "real_fees_paid_usd": round(strategy.real_fees_paid_usd, 4),
                "adverse_mitigations": strategy.adverse_mitigations,
                "vol_surge_count": strategy.vol_surge_count,
                "actions": actions,
                "status": "HEALTHY_ACTIVE"
            }
            with open(STATUS_FILE, "w", encoding="utf-8") as sf:
                json.dump(status_data, sf, indent=2)
                
            with open(FILLS_FILE, "w", encoding="utf-8") as ff:
                json.dump(strategy.recent_fills, ff, indent=2)
                
            if loop_count % 10 == 0:
                log_msg(f"Heartbeat #{loop_count}: Mid={mid_px}, Inv={strategy.inventory}, RealFills={strategy.real_matched_trades}, RealVol=${strategy.real_volume_usd:.2f}, Actions={actions}")
                
        except Exception as e:
            log_msg(f"Exception in main loop: {e}")
            
        elapsed = time.time() - t_start
        sleep_time = max(0.5, 2.0 - elapsed)
        time.sleep(sleep_time)

    # Cleanup upon shutdown
    log_msg("Cleaning up active orders upon shutdown...")
    try:
        pending = client.get_pending_orders(inst_type="SWAP", inst_id="BTC-USDT-SWAP")
        if pending:
            cancel_list = [{"instId": "BTC-USDT-SWAP", "ordId": p.get("ordId")} for p in pending if p.get("ordId")]
            client.cancel_batch_orders(cancel_list)
    except Exception:
        pass
    ws_mgr.stop()
    log_msg("NautilusTrader V2 service successfully terminated.")

if __name__ == "__main__":
    main()
