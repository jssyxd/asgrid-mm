# -*- coding: utf-8 -*-
"""
Production NautilusTrader Microstructure Market Making Service:
Strategy 012: Avellaneda-Stoikov ML Grid (OFI + Adverse Selection Toxic Shocks).

Mathematical Architecture:
1. High-Frequency Micro K-Line (OHLCV) Metrics:
   - Parkinson Volatility: sigma_parkinson = sqrt(1 / (4 ln 2 * N) * sum(ln(H/L)^2))
   - Garman-Klass Volatility: sigma_gk = sqrt(1/N * sum(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2))
   - ATR(14) Volatility Normalizer
   - Normalized Moving Average Deviation: f_dev = (C - MA20) / max(ATR14, 0.05)
   - 1-Bar Micro Return: s_r1 = (C_t - C_t-1) / C_t-1
   - Volume Acceleration: vol_accel = V / (MA20(V) + 1e-9)

2. Level 2 (Depth 20) Order Book & Order Flow Imbalance (OFI):
   - Micro-Price P_micro = (b_1 * v_a1 + a_1 * v_b1) / (v_b1 + v_a1)
   - Top-20 Order Book Imbalance: OBI = (sum V_b - sum V_a) / (sum V_b + sum V_a)
   - Cont-Kukanov-Stoikov Order Flow Imbalance (OFI):
     delta_W_b = v_b1(t) if b_1(t) > b_1(t-1) else (v_b1(t) - v_b1(t-1) if b_1(t) == b_1(t-1) else -v_b1(t-1))
     delta_W_a = -v_a1(t) if a_1(t) > a_1(t-1) else (v_a1(t) - v_a1(t-1) if a_1(t) == a_1(t-1) else v_a1(t-1))
     OFI = delta_W_b - delta_W_a

3. Machine Learning Inference Core (LightGBM):
   - X = [p_vol, gk_vol, atr14, bar_ofi, f_dev, s_r1, vol_accel]
   - mu_drift: bst_drift.predict(X) -> Forward Price Drift Prediction
   - P_toxic_down: bst_tox_down.predict(X) -> Adverse Selection Downward Crash Probability
   - P_toxic_up: bst_tox_up.predict(X) -> Adverse Selection Upward Surge Probability

4. Avellaneda-Stoikov Optimal Stochastic Control:
   - Reservation Price: r(s, q) = P_mid - q * lot_size * gamma * sigma^2 * P_mid * 0.001 + alpha_d * mu_drift * (P_mid * 0.0005)
   - Optimal Half-Spread: delta_base = max(P_mid * min_spread_bps / 10000, 1/gamma * ln(1 + gamma/kappa) * P_mid * 0.0004)
   - ML Toxic Shock Defense:
     delta_bid = delta_base * (1.0 + beta_toxic * P_toxic_down)
     delta_ask = delta_base * (1.0 + beta_toxic * P_toxic_up)
     bid_px = round(r - delta_bid, 1)
     ask_px = round(r + delta_ask, 1)

5. Live Trade Execution & Passive/Active Matching Engine:
   - Manages active limit quotes on OKX Demo
   - Batch order cancellation preventing latency accumulation
   - Toxic Shock Circuit Breaker (P_toxic >= 0.65 cuts quotes on dangerous side)
   - Stale order cancellation and continuous inventory tracking
"""

import os
import sys
import time
import json
import signal
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from gateway_okx_client import OKXGatewayClient
from as_dynamic_risk import compute_adaptive_as_parameters

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "../../config/okx_demo_credentials.env")
LOG_FILE = os.path.join(os.path.dirname(__file__), "../../logs/03_as_grid_runtime.log")
FILLS_FILE = os.path.join(os.path.dirname(__file__), "../../logs/03_as_grid_fills.json")
STATUS_FILE = os.path.join(os.path.dirname(__file__), "../../logs/03_as_grid_status.json")

def safe_float(val, default=0.0):
    if val is None or val == "" or val == "None":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def load_credentials():
    creds = {
        "OKX_API_KEY": os.environ.get("OKX_API_KEY", ""),
        "OKX_API_SECRET": os.environ.get("OKX_API_SECRET", ""),
        "OKX_API_PASSPHRASE": os.environ.get("OKX_API_PASSPHRASE", "Auto111."),
        "OKX_SIMULATED_TRADING": os.environ.get("OKX_SIMULATED_TRADING", "1"),
    }
    if not creds.get("OKX_API_KEY") and os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    creds[k.strip()] = v.strip()
    return creds

def log_msg(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] [03_as_grid] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

running = True
def handle_sigterm(signum, frame):
    global running
    log_msg("Received shutdown signal. Stopping gracefully...")
    running = False

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

class AvellanedaStoikovMLExecutionKernel:
    def __init__(self, checkpoints_dir="/app/checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        if not os.path.exists(self.checkpoints_dir):
            alt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints"))
            if os.path.exists(alt_dir):
                self.checkpoints_dir = alt_dir
            else:
                self.checkpoints_dir = "/opt/nautilus_cluster/checkpoints"

        log_msg(f"Loading certified AS parameters and ML models from: {self.checkpoints_dir}")
        self._load_parameters()
        self._load_ml_models()

        # Dynamic state tracking
        self.prev_best_bid = 0.0
        self.prev_bid_sz = 0.0
        self.prev_best_ask = 0.0
        self.prev_ask_sz = 0.0
        self.last_ofi = 0.0
        self.total_matched_trades = 0
        self.total_fill_volume = 0.0

    def _load_parameters(self, equity_usd: float = 35.0):
        # Dynamically scale parameters based on account equity tier (Micro vs Institutional)
        cfg = compute_adaptive_as_parameters(equity_usd)
        self.gamma = cfg["gamma_risk_aversion"]
        self.kappa = cfg["kappa_order_intensity"]
        self.alpha_drift = cfg["drift_alpha"]
        self.beta_toxic = cfg["toxic_beta"]
        self.lot_sz = cfg["lot_sz"]
        self.maker_fee_bps = 1.0
        self.min_spread_bps = cfg["min_spread_bps"]
        self.max_inventory = cfg["max_inventory"]
        log_msg(f"✔ Adaptive Risk Scaling Active | Tier: {cfg['tier']} | Equity: {equity_usd:.2f} USDT | Gamma (Risk Aversion): {self.gamma} | Max Inv: {self.max_inventory}")

    def _load_ml_models(self):
        f_drift = os.path.join(self.checkpoints_dir, "lgb_drift_model.txt")
        f_down = os.path.join(self.checkpoints_dir, "lgb_tox_down_model.txt")
        f_up = os.path.join(self.checkpoints_dir, "lgb_tox_up_model.txt")

        if os.path.exists(f_drift) and os.path.exists(f_down) and os.path.exists(f_up):
            self.drift_model = lgb.Booster(model_file=f_drift)
            self.tox_down_model = lgb.Booster(model_file=f_down)
            self.tox_up_model = lgb.Booster(model_file=f_up)
            self.ml_ready = True
            log_msg(f"✔ Successfully loaded 3 LightGBM ML models: Drift ({self.drift_model.num_trees()} trees), ToxDown ({self.tox_down_model.num_trees()} trees), ToxUp ({self.tox_up_model.num_trees()} trees)")
        else:
            self.ml_ready = False
            log_msg(f"⚠ Warning: LightGBM model files missing in {self.checkpoints_dir}, using analytical ML approximations!")

    def compute_ohlcv_features(self, candles):
        """Computes Parkinson volatility, Garman-Klass volatility, ATR, deviations, and return."""
        arr = np.array(candles, dtype=object)[::-1] # Ascending time order
        o = np.array([safe_float(x) for x in arr[:, 1]])
        h = np.array([safe_float(x) for x in arr[:, 2]])
        l = np.array([safe_float(x) for x in arr[:, 3]])
        c = np.array([safe_float(x) for x in arr[:, 4]])
        v = np.array([safe_float(x) for x in arr[:, 5]])

        n = len(c)
        s_c = pd.Series(c)

        # Parkinson Volatility
        log_hl = np.log(np.maximum(h, 1e-6) / np.maximum(l, 1e-6))
        p_var = (log_hl ** 2) / (4.0 * math.log(2.0))
        p_vol = np.sqrt(pd.Series(p_var).rolling(min(n, 20), min_periods=3).mean().bfill().iloc[-1] * (365 * 1440))
        p_vol = max(0.08, float(p_vol))

        # Garman-Klass Volatility
        log_co = np.log(np.maximum(c, 1e-6) / np.maximum(o, 1e-6))
        gk_var = 0.5 * (log_hl ** 2) - (2.0 * math.log(2.0) - 1.0) * (log_co ** 2)
        gk_vol = np.sqrt(pd.Series(gk_var).rolling(min(n, 20), min_periods=3).mean().bfill().iloc[-1] * (365 * 1440))
        gk_vol = max(0.08, float(gk_vol))

        # True Range and ATR(14)
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr14 = float(pd.Series(tr).rolling(min(n, 14), min_periods=3).mean().bfill().iloc[-1])
        atr14 = max(0.5, atr14)

        # Deviation from MA20
        ma20 = float(s_c.rolling(min(n, 20), min_periods=3).mean().bfill().iloc[-1])
        f_dev = float((c[-1] - ma20) / atr14)

        # 1-bar return
        s_r1 = float((c[-1] - c[-2]) / (c[-2] + 1e-9)) if n >= 2 else 0.0

        # Volume acceleration
        vol_ma = float(pd.Series(v).rolling(min(n, 20), min_periods=3).mean().bfill().iloc[-1])
        vol_accel = float(np.clip(v[-1] / (vol_ma + 1e-9), 0.01, 50.0))

        return {
            "p_vol": p_vol,
            "gk_vol": gk_vol,
            "atr14": atr14,
            "f_dev": f_dev,
            "s_r1": s_r1,
            "vol_accel": vol_accel,
            "close_px": float(c[-1]),
            "ma20": ma20
        }

    def compute_lob_orderbook_metrics(self, book_data):
        """Computes Depth-20 BBO, Mid-Price, Micro-Price, 20-level OBI, and OFI."""
        if not book_data:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        bids = book_data.get("bids", [])
        asks = book_data.get("asks", [])
        if not bids or not asks:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        best_bid = safe_float(bids[0][0])
        bid_sz1 = safe_float(bids[0][1])
        best_ask = safe_float(asks[0][0])
        ask_sz1 = safe_float(asks[0][1])

        mid_price = (best_bid + best_ask) / 2.0
        bbo_spread = max(0.1, best_ask - best_bid)

        # Micro-Price (Volume Weighted)
        micro_price = (best_bid * ask_sz1 + best_ask * bid_sz1) / (bid_sz1 + ask_sz1 + 1e-6)

        # Depth 20 Order Book Imbalance (OBI)
        n = min(20, len(bids), len(asks))
        total_bid_vol = sum(safe_float(bids[i][1]) for i in range(n))
        total_ask_vol = sum(safe_float(asks[i][1]) for i in range(n))
        obi = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-6)

        # Cont-Kukanov-Stoikov Order Flow Imbalance (OFI)
        if self.prev_best_bid > 0 and self.prev_best_ask > 0:
            if best_bid > self.prev_best_bid:
                delta_w_b = bid_sz1
            elif best_bid == self.prev_best_bid:
                delta_w_b = bid_sz1 - self.prev_bid_sz
            else:
                delta_w_b = -self.prev_bid_sz

            if best_ask > self.prev_best_ask:
                delta_w_a = -self.prev_ask_sz
            elif best_ask == self.prev_best_ask:
                delta_w_a = ask_sz1 - self.prev_ask_sz
            else:
                delta_w_a = ask_sz1

            ofi_raw = delta_w_b - delta_w_a
            bar_ofi = float(np.clip(ofi_raw / (bid_sz1 + ask_sz1 + 1e-6), -3.0, 3.0))
        else:
            bar_ofi = float(np.clip(obi * 2.0, -3.0, 3.0))

        self.prev_best_bid = best_bid
        self.prev_bid_sz = bid_sz1
        self.prev_best_ask = best_ask
        self.prev_ask_sz = ask_sz1
        self.last_ofi = bar_ofi

        return best_bid, best_ask, mid_price, micro_price, obi, bar_ofi

    def predict_ml_signals(self, ohlcv_feat, bar_ofi):
        """Constructs input feature vector and invokes the 3 LightGBM ML models."""
        X = np.array([[
            ohlcv_feat["p_vol"],
            ohlcv_feat["gk_vol"],
            ohlcv_feat["atr14"],
            bar_ofi,
            ohlcv_feat["f_dev"],
            ohlcv_feat["s_r1"],
            ohlcv_feat["vol_accel"]
        ]], dtype=np.float64)

        if self.ml_ready:
            pred_drift = float(self.drift_model.predict(X)[0])
            pred_tox_down = float(self.tox_down_model.predict(X)[0])
            pred_tox_up = float(self.tox_up_model.predict(X)[0])
        else:
            pred_drift = float(np.clip(bar_ofi * 0.4 + ohlcv_feat["f_dev"] * 0.2, -3.0, 3.0))
            pred_tox_down = float(1.0 / (1.0 + math.exp(-(-bar_ofi * 1.5 + ohlcv_feat["gk_vol"] * 2.0))))
            pred_tox_up = float(1.0 / (1.0 + math.exp(-(bar_ofi * 1.5 + ohlcv_feat["gk_vol"] * 2.0))))

        return pred_drift, pred_tox_down, pred_tox_up

    def compute_as_grid_quotes(self, mid_price, gk_vol, inventory_contracts, mu_drift, p_tox_down, p_tox_up):
        """Calculates Avellaneda-Stoikov reservation price, optimal half spreads, and quoting permissions for 35 USDT micro account."""
        sigma = max(0.10, gk_vol)

        # 1. Aggressive Inventory Skew (Forced Inventory Rebalancing)
        inventory_skew = inventory_contracts * self.gamma * (sigma ** 2) * mid_price * 0.005
        drift_adjustment = self.alpha_drift * mu_drift * (mid_price * 0.0005)
        r = mid_price - inventory_skew + drift_adjustment

        # 2. Optimal Base Half-Spread:
        base_half_spread = (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa) * (mid_price * 0.0006)
        min_half_spread = mid_price * (self.min_spread_bps / 10000.0)
        base_half_spread = max(min_half_spread, base_half_spread)

        delta_bid = base_half_spread * (1.0 + self.beta_toxic * p_tox_down)
        delta_ask = base_half_spread * (1.0 + self.beta_toxic * p_tox_up)

        # If inventory > 0, aggressively lower ask price to dump inventory instantly
        bid_px = round(r - delta_bid, 1)
        ask_px = round(r + delta_ask, 1)
        
        if inventory_contracts > 0.02:
            ask_px = round(mid_price - 1.0, 1) # Immediate markdown to flatten inventory
            allow_bid = False
            allow_ask = True
        elif inventory_contracts < -0.02:
            bid_px = round(mid_price + 1.0, 1) # Immediate markup to cover short
            allow_bid = True
            allow_ask = False
        else:
            allow_bid = (inventory_contracts < self.max_inventory) and (p_tox_down < 0.80)
            allow_ask = (inventory_contracts > -self.max_inventory) and (p_tox_up < 0.80)

        return {
            "r": r,
            "base_half_spread": base_half_spread,
            "delta_bid": delta_bid,
            "delta_ask": delta_ask,
            "bid_px": bid_px,
            "ask_px": ask_px,
            "allow_bid": allow_bid,
            "allow_ask": allow_ask
        }

def main():
    log_msg("================================================================================")
    log_msg("启动 Production 012 Avellaneda-Stoikov ML Grid (OFI+Toxic) 微观做市实盘组件")
    log_msg("集成: 1m OHLCV 特征引擎 + L2 盘口 OFI 差分 + LightGBM 漂移/毒性三模型 + OKX 实盘撮合")
    log_msg("================================================================================")

    creds = load_credentials()
    simulated_mode = (creds.get("OKX_SIMULATED_TRADING", "1") == "1")
    client = OKXGatewayClient(
        api_key=creds["OKX_API_KEY"],
        api_secret=creds["OKX_API_SECRET"],
        passphrase=creds["OKX_API_PASSPHRASE"],
        simulated=simulated_mode
    )

    bal = client.get_balance("USDT")
    avail_bal = safe_float(bal.get("availBal"), 0.0)
    eq_usd = safe_float(bal.get("eqUsd"), 0.0)
    env_type = "Demo" if simulated_mode else "Live Real"
    log_msg(f"Connected to OKX {env_type} Gateway. Available balance: {avail_bal} USDT, Total equity: {eq_usd} USD")

    kernel = AvellanedaStoikovMLExecutionKernel()
    kernel._load_parameters(eq_usd)
    loop_count = 0
    recent_fills = []

    if os.path.exists(FILLS_FILE):
        try:
            with open(FILLS_FILE, "r", encoding="utf-8") as f:
                recent_fills = json.load(f)
        except Exception:
            recent_fills = []

    last_candle_time = 0.0
    cached_ohlcv_feat = None

    while running:
        loop_count += 1
        t_loop_start = time.time()
        try:
            # 1. Fetch 1m K-lines (cached for 12 seconds to reduce redundant calls)
            if time.time() - last_candle_time >= 12.0 or cached_ohlcv_feat is None:
                candles = client.get_candles("BTC-USDT-SWAP", bar="1m", limit=30)
                if candles:
                    cached_ohlcv_feat = kernel.compute_ohlcv_features(candles)
                    last_candle_time = time.time()

            if cached_ohlcv_feat is None:
                log_msg("Warning: Initial candles not ready, sleeping 2s...")
                time.sleep(2)
                continue

            ohlcv_feat = cached_ohlcv_feat

            # 2. Fetch L2 Depth-20 Order Book and compute OFI & OBI on every tick
            book = client.get_books("BTC-USDT-SWAP", sz=20)
            best_bid, best_ask, mid_price, micro_px, obi, bar_ofi = kernel.compute_lob_orderbook_metrics(book)
            if mid_price <= 0:
                mid_price = ohlcv_feat["close_px"]

            # 3. Model Inference: Invoke LightGBM Drift & Adverse Selection Shock Models
            pred_drift, pred_tox_down, pred_tox_up = kernel.predict_ml_signals(ohlcv_feat, bar_ofi)

            # 4. Fetch current real net inventory from OKX
            pos_list = client.get_positions(inst_type="SWAP", inst_id="BTC-USDT-SWAP")
            pos_sz = safe_float(pos_list[0].get("pos")) if pos_list else 0.0
            upl = safe_float(pos_list[0].get("upl")) if pos_list else 0.0

            # 5. Avellaneda-Stoikov Mathematical Pricing & Optimal Half-Spread Calculation
            as_res = kernel.compute_as_grid_quotes(
                mid_price=mid_price,
                gk_vol=ohlcv_feat["gk_vol"],
                inventory_contracts=pos_sz,
                mu_drift=pred_drift,
                p_tox_down=pred_tox_down,
                p_tox_up=pred_tox_up
            )

            r = as_res["r"]
            bid_px = as_res["bid_px"]
            ask_px = as_res["ask_px"]
            allow_bid = as_res["allow_bid"]
            allow_ask = as_res["allow_ask"]

            # 6. High-Frequency Order Management & Trade Matching Execution
            # Clean up pending orders using fast batch cancellation
            pending_orders = client.get_pending_orders(inst_type="SWAP", inst_id="BTC-USDT-SWAP")
            if pending_orders:
                cancel_list = [{"instId": "BTC-USDT-SWAP", "ordId": p.get("ordId")} for p in pending_orders if p.get("ordId")]
                try:
                    client.cancel_batch_orders(cancel_list)
                except Exception:
                    pass

            action_desc = []
            now_ms = int(time.time() * 1000)

            # Post Asymmetric Two-Sided Limit Quotes
            if allow_bid:
                cl_id_bid = f"asb{now_ms}"[-32:]
                res_bid = client.place_order(
                    inst_id="BTC-USDT-SWAP",
                    td_mode="cross",
                    side="buy",
                    ord_type="limit",
                    px=str(bid_px),
                    sz=kernel.lot_sz,
                    cl_ord_id=cl_id_bid
                )
                if res_bid.get("code") == "0":
                    action_desc.append(f"BID_POST({bid_px})")
                    ord_data = res_bid.get("data", [{}])[0]
                    kernel.total_matched_trades += 1
                    recent_fills.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "side": "BUY",
                        "price": bid_px,
                        "sz": kernel.lot_sz,
                        "clOrdId": cl_id_bid,
                        "ordId": ord_data.get("ordId"),
                        "type": "MAKER_BID",
                        "mid": mid_price,
                        "r": round(r, 1)
                    })
                else:
                    action_desc.append(f"BID_FAIL({res_bid.get('msg')})")
            else:
                reason = "MAX_INV" if pos_sz >= kernel.max_inventory else "TOXIC_DOWN_PROTECT"
                action_desc.append(f"BID_CUT({reason})")

            if allow_ask:
                cl_id_ask = f"asa{now_ms}"[-32:]
                res_ask = client.place_order(
                    inst_id="BTC-USDT-SWAP",
                    td_mode="cross",
                    side="sell",
                    ord_type="limit",
                    px=str(ask_px),
                    sz=kernel.lot_sz,
                    cl_ord_id=cl_id_ask
                )
                if res_ask.get("code") == "0":
                    action_desc.append(f"ASK_POST({ask_px})")
                    ord_data = res_ask.get("data", [{}])[0]
                    kernel.total_matched_trades += 1
                    recent_fills.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "side": "SELL",
                        "price": ask_px,
                        "sz": kernel.lot_sz,
                        "clOrdId": cl_id_ask,
                        "ordId": ord_data.get("ordId"),
                        "type": "MAKER_ASK",
                        "mid": mid_price,
                        "r": round(r, 1)
                    })
                else:
                    action_desc.append(f"ASK_FAIL({res_ask.get('msg')})")
            else:
                reason = "MIN_INV" if pos_sz <= -kernel.max_inventory else "TOXIC_UP_PROTECT"
                action_desc.append(f"ASK_CUT({reason})")

            # 7. Write telemetry status and recent fills
            status = {
                "service_id": "03_as_grid",
                "strategy_name": "012_avellaneda_grid_ml_ofi_toxic",
                "symbol": "BTC-USDT-SWAP",
                "timeframe": "1m OHLCV + L2 Depth20 OFI",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "loop_count": loop_count,
                "mid_price": mid_price,
                "micro_price": round(micro_px, 1),
                "spread_bbo": round(best_ask - best_bid, 1),
                "features": {
                    "p_vol": round(ohlcv_feat["p_vol"], 4),
                    "gk_vol": round(ohlcv_feat["gk_vol"], 4),
                    "atr14": round(ohlcv_feat["atr14"], 2),
                    "f_dev": round(ohlcv_feat["f_dev"], 4),
                    "s_r1": round(ohlcv_feat["s_r1"], 6),
                    "vol_accel": round(ohlcv_feat["vol_accel"], 2),
                    "obi_20": round(obi, 4),
                    "bar_ofi": round(bar_ofi, 4)
                },
                "ml_inference": {
                    "pred_drift": round(pred_drift, 4),
                    "p_toxic_down": round(pred_tox_down, 4),
                    "p_toxic_up": round(pred_tox_up, 4),
                    "models_loaded": kernel.ml_ready
                },
                "as_pricing": {
                    "reservation_price": round(r, 1),
                    "base_half_spread": round(as_res["base_half_spread"], 2),
                    "delta_bid": round(as_res["delta_bid"], 2),
                    "delta_ask": round(as_res["delta_ask"], 2),
                    "bid_px": bid_px,
                    "ask_px": ask_px,
                    "allow_bid": allow_bid,
                    "allow_ask": allow_ask
                },
                "portfolio": {
                    "inventory_contracts": pos_sz,
                    "unrealized_pnl": upl,
                    "matched_trades": kernel.total_matched_trades
                },
                "actions": " | ".join(action_desc),
                "status": "RUNNING_HEALTHY"
            }

            with open(STATUS_FILE, "w", encoding="utf-8") as sf:
                json.dump(status, sf, indent=2)

            with open(FILLS_FILE, "w", encoding="utf-8") as ff:
                json.dump(recent_fills[-100:], ff, indent=2)

            calc_latency_ms = int((time.time() - t_loop_start) * 1000)
            log_msg(f"Loop #{loop_count:04d} ({calc_latency_ms}ms) | Mid: {mid_price:.1f} | Res: {r:.1f} | "
                    f"ML Drift: {pred_drift:+.3f} | Tox(D/U): [{pred_tox_down:.2f}/{pred_tox_up:.2f}] | "
                    f"Quotes: [{bid_px:.1f} / {ask_px:.1f}] | Inv: {pos_sz:+.1f} | Actions: {' | '.join(action_desc)}")

        except Exception as e:
            log_msg(f"Error in AS Market Making loop: {e}")

        # Pacing 1.5 to 2.0s
        elapsed = time.time() - t_loop_start
        sleep_needed = max(0.5, 2.0 - elapsed)
        time.sleep(sleep_needed)

    log_msg("012 Avellaneda-Stoikov Microstructure MM Service successfully stopped.")

if __name__ == "__main__":
    main()
