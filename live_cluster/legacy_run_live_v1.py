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

sys.path.append(os.path.dirname(__file__))
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
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

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
            cand1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../checkpoints"))
            cand2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints"))
            if os.path.exists(cand1):
                self.checkpoints_dir = cand1
            elif os.path.exists(cand2):
                self.checkpoints_dir = cand2
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

        # Causal ADX(14) calculation for structural macro-trend detection
        if n >= 15:
            delta_h = h[1:] - h[:-1]
            delta_l = l[:-1] - l[1:]
            plus_dm = np.where((delta_h > delta_l) & (delta_h > 0), delta_h, 0.0)
            minus_dm = np.where((delta_l > delta_h) & (delta_l > 0), delta_l, 0.0)
            
            smooth_tr = pd.Series(tr[1:]).rolling(min(n-1, 14), min_periods=3).mean().bfill()
            smooth_plus = pd.Series(plus_dm).rolling(min(n-1, 14), min_periods=3).mean().bfill()
            smooth_minus = pd.Series(minus_dm).rolling(min(n-1, 14), min_periods=3).mean().bfill()
            
            plus_di = 100.0 * (smooth_plus / (smooth_tr + 1e-9))
            minus_di = 100.0 * (smooth_minus / (smooth_tr + 1e-9))
            dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
            adx14 = float(dx.rolling(min(n-1, 14), min_periods=3).mean().bfill().iloc[-1])
            curr_plus_di = float(plus_di.iloc[-1])
            curr_minus_di = float(minus_di.iloc[-1])
            
            if curr_plus_di > curr_minus_di and c[-1] > ma20:
                trend_dir = 1
            elif curr_minus_di > curr_plus_di and c[-1] < ma20:
                trend_dir = -1
            else:
                trend_dir = 0
        else:
            adx14 = 15.0
            trend_dir = 0

        return {
            "p_vol": p_vol,
            "gk_vol": gk_vol,
            "atr14": atr14,
            "f_dev": f_dev,
            "s_r1": s_r1,
            "vol_accel": vol_accel,
            "close_px": float(c[-1]),
            "ma20": ma20,
            "adx14": adx14,
            "trend_dir": trend_dir
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

    def compute_as_grid_quotes(
        self,
        mid_price,
        gk_vol,
        inventory_contracts,
        mu_drift,
        p_tox_down,
        p_tox_up,
        adx14=0.0,
        trend_dir=0,
        cooldown_bid=False,
        cooldown_ask=False
    ):
        """Calculates Avellaneda-Stoikov quotes with smooth inventory skew, ADX trend protection, and post-fill cooldown."""
        sigma = max(0.10, gk_vol)

        # 1. Smooth Inventory Skew (Calibrated Reservation Price tilt, avoiding gamma blowouts)
        inventory_skew = inventory_contracts * self.gamma * (sigma ** 2) * mid_price * 0.001
        drift_adjustment = self.alpha_drift * mu_drift * (mid_price * 0.0005)
        r = mid_price - inventory_skew + drift_adjustment

        # 2. Optimal Base Half-Spread with mandatory fee hurdle
        base_half_spread = (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa) * (mid_price * 0.0006)
        min_half_spread = mid_price * (self.min_spread_bps / 10000.0)
        base_half_spread = max(min_half_spread, base_half_spread)

        # 3. Smooth continuous inventory ratio in [-1.0, 1.0] (replaces suicidal mid ± 1.0 jumps)
        inv_ratio = float(np.clip(inventory_contracts / max(self.max_inventory, 1e-6), -1.0, 1.0))
        inv_bias = inv_ratio * base_half_spread * 0.75

        # Asymmetric spreads: long inventory widens bid (less buying), narrows ask (more selling)
        delta_bid = base_half_spread * (1.0 + self.beta_toxic * p_tox_down) + inv_bias
        delta_ask = base_half_spread * (1.0 + self.beta_toxic * p_tox_up) - inv_bias

        # Strict fee hurdle: half-spread must never fall below minimum fee threshold
        delta_bid = max(min_half_spread * 0.8, delta_bid)
        delta_ask = max(min_half_spread * 0.8, delta_ask)

        bid_px = round(r - delta_bid, 1)
        ask_px = round(r + delta_ask, 1)

        # Guarantee true Maker quotes: strictly prevent crossing the book into Taker fills
        bid_px = min(bid_px, round(mid_price - min_half_spread * 0.5, 1))
        ask_px = max(ask_px, round(mid_price + min_half_spread * 0.5, 1))

        allow_bid = True
        allow_ask = True

        # Hard inventory boundary checks
        if inventory_contracts >= self.max_inventory:
            allow_bid = False
        if inventory_contracts <= -self.max_inventory:
            allow_ask = False

        # Toxicity circuit breaker (lowered to 0.65 for strict adverse selection defense)
        if p_tox_down >= 0.65:
            allow_bid = False
        if p_tox_up >= 0.65:
            allow_ask = False

        # ADX Macro Trend Defense: prevent catching falling knives or shorting breakouts
        if adx14 >= 28.0:
            if trend_dir < 0:  # Confirmed strong downtrend: cut bids completely
                allow_bid = False
            elif trend_dir > 0:  # Confirmed strong uptrend: cut asks completely
                allow_ask = False

        # Post-Fill Cooldown: prevent cascade sweeps
        if cooldown_bid:
            allow_bid = False
        if cooldown_ask:
            allow_ask = False

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

    # State tracking for Post-Fill Cooldown & Queue Retention
    last_pos_sz = 0.0
    last_fill_time_bid = 0.0
    last_fill_time_ask = 0.0
    FILL_COOLDOWN_SEC = 5.0

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

            # 4. Fetch current real net inventory from OKX & Detect Fills
            pos_list = client.get_positions(inst_type="SWAP", inst_id="BTC-USDT-SWAP")
            pos_sz = safe_float(pos_list[0].get("pos")) if pos_list else 0.0
            upl = safe_float(pos_list[0].get("upl")) if pos_list else 0.0

            # Trigger post-fill cooldown on sudden position change
            if loop_count > 1:
                if pos_sz > last_pos_sz:
                    last_fill_time_bid = time.time()
                    log_msg(f"⚡ Fill detected on BUY side ({last_pos_sz} -> {pos_sz}), activating {FILL_COOLDOWN_SEC}s cooldown")
                elif pos_sz < last_pos_sz:
                    last_fill_time_ask = time.time()
                    log_msg(f"⚡ Fill detected on SELL side ({last_pos_sz} -> {pos_sz}), activating {FILL_COOLDOWN_SEC}s cooldown")
            last_pos_sz = pos_sz

            cooldown_bid = (time.time() - last_fill_time_bid) < FILL_COOLDOWN_SEC
            cooldown_ask = (time.time() - last_fill_time_ask) < FILL_COOLDOWN_SEC

            # 5. Avellaneda-Stoikov Mathematical Pricing & Optimal Half-Spread Calculation
            as_res = kernel.compute_as_grid_quotes(
                mid_price=mid_price,
                gk_vol=ohlcv_feat["gk_vol"],
                inventory_contracts=pos_sz,
                mu_drift=pred_drift,
                p_tox_down=pred_tox_down,
                p_tox_up=pred_tox_up,
                adx14=ohlcv_feat.get("adx14", 0.0),
                trend_dir=ohlcv_feat.get("trend_dir", 0),
                cooldown_bid=cooldown_bid,
                cooldown_ask=cooldown_ask
            )

            r = as_res["r"]
            bid_px = as_res["bid_px"]
            ask_px = as_res["ask_px"]
            allow_bid = as_res["allow_bid"]
            allow_ask = as_res["allow_ask"]

            # 6. Order Management with FIFO Queue Position Retention
            # Price tolerance: do NOT cancel orders if target price has barely moved
            PRICE_TOLERANCE_USDT = max(1.5, mid_price * 0.0001)

            pending_orders = client.get_pending_orders(inst_type="SWAP", inst_id="BTC-USDT-SWAP")
            active_bid_order = None
            active_ask_order = None
            orders_to_cancel = []

            for p in pending_orders:
                side = p.get("side")
                px = safe_float(p.get("px"))
                ord_id = p.get("ordId")
                if side == "buy":
                    if not allow_bid or abs(px - bid_px) > PRICE_TOLERANCE_USDT:
                        orders_to_cancel.append({"instId": "BTC-USDT-SWAP", "ordId": ord_id})
                    else:
                        active_bid_order = p
                elif side == "sell":
                    if not allow_ask or abs(px - ask_px) > PRICE_TOLERANCE_USDT:
                        orders_to_cancel.append({"instId": "BTC-USDT-SWAP", "ordId": ord_id})
                    else:
                        active_ask_order = p

            if orders_to_cancel:
                try:
                    client.cancel_batch_orders(orders_to_cancel)
                except Exception:
                    pass

            action_desc = []
            now_ms = int(time.time() * 1000)

            # Post Asymmetric Two-Sided Limit Quotes (Post-Only)
            if allow_bid:
                if active_bid_order:
                    action_desc.append(f"BID_RETAINED_QUEUE({safe_float(active_bid_order.get('px'))})")
                else:
                    cl_id_bid = f"asb{now_ms}"[-32:]
                    res_bid = client.place_order(
                        inst_id="BTC-USDT-SWAP",
                        td_mode="cross",
                        side="buy",
                        ord_type="post_only",
                        px=str(bid_px),
                        sz=kernel.lot_sz,
                        cl_ord_id=cl_id_bid
                    )
                    if res_bid.get("code") == "0":
                        action_desc.append(f"BID_POST({bid_px})")
                        ord_data = res_bid.get("data", [{}])[0]
                        action_desc.append(f"BID_ORDER_OPEN({ord_data.get('ordId')})")
                    else:
                        action_desc.append(f"BID_FAIL({res_bid.get('msg')})")
            else:
                reason = "MAX_INV" if pos_sz >= kernel.max_inventory else (
                    "TOXIC_DOWN_PROTECT" if pred_tox_down >= 0.65 else (
                        "ADX_TREND_DOWN" if ohlcv_feat.get("adx14", 0) >= 28 and ohlcv_feat.get("trend_dir", 0) < 0 else "COOLDOWN"
                    )
                )
                action_desc.append(f"BID_CUT({reason})")

            if allow_ask:
                if active_ask_order:
                    action_desc.append(f"ASK_RETAINED_QUEUE({safe_float(active_ask_order.get('px'))})")
                else:
                    cl_id_ask = f"asa{now_ms}"[-32:]
                    res_ask = client.place_order(
                        inst_id="BTC-USDT-SWAP",
                        td_mode="cross",
                        side="sell",
                        ord_type="post_only",
                        px=str(ask_px),
                        sz=kernel.lot_sz,
                        cl_ord_id=cl_id_ask
                    )
                    if res_ask.get("code") == "0":
                        action_desc.append(f"ASK_POST({ask_px})")
                        ord_data = res_ask.get("data", [{}])[0]
                        action_desc.append(f"ASK_ORDER_OPEN({ord_data.get('ordId')})")
                    else:
                        action_desc.append(f"ASK_FAIL({res_ask.get('msg')})")
            else:
                reason = "MIN_INV" if pos_sz <= -kernel.max_inventory else (
                    "TOXIC_UP_PROTECT" if pred_tox_up >= 0.65 else (
                        "ADX_TREND_UP" if ohlcv_feat.get("adx14", 0) >= 28 and ohlcv_feat.get("trend_dir", 0) > 0 else "COOLDOWN"
                    )
                )
                action_desc.append(f"ASK_CUT({reason})")
            # 7. Write telemetry status and recent fills
                        # Query real account fills to count true executed trades
            try:
                bills_res = client._sign_and_request('GET', '/api/v5/account/bills', {'instType': 'SWAP', 'limit': '20'})
                real_bills = bills_res.get('data', [])
                valid_trades = [b for b in real_bills if b.get('subType') in ['1', '2']]
                if valid_trades:
                    kernel.total_matched_trades = len(valid_trades)
            except Exception:
                pass

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
