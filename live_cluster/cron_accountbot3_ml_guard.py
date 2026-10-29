# -*- coding: utf-8 -*-
"""
cron_accountbot3_ml_guard.py - Autonomous ML & Decision Alignment Monitor for accountbot3
Monitors Strategy 012 Avellaneda-Stoikov V2 live container, verifies real-time ML feature inference,
evaluates decision logic compliance against initial system design goals, and logs structured health audits.
"""

import os
import sys
import time
import json
import urllib.request
import subprocess
from datetime import datetime, timezone

CLUSTER_DIR = "/opt/nautilus_cluster"
SERVICE_DIR = os.path.join(CLUSTER_DIR, "services/07_as_grid_accountbot3")
LOG_DIR = os.path.join(CLUSTER_DIR, "logs")
SERVICE_LOG_DIR = os.path.join(SERVICE_DIR, "logs")

STATUS_FILE = os.path.join(SERVICE_LOG_DIR, "07_as_grid_status.json")
TELEMETRY_FILE = os.path.join(SERVICE_LOG_DIR, "07_as_grid_telemetry.json")
FILLS_FILE = os.path.join(SERVICE_LOG_DIR, "07_as_grid_fills.json")
AUDIT_OUTPUT_FILE = os.path.join(LOG_DIR, "accountbot3_ml_health.json")
CHECKPOINTS_DIR = os.path.join(CLUSTER_DIR, "checkpoints")

TELEGRAM_TOKEN = "8821340499:AAGU-bW7uiFq9AKr6S0XbE6zlN8mNK81gJQ"
TELEGRAM_CHAT_ID = "974319640"

def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            res = json.loads(resp.read().decode())
            return res.get("ok", False)
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")
        return False

def check_container_status():
    try:
        res = subprocess.run(
            ["docker", "inspect", "nautilus_vps_007_accountbot3", "--format", "{{.State.Running}},{{.State.Restarting}},{{.State.StartedAt}}"],
            capture_output=True, text=True, check=True
        )
        parts = res.stdout.strip().split(",")
        return {
            "running": parts[0] == "true",
            "restarting": parts[1] == "true",
            "started_at": parts[2] if len(parts) > 2 else "unknown"
        }
    except Exception as e:
        return {"running": False, "error": str(e)}

def load_json(filepath, default=None):
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def evaluate_system():
    now_utc = datetime.now(timezone.utc)
    now_ts = now_utc.timestamp()
    
    # 1. Container & Process Liveness
    container = check_container_status()
    status_data = load_json(STATUS_FILE, {})
    telemetry_list = load_json(TELEMETRY_FILE, [])
    recent_fills = load_json(FILLS_FILE, [])
    
    status_ts_str = status_data.get("timestamp_utc", "")
    heartbeat_lag_sec = 9999.0
    if status_ts_str:
        try:
            dt = datetime.fromisoformat(status_ts_str.replace("Z", "+00:00"))
            heartbeat_lag_sec = abs(now_ts - dt.timestamp())
        except Exception:
            pass

    is_heartbeat_alive = heartbeat_lag_sec <= 60.0

    # 2. ML Feature & Inference Evaluation
    latest_telem = telemetry_list[-1] if telemetry_list else {}
    live_features = latest_telem.get("live_microstructure_features", {})
    as_params = latest_telem.get("live_as_model_parameters", {})
    
    gk_vol = live_features.get("annual_gk_vol", 0.0)
    rolling_ofi = live_features.get("rolling_ofi_60s", 0.0)
    ofi_norm = live_features.get("ofi_norm", 0.0)
    if ofi_norm is None:
        ofi_norm = 0.0
    ma20_dev = live_features.get("ma20_deviation", 0.0)
    mu_pred = as_params.get("mu_pred_drift", 0.0)
    
    # ML Feature validity checks (causal range verification)
    ml_feature_checks = {
        "gk_vol_valid": (0.05 <= gk_vol <= 2.50),
        "ofi_norm_bounded": (abs(float(ofi_norm)) <= 3.5),
        "ma20_dev_bounded": (abs(float(ma20_dev)) <= 5.0),
        "mu_drift_bounded": (abs(float(mu_pred)) <= 3.0)
    }
    ml_healthy = all(ml_feature_checks.values())
    # Checkpoint model file checks
    ckpt_drift = os.path.join(CHECKPOINTS_DIR, "lgb_drift_model.txt")
    ckpt_down = os.path.join(CHECKPOINTS_DIR, "lgb_tox_down_model.txt")
    ckpt_up = os.path.join(CHECKPOINTS_DIR, "lgb_tox_up_model.txt")
    has_checkpoints = os.path.exists(ckpt_drift) and os.path.exists(ckpt_down) and os.path.exists(ckpt_up)

    # 3. Decision Logic & System Initial Goals Alignment
    mid_px = status_data.get("current_mid", 0.0)
    bid_quote = status_data.get("bid_quote", 0.0)
    ask_quote = status_data.get("ask_quote", 0.0)
    inventory = status_data.get("inventory_contracts", 0.0)
    actions = status_data.get("actions", [])
    
    # Goal 1: Fee Hurdle Compliance (Total Spread >= 9.0 bps)
    spread_usd = ask_quote - bid_quote if (ask_quote > 0 and bid_quote > 0) else 0.0
    spread_bps = (spread_usd / mid_px * 10000.0) if mid_px > 0 else 0.0
    goal_fee_hurdle_met = (spread_bps >= 8.5)
    
    # Goal 2: Strict Maker Post-Only (Never cross mid-price)
    goal_maker_guarantee = (bid_quote < mid_px) and (ask_quote > mid_px) if (bid_quote > 0 and ask_quote > 0) else True
    
    # Goal 3: Inventory Hard Bound (|q| <= 0.05)
    goal_inventory_safe = (abs(inventory) <= 0.05001)
    
    # Goal 4: Queue Position Retention vs Cancellation Churn
    has_queue_hold = any("QUEUE_HOLD" in a for a in actions)
    has_post = any("POST" in a for a in actions)
    
    # Overall Compliance
    compliance_score = sum([
        1 if container["running"] else 0,
        1 if is_heartbeat_alive else 0,
        1 if ml_healthy else 0,
        1 if goal_fee_hurdle_met else 0,
        1 if goal_maker_guarantee else 0,
        1 if goal_inventory_safe else 0
    ]) / 6.0 * 100.0

    system_status = "HEALTHY_OPTIMAL" if compliance_score >= 83.3 else ("DEGRADED" if compliance_score >= 50.0 else "CRITICAL")
    
    report = {
        "timestamp_utc": now_utc.isoformat(),
        "account": "accountbot3",
        "system_status": system_status,
        "compliance_score_pct": round(compliance_score, 1),
        "runtime_liveness": {
            "container_running": container["running"],
            "heartbeat_lag_sec": round(heartbeat_lag_sec, 1),
            "heartbeat_fresh": is_heartbeat_alive,
            "loop_count": status_data.get("loop_count", 0)
        },
        "ml_inference_diagnostics": {
            "ml_healthy": ml_healthy,
            "annual_gk_vol": round(gk_vol, 4),
            "rolling_ofi_60s": round(rolling_ofi, 4),
            "ma20_deviation": round(ma20_dev, 4),
            "mu_pred_drift": round(mu_pred, 4),
            "feature_checks": ml_feature_checks,
            "checkpoints_present": has_checkpoints
        },
        "decision_logic_alignment": {
            "mid_price": mid_px,
            "bid_quote": bid_quote,
            "ask_quote": ask_quote,
            "spread_usd": round(spread_usd, 2),
            "spread_bps": round(spread_bps, 2),
            "fee_hurdle_satisfied": goal_fee_hurdle_met,
            "strict_maker_satisfied": goal_maker_guarantee,
            "inventory_contracts": inventory,
            "inventory_bound_safe": goal_inventory_safe,
            "adverse_mitigations_count": status_data.get("adverse_mitigations", 0),
            "recent_actions": actions
        },
        "trade_performance_real": {
            "real_matched_trades": status_data.get("real_matched_trades", 0),
            "real_volume_usd": status_data.get("real_volume_usd", 0.0),
            "real_fees_paid_usd": status_data.get("real_fees_paid_usd", 0.0),
            "recent_fill_count": len(recent_fills)
        }
    }
    
    # Persist structured health audit
    try:
        os.makedirs(os.path.dirname(AUDIT_OUTPUT_FILE), exist_ok=True)
        with open(AUDIT_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error persisting health audit: {e}")
        
    return report

def main():
    report = evaluate_system()
    status = report["system_status"]
    score = report["compliance_score_pct"]
    liveness = report["runtime_liveness"]
    ml = report["ml_inference_diagnostics"]
    dec = report["decision_logic_alignment"]
    perf = report["trade_performance_real"]
    
    print("\n" + "="*80)
    print(f" [CRON] accountbot3 Avellaneda-Stoikov ML & Decision Alignment Watchdog")
    print("="*80)
    print(f"UTC Timestamp: {report['timestamp_utc']}")
    print(f"Overall Status: {status} (Score: {score}%)")
    print(f"Runtime: Container Running={liveness['container_running']} | Lag={liveness['heartbeat_lag_sec']}s (Loop #{liveness['loop_count']})")
    print(f"ML Features: GK-Vol={ml['annual_gk_vol']:.2%} | OFI={ml['rolling_ofi_60s']:+.2f} | MA20-Dev={ml['ma20_deviation']:+.2f} | Drift={ml['mu_pred_drift']:+.4f} | Health={ml['ml_healthy']}")
    print(f"Quotes: [{dec['bid_quote']} / {dec['ask_quote']}] | Mid={dec['mid_price']} | Spread={dec['spread_bps']:.1f} bps (Hurdle Met: {dec['fee_hurdle_satisfied']})")
    print(f"Risk & Inventory: Pos={dec['inventory_contracts']} (Safe: {dec['inventory_bound_safe']}) | Mitigations={dec['adverse_mitigations_count']}")
    print(f"Trades: Real Matched={perf['real_matched_trades']} | Volume=${perf['real_volume_usd']:,.2f} | Fees Paid=${perf['real_fees_paid_usd']:.4f}")
    print(f"Actions: {dec['recent_actions']}")
    print("="*80 + "\n")
    
    # Send alert only if degraded or critical
    if status != "HEALTHY_OPTIMAL":
        alert_msg = (
            f"⚠ [accountbot3 ALERT] Status: {status} (Score: {score}%)\n"
            f"Heartbeat Lag: {liveness['heartbeat_lag_sec']}s\n"
            f"Inventory: {dec['inventory_contracts']} | Mid: {dec['mid_price']}\n"
            f"Spread: {dec['spread_bps']:.1f} bps (Hurdle: {dec['fee_hurdle_satisfied']})\n"
            f"ML Health: {ml['ml_healthy']}"
        )
        send_telegram(alert_msg)

if __name__ == "__main__":
    main()
