# -*- coding: utf-8 -*-
"""
okx_ws_manager.py - High-Performance OKX WebSocket Manager
- Subscribes to Public: books5, trades for BTC-USDT-SWAP
- Subscribes to Private: orders (real execution fills), account (live equity/margin)
- Computes real-time tick OFI (Order Flow Imbalance) and Micro-Price
- Strictly delivers real execution fills (state == 'filled' / 'partially_filled')
"""

import socket
# Force IPv4 socket resolution globally for OKX requests
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(*args, **kwargs):
    responses = _orig_getaddrinfo(*args, **kwargs)
    ipv4_res = [r for r in responses if r[0] == socket.AF_INET]
    return ipv4_res if ipv4_res else responses
socket.getaddrinfo = _ipv4_getaddrinfo

import asyncio
import time
import json
import hmac
import hashlib
import base64
import threading
from typing import Dict, Any, List, Optional
import websockets

class OKXWebSocketManager:
    def __init__(self, api_key: str, api_secret: str, passphrase: str, inst_id: str = "BTC-USDT-SWAP"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.inst_id = inst_id
        
        self.public_uri = "wss://ws.okx.com:443/ws/v5/public"
        self.private_uri = "wss://ws.okx.com:443/ws/v5/private"
        
        self.running = False
        self.lock = threading.Lock()
        
        # Real-time LOB cache
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.best_bid_sz = 0.0
        self.best_ask_sz = 0.0
        self.mid_price = 0.0
        self.micro_price = 0.0
        self.last_book_time = 0.0
        
        # OFI tick state
        self.prev_b1 = 0.0
        self.prev_v_b1 = 0.0
        self.prev_a1 = 0.0
        self.prev_v_a1 = 0.0
        self.ofi_accum = 0.0
        self.recent_ofi_window = []  # (ts, ofi_val)
        
        # Real fill queue
        self.unprocessed_fills: List[Dict[str, Any]] = []
        self.known_fill_ids = set()
        
        # Account cache
        self.total_eq = 0.0
        self.avail_eq = 0.0
        self.position_sz = 0.0
        
        self.ws_thread = None

    def start(self):
        self.running = True
        self.ws_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.ws_thread.start()

    def stop(self):
        self.running = False

    def _run_event_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(asyncio.gather(
            self._public_ws_loop(),
            self._private_ws_loop()
        ))

    async def _public_ws_loop(self):
        while self.running:
            try:
                async with websockets.connect(self.public_uri, ping_interval=20, ping_timeout=10) as ws:
                    sub_msg = {
                        "op": "subscribe",
                        "args": [
                            {"channel": "books5", "instId": self.inst_id},
                            {"channel": "trades", "instId": self.inst_id}
                        ]
                    }
                    await ws.send(json.dumps(sub_msg))
                    
                    while self.running:
                        msg_str = await ws.recv()
                        msg = json.loads(msg_str)
                        arg = msg.get("arg", {})
                        channel = arg.get("channel")
                        data_list = msg.get("data", [])
                        
                        if channel == "books5" and data_list:
                            self._handle_book(data_list[0])
                        elif channel == "trades" and data_list:
                            self._handle_trades(data_list)
                            
            except Exception as e:
                await asyncio.sleep(2)

    def _handle_book(self, book_data: Dict[str, Any]):
        bids = book_data.get("bids", [])
        asks = book_data.get("asks", [])
        if not bids or not asks:
            return
            
        b1, v_b1 = float(bids[0][0]), float(bids[0][1])
        a1, v_a1 = float(asks[0][0]), float(asks[0][1])
        mid = (b1 + a1) * 0.5
        micro = (b1 * v_a1 + a1 * v_b1) / (v_b1 + v_a1 + 1e-9)
        
        with self.lock:
            # Cont-Kukanov-Stoikov OFI continuous increment
            if self.prev_b1 > 0:
                if b1 > self.prev_b1:
                    delta_b = v_b1
                elif b1 == self.prev_b1:
                    delta_b = v_b1 - self.prev_v_b1
                else:
                    delta_b = -self.prev_v_b1
                    
                if a1 > self.prev_a1:
                    delta_a = -self.prev_v_a1
                elif a1 == self.prev_a1:
                    delta_a = v_a1 - self.prev_v_a1
                else:
                    delta_a = v_a1
                    
                ofi_step = delta_b - delta_a
                self.ofi_accum += ofi_step
                now = time.time()
                self.recent_ofi_window.append((now, ofi_step))
                # keep 60 seconds
                cutoff = now - 60.0
                while self.recent_ofi_window and self.recent_ofi_window[0][0] < cutoff:
                    self.recent_ofi_window.pop(0)
                    
            self.prev_b1 = b1
            self.prev_v_b1 = v_b1
            self.prev_a1 = a1
            self.prev_v_a1 = v_a1
            
            self.best_bid = b1
            self.best_ask = a1
            self.best_bid_sz = v_b1
            self.best_ask_sz = v_a1
            self.mid_price = mid
            self.micro_price = micro
            self.last_book_time = time.time()

    def _handle_trades(self, trades: List[Dict[str, Any]]):
        now = time.time()
        with self.lock:
            for tr in trades:
                sz = float(tr.get("sz", 0))
                side = tr.get("side")
                signed_sz = sz if side == "buy" else -sz
                self.recent_ofi_window.append((now, signed_sz))

    async def _private_ws_loop(self):
        while self.running:
            try:
                async with websockets.connect(self.private_uri, ping_interval=20, ping_timeout=10) as ws:
                    ts = str(int(time.time()))
                    msg = f"{ts}GET/users/self/verify"
                    sig = base64.b64encode(hmac.new(self.api_secret.encode('utf-8'), msg.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')
                    login_msg = {
                        "op": "login",
                        "args": [{"apiKey": self.api_key, "passphrase": self.passphrase, "timestamp": ts, "sign": sig}]
                    }
                    await ws.send(json.dumps(login_msg))
                    login_resp = await ws.recv()
                    login_data = json.loads(login_resp)
                    if login_data.get("code") != "0":
                        await asyncio.sleep(5)
                        continue
                        
                    sub_msg = {
                        "op": "subscribe",
                        "args": [
                            {"channel": "orders", "instType": "SWAP", "instId": self.inst_id},
                            {"channel": "account"}
                        ]
                    }
                    await ws.send(json.dumps(sub_msg))
                    
                    while self.running:
                        msg_str = await ws.recv()
                        msg = json.loads(msg_str)
                        arg = msg.get("arg", {})
                        channel = arg.get("channel")
                        data_list = msg.get("data", [])
                        
                        if channel == "orders" and data_list:
                            self._handle_order_event(data_list)
                        elif channel == "account" and data_list:
                            self._handle_account_event(data_list[0])
                            
            except Exception as e:
                await asyncio.sleep(2)

    def _handle_order_event(self, orders: List[Dict[str, Any]]):
        """Strict Real Fill Verification: Only accept filled or partially_filled states with positive filled size."""
        with self.lock:
            for ord_data in orders:
                state = ord_data.get("state")
                fill_sz = float(ord_data.get("fillSz", 0))
                trade_id = ord_data.get("tradeId") or ord_data.get("ordId")
                
                if state in ["filled", "partially_filled"] and fill_sz > 0:
                    if trade_id not in self.known_fill_ids:
                        self.known_fill_ids.add(trade_id)
                        self.unprocessed_fills.append({
                            "tradeId": trade_id,
                            "ordId": ord_data.get("ordId"),
                            "clOrdId": ord_data.get("clOrdId"),
                            "side": ord_data.get("side"),
                            "fillPx": float(ord_data.get("fillPx", ord_data.get("avgPx", 0))),
                            "fillSz": fill_sz,
                            "fee": float(ord_data.get("fee", 0)),
                            "feeCcy": ord_data.get("feeCcy", "USDT"),
                            "state": state,
                            "timestamp": ord_data.get("uTime") or str(int(time.time() * 1000))
                        })

    def _handle_account_event(self, acc_data: Dict[str, Any]):
        with self.lock:
            self.total_eq = float(acc_data.get("totalEq", 0))
            details = acc_data.get("details", [])
            for d in details:
                if d.get("ccy") == "USDT":
                    self.avail_eq = float(d.get("availEq", 0))

    def get_latest_book(self) -> Dict[str, float]:
        with self.lock:
            return {
                "bid": self.best_bid,
                "ask": self.best_ask,
                "bid_sz": self.best_bid_sz,
                "ask_sz": self.best_ask_sz,
                "mid": self.mid_price,
                "micro": self.micro_price,
                "age": time.time() - self.last_book_time
            }

    def get_rolling_ofi(self) -> float:
        with self.lock:
            now = time.time()
            cutoff = now - 60.0
            recent = [v for ts, v in self.recent_ofi_window if ts >= cutoff]
            return float(sum(recent)) if recent else 0.0

    def pop_new_fills(self) -> List[Dict[str, Any]]:
        with self.lock:
            fills = list(self.unprocessed_fills)
            self.unprocessed_fills.clear()
            return fills
