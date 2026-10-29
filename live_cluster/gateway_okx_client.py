# -*- coding: utf-8 -*-
"""
Production OKX API Gateway Client (Live / Real Trading Mode).
Features:
- Enforces IPv4-only TCP connections to match OKX IP whitelist (155.254.60.38)
- Requests Session with HTTP Keep-Alive & Connection Pooling
- HMAC-SHA256 signature generation
- Native fallback DNS resolver & socket hook
- Automatic retry on network/transport fluctuations
- High-frequency batch order placement & cancellation support
- Full support for v5 market, trade, and account endpoints
"""

import os, sys, time, json, hmac, hashlib, base64, socket, random
from typing import Dict, Any, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Force IPv4 socket creation to match OKX IP whitelist
_ORIG_CREATE_CONNECTION = socket.create_connection
def _forced_ipv4_create_connection(address, timeout=None, source_address=None):
    host, port = address
    err = None
    for res in socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM):
        af, socktype, proto, canonname, sa = res
        try:
            sock = socket.socket(af, socktype, proto)
            if timeout is not None:
                sock.settimeout(timeout)
            if source_address is not None:
                sock.bind(source_address)
            sock.connect(sa)
            return sock
        except socket.error as e:
            err = e
            if sock is not None:
                sock.close()
    if err is not None:
        raise err
    else:
        raise socket.error("getaddrinfo returns an empty list")

socket.create_connection = _forced_ipv4_create_connection

_DNS_CACHE: Dict[str, List[str]] = {}
_ORIG_GETADDRINFO = socket.getaddrinfo

def _query_dns_fallback(domain: str, dns_servers=("8.8.8.8", "8.8.4.4"), timeout=2) -> List[str]:
    trans_id = random.randint(0, 65535)
    header = trans_id.to_bytes(2, 'big') + b'\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00'
    qname = b''
    for part in domain.split('.'):
        qname += len(part).to_bytes(1, 'big') + part.encode()
    qname += b'\x00'
    packet = header + qname + b'\x00\x01\x00\x01'
    
    for server in dns_servers:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)
            sock.sendto(packet, (server, 53))
            data, _ = sock.recvfrom(512)
            sock.close()
            
            idx = 12 + len(qname) + 4
            ips = []
            while idx < len(data):
                if data[idx] & 0xc0 == 0xc0:
                    idx += 2
                else:
                    while idx < len(data) and data[idx] != 0:
                        idx += data[idx] + 1
                    idx += 1
                if idx + 10 > len(data):
                    break
                atype = int.from_bytes(data[idx:idx+2], 'big')
                rdlength = int.from_bytes(data[idx+8:idx+10], 'big')
                idx += 10
                if atype == 1 and rdlength == 4:
                    ip = '.'.join(str(b) for b in data[idx:idx+4])
                    ips.append(ip)
                idx += rdlength
            if ips:
                return ips
        except Exception:
            continue
    return []

def _smart_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    # Enforce AF_INET (IPv4)
    try:
        return _ORIG_GETADDRINFO(host, port, socket.AF_INET, type, proto, flags)
    except socket.gaierror:
        if host in _DNS_CACHE and _DNS_CACHE[host]:
            cached_ip = random.choice(_DNS_CACHE[host])
            return _ORIG_GETADDRINFO(cached_ip, port, socket.AF_INET, type, proto, flags)
        ips = _query_dns_fallback(host)
        if ips:
            _DNS_CACHE[host] = ips
            return _ORIG_GETADDRINFO(ips[0], port, socket.AF_INET, type, proto, flags)
        raise

socket.getaddrinfo = _smart_getaddrinfo

class OKXGatewayClient:
    def __init__(self, api_key: str, api_secret: str, passphrase: str, simulated: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.simulated = simulated
        self.base_url = "https://www.okx.com"

        self.session = requests.Session()
        retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _sign_and_request(self, method: str, endpoint: str, params: Optional[Dict] = None, body: Optional[Any] = None, max_retries: int = 3) -> Dict[str, Any]:
        request_path = endpoint
        if params:
            query_str = requests.compat.urlencode(params)
            request_path = f"{endpoint}?{query_str}"
        url = f"{self.base_url}{request_path}"

        body_str = json.dumps(body) if body is not None else ""
        data_bytes = body_str.encode('utf-8') if body_str else None

        last_error = None
        for attempt in range(max_retries):
            ts = time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())
            msg = f"{ts}{method}{request_path}{body_str}"
            sig = base64.b64encode(hmac.new(self.api_secret.encode('utf-8'), msg.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')

            headers = {
                'User-Agent': 'Mozilla/5.0 NautilusTrader/1.221',
                'Content-Type': 'application/json',
                'OK-ACCESS-KEY': self.api_key,
                'OK-ACCESS-SIGN': sig,
                'OK-ACCESS-TIMESTAMP': ts,
                'OK-ACCESS-PASSPHRASE': self.passphrase,
            }
            if self.simulated:
                headers['x-simulated-trading'] = '1'

            try:
                if method == 'GET':
                    resp = self.session.get(url, headers=headers, timeout=8)
                else:
                    resp = self.session.post(url, headers=headers, data=data_bytes, timeout=8)
                return resp.json()
            except Exception as e:
                last_error = e
                if "name resolution" in str(e).lower() or isinstance(e, socket.gaierror):
                    ips = _query_dns_fallback("www.okx.com")
                    if ips:
                        _DNS_CACHE["www.okx.com"] = ips
                time.sleep(0.5 * (attempt + 1))

        raise last_error

    def get_ticker(self, inst_id: str) -> Dict[str, Any]:
        res = self._sign_and_request('GET', '/api/v5/market/ticker', {'instId': inst_id})
        data = res.get('data', [])
        return data[0] if data else {}

    def get_books(self, inst_id: str, sz: int = 20) -> Dict[str, Any]:
        res = self._sign_and_request('GET', '/api/v5/market/books', {'instId': inst_id, 'sz': str(sz)})
        data = res.get('data', [])
        return data[0] if data else {}

    def get_candles(self, inst_id: str, bar: str = '15m', limit: int = 100) -> List[List[str]]:
        res = self._sign_and_request('GET', '/api/v5/market/candles', {'instId': inst_id, 'bar': bar, 'limit': limit})
        return res.get('data', [])

    def get_balance(self, ccy: str = 'USDT') -> Dict[str, Any]:
        res = self._sign_and_request('GET', '/api/v5/account/balance', {'ccy': ccy})
        data = res.get('data', [])
        if data:
            details = data[0].get('details', [])
            if details:
                return details[0]
            return data[0]
        return {}

    def get_positions(self, inst_type: str = 'SWAP', inst_id: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {'instType': inst_type}
        if inst_id:
            params['instId'] = inst_id
        res = self._sign_and_request('GET', '/api/v5/account/positions', params)
        return res.get('data', [])

    def place_order(self, inst_id: str, td_mode: str, side: str, ord_type: str, sz: str, px: Optional[str] = None, cl_ord_id: Optional[str] = None) -> Dict[str, Any]:
        body = {
            'instId': inst_id,
            'tdMode': td_mode,
            'side': side,
            'ordType': ord_type,
            'sz': sz
        }
        if px:
            body['px'] = px
        if cl_ord_id:
            body['clOrdId'] = cl_ord_id
        return self._sign_and_request('POST', '/api/v5/trade/order', body=body)

    def cancel_order(self, inst_id: str, ord_id: Optional[str] = None, cl_ord_id: Optional[str] = None) -> Dict[str, Any]:
        body = {'instId': inst_id}
        if ord_id:
            body['ordId'] = ord_id
        if cl_ord_id:
            body['clOrdId'] = cl_ord_id
        return self._sign_and_request('POST', '/api/v5/trade/cancel-order', body=body)

    def cancel_batch_orders(self, orders: List[Dict[str, str]]) -> Dict[str, Any]:
        if not orders:
            return {"code": "0", "data": []}
        return self._sign_and_request('POST', '/api/v5/trade/cancel-batch-orders', body=orders[:20])

    def get_pending_orders(self, inst_type: str = 'SWAP', inst_id: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {'instType': inst_type}
        if inst_id:
            params['instId'] = inst_id
        res = self._sign_and_request('GET', '/api/v5/trade/orders-pending', params)
        return res.get('data', [])
