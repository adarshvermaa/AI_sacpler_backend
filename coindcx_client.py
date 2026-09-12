"""
AlphaScalper - CoinDCX High-Performance API Client
Implements HMAC-SHA256 authenticated REST calls for Futures, Margin, and Spot endpoints.
Supports both LIVE exchange execution and PAPER simulation mode.
"""

import time
import hmac
import hashlib
import json
import logging
from typing import Dict, Any, Optional, List
import httpx

from config import settings

logger = logging.getLogger("AlphaScalper.CoinDCXClient")


class CoinDCXClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        public_url: Optional[str] = None,
        is_paper: Optional[bool] = None
    ):
        self.api_key = api_key or settings.COINDCX_API_KEY
        self.api_secret = api_secret or settings.COINDCX_API_SECRET
        self.base_url = (base_url or settings.COINDCX_REST_BASE_URL).rstrip("/")
        self.public_url = (public_url or settings.COINDCX_PUBLIC_REST_URL).rstrip("/")
        self.is_paper = settings.TRADING_MODE.upper() == "PAPER" if is_paper is None else is_paper
        
        # Pre-compile secret bytes for ultra-fast HMAC signing
        self._secret_bytes = bytes(self.api_secret, encoding="utf-8")
        
        # HTTP client with persistent connection pooling (lazy initialized)
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # In-memory mock state for Paper Trading
        self._paper_orders: Dict[str, Dict[str, Any]] = {}
        self._paper_positions: Dict[str, Dict[str, Any]] = {}
        self._paper_balance = {
            "USDT": settings.INITIAL_SIMULATION_BALANCE,
            "INR": settings.INITIAL_SIMULATION_BALANCE * 89.0
        }
        self._paper_order_counter = 100000
        
        # Diagnostic & Connectivity Verification Cache
        self._last_verification_result: Optional[Dict[str, Any]] = None
        self._last_verification_time: float = 0.0
        self._instrument_details_cache: Dict[str, Dict[str, Any]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Returns or instantiates active httpx AsyncClient tied to current loop."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=10.0,
                limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)
            )
        return self._http_client

    async def _safe_get(self, url: str, **kwargs) -> httpx.Response:
        """Resilient GET request with automatic retry if loop or transport closed."""
        client = await self._get_client()
        try:
            return await client.get(url, **kwargs)
        except (httpx.TransportError, RuntimeError):
            try:
                if not client.is_closed:
                    await client.aclose()
            except Exception:
                pass
            self._http_client = httpx.AsyncClient(
                timeout=10.0,
                limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)
            )
            return await self._http_client.get(url, **kwargs)

    async def _safe_post(self, url: str, **kwargs) -> httpx.Response:
        """Resilient POST request with automatic retry if loop or transport closed."""
        client = await self._get_client()
        try:
            return await client.post(url, **kwargs)
        except (httpx.TransportError, RuntimeError):
            try:
                if not client.is_closed:
                    await client.aclose()
            except Exception:
                pass
            self._http_client = httpx.AsyncClient(
                timeout=10.0,
                limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)
            )
            return await self._http_client.post(url, **kwargs)

    def _generate_signature(self, payload_dict: Dict[str, Any]) -> tuple[str, str]:
        """
        Creates HMAC-SHA256 signature using compact JSON serialization (separators=(',', ':')).
        Returns (json_body, signature_hex).
        """
        json_body = json.dumps(payload_dict, separators=(',', ':'))
        signature = hmac.new(
            self._secret_bytes,
            json_body.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return json_body, signature

    def _get_auth_headers(self, signature: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-AUTH-APIKEY": self.api_key,
            "X-AUTH-SIGNATURE": signature
        }

    async def get_account_balances(self) -> Dict[str, float]:
        """
        Retrieves user balances across USDT, INR, and other assets from CoinDCX.
        """
        if self.is_paper:
            return self._paper_balance.copy()
            
        timestamp = int(round(time.time() * 1000))
        body = {"timestamp": timestamp}
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/users/balances"
        headers = self._get_auth_headers(signature)
        
        try:
            resp = await self._safe_post(url, data=json_body, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                balances: Dict[str, float] = {}
                for item in data:
                    curr = item.get("currency")
                    bal = float(item.get("balance", 0.0))
                    if curr:
                        balances[curr] = bal
                return balances
            else:
                logger.error(f"Failed to fetch balances: {resp.status_code} {resp.text}")
                return {}
        except Exception as e:
            logger.error(f"Error fetching account balances: {e}")
            return {}

    async def get_usable_balance_usdt(self) -> float:
        """
        Calculates total usable USDT margin balance (USDT + INR converted).
        """
        balances = await self.get_account_balances()
        usdt = float(balances.get("USDT", 0.0))
        inr = float(balances.get("INR", 0.0))
        return round(usdt + (inr / 89.0), 4)

    async def verify_api_connectivity(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Comprehensive diagnostic verification:
        1. Public REST market data connectivity (active contracts)
        2. Authenticated API Key & Secret signature validity
        3. User Account Balance (USDT, INR, Crypto Dust)
        4. Real Open Futures Positions (active_pos != 0)
        5. Balance-Aware Execution Strategy Determination
        """
        # Return cached verification if within 10 seconds unless forced
        now = time.time()
        if not force_refresh and self._last_verification_result is not None and (now - self._last_verification_time < 10.0):
            return self._last_verification_result

        result: Dict[str, Any] = {
            "public_api": "FAILED",
            "auth_api": "FAILED",
            "trading_mode": "LIVE" if not self.is_paper else "PAPER",
            "balances": {},
            "total_usdt_balance": 0.0,
            "min_required_usdt": 6.0,
            "is_balance_sufficient": False,
            "active_positions_count": 0,
            "strategy_status": "BALANCE_GUARD_ACTIVE",
            "strategy_message": "",
            "error": None,
            "timestamp": int(now * 1000)
        }
        
        # Test 1: Public endpoint (market data)
        try:
            active_inst = await self.get_active_futures_instruments()
            if active_inst and len(active_inst) > 0:
                result["public_api"] = "CONNECTED"
                result["active_instruments_count"] = len(active_inst)
            else:
                result["public_api"] = "WARNING (Empty active instruments)"
        except Exception as e:
            result["public_api"] = f"ERROR: {str(e)}"
            
        # Test 2: Authenticated endpoint & account balances
        try:
            balances = await self.get_account_balances()
            if balances is not None and len(balances) > 0:
                result["auth_api"] = "AUTHENTICATED"
                result["balances"] = {k: round(v, 6) for k, v in balances.items() if v > 0}
                usdt_bal = float(balances.get("USDT", 0.0))
                inr_bal = float(balances.get("INR", 0.0))
                total_usdt = round(usdt_bal + (inr_bal / 89.0), 4)
                result["total_usdt_balance"] = total_usdt
                
                # Check CoinDCX minimum notional margin sufficiency
                if self.is_paper:
                    result["is_balance_sufficient"] = True
                    result["strategy_status"] = "SIMULATED_EXECUTION"
                    result["strategy_message"] = f"Paper trading active with ${total_usdt:.2f} simulated capital."
                else:
                    if total_usdt >= result["min_required_usdt"]:
                        result["is_balance_sufficient"] = True
                        result["strategy_status"] = "LIVE_EXECUTION_READY"
                        result["strategy_message"] = f"Account verified with ${total_usdt:.2f} USDT. Live order sizing enabled."
                    else:
                        result["is_balance_sufficient"] = False
                        result["strategy_status"] = "BALANCE_GUARD_ACTIVE"
                        result["strategy_message"] = (
                            f"Account verified! Current balance is ${total_usdt:.4f} USDT. "
                            f"CoinDCX requires minimum $6.00 USDT per futures order. "
                            f"Market screener & AI indicators are actively analyzing live data; "
                            f"live execution orders are safely held by Balance Guard until wallet is funded."
                        )
            else:
                result["auth_api"] = "FAILED (Invalid credentials or empty)"
                result["strategy_status"] = "AUTH_FAILED"
                result["strategy_message"] = "Unable to authenticate with CoinDCX API credentials."
        except Exception as e:
            result["auth_api"] = f"ERROR: {str(e)}"
            result["strategy_status"] = "AUTH_ERROR"
            result["strategy_message"] = f"Authentication error: {str(e)}"
            
        # Test 3: Futures positions check (filtering only non-zero active positions)
        try:
            pos = await self.get_futures_positions()
            if isinstance(pos, list):
                open_pos = [p for p in pos if abs(float(p.get("active_pos", 0.0))) > 1e-6]
                result["active_positions_count"] = len(open_pos)
                result["open_positions"] = open_pos
            else:
                result["active_positions_count"] = 0
        except Exception as e:
            logger.debug(f"Positions check error: {e}")

        # Cache result
        self._last_verification_result = result
        self._last_verification_time = now
        return result

    # ==========================================
    # PUBLIC MARKET DATA ENDPOINTS
    # ==========================================

    async def get_active_futures_instruments(self, margin_currency: str = "USDT") -> List[str]:
        """Fetch list of all active Perpetual Futures instruments from CoinDCX."""
        url = f"{self.base_url}/exchange/v1/derivatives/futures/data/active_instruments?margin_currency_short_name[]={margin_currency}"
        try:
            resp = await self._safe_get(url)
            if resp.status_code == 200:
                return resp.json()
            logger.error(f"Failed to fetch active instruments: {resp.status_code} {resp.text}")
            return []
        except Exception as e:
            logger.error(f"Error fetching active futures instruments: {e}")
            return []

    async def get_futures_instrument_details(self, pair: str, margin_currency: str = "USDT") -> Optional[Dict[str, Any]]:
        """Fetch detailed contract specifications (tick size, min qty, max leverage)."""
        url = f"{self.base_url}/exchange/v1/derivatives/futures/data/instrument?pair={pair}&margin_currency_short_name={margin_currency}"
        try:
            resp = await self._safe_get(url)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("instrument")
            return None
        except Exception as e:
            logger.error(f"Error fetching instrument details for {pair}: {e}")
            return None

    async def get_cached_instrument_details(self, pair: str) -> Optional[Dict[str, Any]]:
        """Returns cached instrument specifications to minimize latency."""
        if pair not in self._instrument_details_cache:
            details = await self.get_futures_instrument_details(pair)
            if details:
                self._instrument_details_cache[pair] = details
        return self._instrument_details_cache.get(pair)

    async def sanitize_order_params(
        self,
        pair: str,
        target_notional_usdt: float,
        price: float,
        requested_leverage: float
    ) -> tuple[float, int, float]:
        """
        Guarantees order parameters strictly comply with CoinDCX rules:
        1. Leverage <= max permitted leverage for instrument
        2. Quantity >= min_quantity and rounded to exact quantity_increment
        3. Total notional >= min_notional required by CoinDCX
        Returns: (safe_quantity, safe_leverage, actual_notional)
        """
        import math
        details = await self.get_cached_instrument_details(pair)
        max_lev = float(details.get("max_leverage_long", 10.0)) if details else 10.0
        safe_leverage = int(min(requested_leverage, max_lev))
        
        min_notional = float(details.get("min_notional", 6.0)) if details else 6.0
        min_qty = float(details.get("min_quantity", 0.001)) if details else 0.001
        qty_inc = float(details.get("quantity_increment", 0.001)) if details else 0.001
        
        # Buffer notional slightly above min_notional
        effective_notional = max(target_notional_usdt, min_notional * 1.02)
        raw_qty = effective_notional / max(price, 1e-6)
        
        # Ceil to valid step increment
        steps = math.ceil(round(raw_qty / qty_inc, 6))
        safe_qty = round(steps * qty_inc, 6)
        safe_qty = max(safe_qty, min_qty)
        
        actual_notional = round(safe_qty * price, 4)
        return safe_qty, safe_leverage, actual_notional

    async def get_realtime_futures_prices(self) -> Dict[str, Any]:
        """Fetch current realtime prices, 24h high/low, and volume across futures."""
        url = f"{self.public_url}/market_data/v3/current_prices/futures/rt"
        try:
            resp = await self._safe_get(url)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception as e:
            logger.error(f"Error fetching RT futures prices: {e}")
            return {}

    async def get_futures_orderbook(self, pair: str, depth: int = 20) -> Dict[str, Any]:
        """Fetch live L2 Order Book depth from CoinDCX."""
        url = f"{self.public_url}/market_data/orderbook?pair={pair}"
        try:
            resp = await self._safe_get(url)
            if resp.status_code == 200:
                data = resp.json()
                bids = data.get("bids", {})
                asks = data.get("asks", {})
                # Sort and slice to requested depth
                sorted_bids = dict(sorted([(p, q) for p, q in bids.items()], key=lambda x: float(x[0]), reverse=True)[:depth])
                sorted_asks = dict(sorted([(p, q) for p, q in asks.items()], key=lambda x: float(x[0]))[:depth])
                return {"bids": sorted_bids, "asks": sorted_asks, "timestamp": data.get("timestamp")}
            return {}
        except Exception as e:
            logger.error(f"Error fetching orderbook for {pair}: {e}")
            return {}

    async def get_futures_candlesticks(
        self,
        pair: str,
        resolution: str = "1",
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical and recent candlestick bars.
        resolution: '1' (1m), '5' (5m), '60' (1h), '1D' (1d).
        """
        now = int(time.time())
        from_time = from_ts or (now - 3600 * 4)  # Last 4 hours default
        to_time = to_ts or now
        
        url = f"{self.public_url}/market_data/candlesticks"
        params = {
            "pair": pair,
            "from": from_time,
            "to": to_time,
            "resolution": resolution,
            "pcode": "f"
        }
        try:
            resp = await self._safe_get(url, params=params)
            if resp.status_code == 200:
                res_data = resp.json()
                return res_data.get("data", [])
            return []
        except Exception as e:
            logger.error(f"Error fetching candles for {pair}: {e}")
            return []

    async def get_spot_markets_details(self) -> List[Dict[str, Any]]:
        """Fetch details for all spot markets."""
        url = f"{self.base_url}/exchange/v1/markets_details"
        try:
            resp = await self._safe_get(url)
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception as e:
            logger.error(f"Error fetching spot market details: {e}")
            return []

    # ==========================================
    # AUTHENTICATED FUTURES TRADING ENDPOINTS
    # ==========================================

    async def create_futures_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        total_quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        leverage: Optional[float] = None,
        position_margin_type: str = "isolated",
        margin_currency: str = "USDT",
        time_in_force: Optional[str] = "good_till_cancel"
    ) -> Dict[str, Any]:
        """
        Place a new Futures order (market, limit, stop_market, take_profit_market).
        """
        if self.is_paper:
            # Simulated Execution for Paper Trading
            self._paper_order_counter += 1
            order_id = f"SIM-{self._paper_order_counter}"
            fill_price = price or 100.0  # Will be mapped to latest mark price
            
            sim_order = {
                "id": order_id,
                "pair": pair,
                "side": side.lower(),
                "order_type": order_type,
                "status": "filled" if order_type == "market_order" else "open",
                "price": fill_price,
                "avg_price": fill_price,
                "total_quantity": total_quantity,
                "remaining_quantity": 0.0 if order_type == "market_order" else total_quantity,
                "leverage": leverage or settings.DEFAULT_LEVERAGE,
                "position_margin_type": position_margin_type,
                "margin_currency_short_name": margin_currency,
                "created_at": int(time.time() * 1000)
            }
            self._paper_orders[order_id] = sim_order
            
            # Update Simulated Position
            pos = self._paper_positions.get(pair, {
                "id": f"POS-{pair}",
                "pair": pair,
                "active_pos": 0.0,
                "avg_price": 0.0,
                "leverage": leverage or settings.DEFAULT_LEVERAGE,
                "margin_type": position_margin_type,
                "liquidation_price": 0.0,
                "take_profit_trigger": None,
                "stop_loss_trigger": None
            })
            
            delta_qty = total_quantity if side.lower() == "buy" else -total_quantity
            new_pos_qty = pos["active_pos"] + delta_qty
            pos["active_pos"] = round(new_pos_qty, 4)
            pos["avg_price"] = fill_price
            self._paper_positions[pair] = pos
            
            logger.info(f"[PAPER] Order Created: {pair} {side} {total_quantity} @ {fill_price} (ID: {order_id})")
            return sim_order

        # LIVE API EXECUTION
        timestamp = int(round(time.time() * 1000))
        order_payload: Dict[str, Any] = {
            "side": side.lower(),
            "pair": pair,
            "order_type": order_type,
            "total_quantity": total_quantity,
            "notification": "no_notification",
            "position_margin_type": position_margin_type,
            "margin_currency_short_name": margin_currency
        }
        
        if order_type != "market_order":
            if price is not None:
                order_payload["price"] = str(price)
            order_payload["time_in_force"] = time_in_force or "good_till_cancel"
            if stop_price is not None:
                order_payload["stop_price"] = str(stop_price)
        # Market orders strictly exclude price, time_in_force, and stop_price in CoinDCX API
            
        if leverage is not None:
            order_payload["leverage"] = int(leverage)
            
        body = {
            "timestamp": timestamp,
            "order": order_payload
        }
        
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/derivatives/futures/orders/create"
        headers = self._get_auth_headers(signature)
        
        resp = await self._safe_post(url, data=json_body, headers=headers)
        if resp.status_code != 200:
            logger.error(f"CoinDCX create order failed [{resp.status_code}]: {resp.text}")
        else:
            logger.info(f"CoinDCX order placed successfully: {resp.text}")
            
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text}

    async def get_position_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Finds open position for a given pair symbol on CoinDCX."""
        positions = await self.get_futures_positions()
        for p in positions:
            if p.get("pair") == symbol and abs(float(p.get("active_pos", 0.0))) > 1e-6:
                return p
        return None

    async def cancel_futures_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open futures order on CoinDCX."""
        if self.is_paper:
            if order_id in self._paper_orders:
                self._paper_orders[order_id]["status"] = "cancelled"
                return {"message": "success", "status": 200, "code": 200}
            return {"message": "order not found", "status": 404}

        timestamp = int(round(time.time() * 1000))
        body = {"timestamp": timestamp, "id": order_id}
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/derivatives/futures/orders/cancel"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        return {"status_code": resp.status_code, "text": resp.text}

    async def create_futures_tpsl(
        self,
        position_id: str,
        tp_stop_price: float,
        sl_stop_price: float
    ) -> Dict[str, Any]:
        """
        Create Take Profit and Stop Loss triggers directly on an open position.
        Accepts either exact position UUID or symbol (e.g. 'B-BTC_USDT').
        """
        if self.is_paper:
            # Paper mode: update position TP/SL
            for pair, pos in self._paper_positions.items():
                if pos.get("id") == position_id or pair == position_id:
                    pos["take_profit_trigger"] = tp_stop_price
                    pos["stop_loss_trigger"] = sl_stop_price
                    return {
                        "status": "success",
                        "take_profit": {"stop_price": tp_stop_price, "order_type": "take_profit_market"},
                        "stop_loss": {"stop_price": sl_stop_price, "order_type": "stop_market"}
                    }
            return {"status": "success"}

        actual_id = position_id
        if position_id.startswith("B-"):
            pos = await self.get_position_by_symbol(position_id)
            if pos and pos.get("id"):
                actual_id = pos["id"]
            else:
                logger.warning(f"No active open position found on exchange for {position_id} to attach TP/SL")
                return {"status": "no_active_position"}

        timestamp = int(round(time.time() * 1000))
        body = {
            "timestamp": timestamp,
            "id": actual_id,
            "take_profit": {
                "stop_price": str(tp_stop_price),
                "order_type": "take_profit_market"
            },
            "stop_loss": {
                "stop_price": str(sl_stop_price),
                "order_type": "stop_market"
            }
        }
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/derivatives/futures/positions/create_tpsl"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        return resp.json()

    async def exit_futures_position(self, position_id: str) -> Dict[str, Any]:
        """
        Instantly exit an active futures position at market.
        Accepts either exact position UUID or symbol (e.g. 'B-BTC_USDT').
        """
        if self.is_paper:
            for pair, pos in list(self._paper_positions.items()):
                if pos.get("id") == position_id or pair == position_id:
                    pos["active_pos"] = 0.0
                    pos["take_profit_trigger"] = None
                    pos["stop_loss_trigger"] = None
                    logger.info(f"[PAPER] Position exited: {pair}")
                    return {"message": "success", "status": 200, "code": 200}
            return {"message": "position not found", "status": 404}

        actual_id = position_id
        if position_id.startswith("B-"):
            pos = await self.get_position_by_symbol(position_id)
            if pos and pos.get("id"):
                actual_id = pos["id"]
            else:
                logger.warning(f"No active open position found on exchange for {position_id} to exit")
                return {"status": "no_active_position"}

        timestamp = int(round(time.time() * 1000))
        body = {"timestamp": timestamp, "id": actual_id}
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/derivatives/futures/positions/exit"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        return resp.json()

    async def get_futures_positions(
        self,
        margin_currency: str = "USDT"
    ) -> List[Dict[str, Any]]:
        """Fetch all currently open derivatives/futures positions."""
        if self.is_paper:
            return [p for p in self._paper_positions.values() if abs(p.get("active_pos", 0)) > 0]

        timestamp = int(round(time.time() * 1000))
        body = {
            "timestamp": timestamp,
            "page": "1",
            "size": "50",
            "margin_currency_short_name": [margin_currency]
        }
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/derivatives/futures/positions"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        return []

    async def update_position_leverage(
        self,
        pair: str,
        leverage: float,
        margin_currency: str = "USDT"
    ) -> Dict[str, Any]:
        """Update leverage on an instrument before or during trading."""
        if self.is_paper:
            if pair in self._paper_positions:
                self._paper_positions[pair]["leverage"] = leverage
            return {"message": "success", "status": 200}

        timestamp = int(round(time.time() * 1000))
        body = {
            "timestamp": timestamp,
            "leverage": str(int(leverage)),
            "pair": pair,
            "margin_currency_short_name": margin_currency
        }
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/derivatives/futures/positions/update_leverage"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        return resp.json()

    async def close(self):
        """Close the underlying HTTP connection pool."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
