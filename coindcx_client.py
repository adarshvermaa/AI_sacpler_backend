"""
AlphaScalper - CoinDCX High-Performance API Client
Implements HMAC-SHA256 authenticated REST calls for Futures, Margin, and Spot endpoints.
Supports both LIVE exchange execution and PAPER simulation mode.
"""

import time
import asyncio
import hmac
import hashlib
import json
import logging
from typing import Dict, Any, Optional, List, Union
import httpx

import math
from config import settings

logger = logging.getLogger("AlphaScalper.CoinDCXClient")


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts any value to float, handling None, NaN, inf, invalid strings."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    """Safely converts any value to int, handling None, floats, strings."""
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


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
                    bal = safe_float(item.get("balance"), 0.0)
                    if curr:
                        balances[curr] = bal
                return balances
            else:
                logger.error(f"Failed to fetch balances: {resp.status_code} {resp.text}")
                return {}
        except Exception as e:
            logger.error(f"Error fetching account balances: {e}")
            return {}

    async def get_futures_wallets(self) -> List[Dict[str, Any]]:
        """
        Retrieves user derivative futures wallets from CoinDCX.
        Returns wallet balances including INR and USDT futures collateral.
        """
        if self.is_paper:
            return [
                {"currency_short_name": "INR", "balance": str(self._paper_balance.get("INR", 861.0)), "locked_balance": "0.0"},
                {"currency_short_name": "USDT", "balance": str(self._paper_balance.get("USDT", 10.0)), "locked_balance": "0.0"}
            ]

        client = await self._get_client()
        empty_sig = hmac.new(self._secret_bytes, b'', hashlib.sha256).hexdigest()
        headers = {
            "X-AUTH-APIKEY": self.api_key,
            "X-AUTH-SIGNATURE": empty_sig
        }
        url = f"{self.base_url}/exchange/v1/derivatives/futures/wallets"
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"Failed to fetch futures wallets [{resp.status_code}]: {resp.text}")
                return []
        except Exception as e:
            logger.error(f"Error fetching futures wallets: {e}")
            return []

    async def get_futures_inr_balance(self, inr_usd_rate: float = 87.5) -> Dict[str, float]:
        """
        Retrieves exact Futures INR wallet balance, locked margin, and available free margin.
        """
        wallets = await self.get_futures_wallets()
        total_inr = 0.0
        locked_inr = 0.0
        for w in wallets:
            if w.get("currency_short_name") == "INR":
                total_inr = safe_float(w.get("balance"), 0.0)
                locked_inr = safe_float(w.get("locked_balance"), 0.0)
                break
        avail_inr = max(0.0, total_inr - locked_inr)
        return {
            "total_inr": round(total_inr, 2),
            "locked_inr": round(locked_inr, 2),
            "available_inr": round(avail_inr, 2),
            "usd_rate": inr_usd_rate,
            "total_usdt_equiv": round(total_inr / inr_usd_rate, 4),
            "available_usdt_equiv": round(avail_inr / inr_usd_rate, 4)
        }

    async def get_usable_balance_usdt(self) -> float:
        """
        Calculates total usable USDT margin balance across Futures INR wallet, Futures USDT wallet, and Spot balances.
        """
        if self.is_paper:
            return 10.0

        fut_inr = await self.get_futures_inr_balance()
        fut_free_usdt = safe_float(fut_inr.get("available_usdt_equiv"), 0.0)

        balances = await self.get_account_balances()
        spot_usdt = safe_float(balances.get("USDT"), 0.0)
        spot_inr = safe_float(balances.get("INR"), 0.0) / 87.5

        return round(fut_free_usdt + spot_usdt + spot_inr, 4)

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
            
        # Test 2: Authenticated endpoint & account balances (Spot & Futures INR)
        try:
            balances = await self.get_account_balances()
            fut_inr_info = await self.get_futures_inr_balance()
            result["futures_inr_wallet"] = fut_inr_info
            
            if balances is not None or fut_inr_info.get("total_inr", 0) > 0:
                result["auth_api"] = "AUTHENTICATED"
                result["balances"] = {k: round(v, 6) for k, v in balances.items() if v > 0}
                
                spot_usdt = safe_float(balances.get("USDT"), 0.0) if balances else 0.0
                spot_inr = safe_float(balances.get("INR"), 0.0) if balances else 0.0
                fut_free_usdt = safe_float(fut_inr_info.get("available_usdt_equiv"), 0.0)
                fut_total_usdt = safe_float(fut_inr_info.get("total_usdt_equiv"), 0.0)
                
                # Total usable margin is available futures free margin + spot assets
                total_usable_usdt = round(fut_free_usdt + spot_usdt + (spot_inr / 87.5), 4)
                total_equity_usdt = round(fut_total_usdt + spot_usdt + (spot_inr / 87.5), 4)
                
                result["total_usdt_balance"] = total_equity_usdt
                result["available_usdt_balance"] = total_usable_usdt
                result["futures_inr_balance"] = safe_float(fut_inr_info.get("total_inr"), 0.0)
                result["futures_inr_available"] = safe_float(fut_inr_info.get("available_inr"), 0.0)
                result["futures_inr_locked"] = safe_float(fut_inr_info.get("locked_inr"), 0.0)
                
                # Check CoinDCX minimum notional margin sufficiency
                if self.is_paper:
                    result["is_balance_sufficient"] = True
                    result["strategy_status"] = "SIMULATED_EXECUTION"
                    result["strategy_message"] = f"Paper trading active with ${total_usable_usdt:.2f} simulated capital."
                else:
                    if total_usable_usdt >= result["min_required_usdt"] or total_equity_usdt >= result["min_required_usdt"]:
                        result["is_balance_sufficient"] = True
                        result["strategy_status"] = "LIVE_EXECUTION_READY"
                        result["strategy_message"] = (
                            f"CoinDCX Futures INR Wallet Verified! Total: ₹{fut_inr_info.get('total_inr', 0.0):.2f} INR "
                            f"(~${total_equity_usdt:.2f} USDT) | Available Free Margin: ₹{fut_inr_info.get('available_inr', 0.0):.2f} INR "
                            f"(~${total_usable_usdt:.2f} USDT). Dynamic live scalper is fully funded and active!"
                        )
                    else:
                        result["is_balance_sufficient"] = False
                        result["strategy_status"] = "BALANCE_GUARD_ACTIVE"
                        result["strategy_message"] = (
                            f"Account verified! Total balance is ₹{fut_inr_info.get('total_inr', 0.0):.2f} INR (~${total_usable_usdt:.4f} USDT). "
                            f"CoinDCX requires minimum $6.00 USDT per futures order. "
                            f"Market screener & AI indicators are actively analyzing live data; "
                            f"live execution orders are safely held by Balance Guard until wallet margin reaches $6.00."
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
                open_pos = [p for p in pos if isinstance(p, dict) and abs(safe_float(p.get("active_pos"), 0.0)) > 1e-6]
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
        target_notional_usdt: Optional[float] = None,
        price: float = 1.0,
        requested_leverage: Optional[float] = None,
        **kwargs
    ) -> tuple[float, int, float]:
        """
        Guarantees order parameters strictly comply with CoinDCX rules:
        1. Leverage <= max permitted leverage for instrument
        2. Quantity >= min_quantity and rounded to exact quantity_increment
        3. Total notional >= min_notional required by CoinDCX
        Returns: (safe_quantity, safe_leverage, actual_notional)
        """
        notional = safe_float(target_notional_usdt or kwargs.get("target_notional"), 6.0)
        if notional <= 0:
            notional = 6.0

        lev = safe_float(requested_leverage or kwargs.get("desired_leverage"), 10.0)
        if lev <= 0:
            lev = 10.0

        safe_price = safe_float(price, 1.0)
        if safe_price <= 0:
            safe_price = 1.0

        details = await self.get_cached_instrument_details(pair)
        if not isinstance(details, dict):
            details = {}

        # Max leverage check
        raw_max_lev = (
            details.get("max_leverage_long")
            or details.get("max_leverage")
            or details.get("leverage")
        )
        max_lev = safe_float(raw_max_lev, 20.0)
        if max_lev <= 0:
            max_lev = 20.0
        safe_leverage = max(1, int(min(lev, max_lev)))

        # Min notional check
        raw_min_notional = details.get("min_notional") or details.get("minimum_notional")
        min_notional = safe_float(raw_min_notional, 6.0)
        if min_notional <= 0:
            min_notional = 6.0

        # Min quantity check
        raw_min_qty = (
            details.get("min_quantity")
            or details.get("minimum_quantity")
            or details.get("min_order_quantity")
        )
        min_qty = safe_float(raw_min_qty, 0.001)
        if min_qty <= 0:
            min_qty = 0.001

        # Quantity increment check
        raw_qty_inc = (
            details.get("quantity_increment")
            or details.get("step")
            or details.get("quantity_step")
            or details.get("contract_size")
        )
        qty_inc = safe_float(raw_qty_inc, 0.001)
        if qty_inc <= 0:
            qty_inc = 0.001

        # Buffer notional slightly above min_notional
        effective_notional = max(notional, min_notional * 1.02)
        raw_qty = effective_notional / max(safe_price, 1e-6)

        # Ceil to valid step increment
        steps = math.ceil(round(raw_qty / qty_inc, 6))
        safe_qty = steps * qty_inc
        safe_qty = max(safe_qty, min_qty)

        # Format decimals cleanly based on qty_inc
        if qty_inc >= 1.0:
            safe_qty = float(round(safe_qty))
        else:
            inc_str = f"{qty_inc:.8f}".rstrip("0")
            decimals = len(inc_str.split(".")[1]) if "." in inc_str else 3
            safe_qty = round(safe_qty, max(0, min(decimals, 8)))

        actual_notional = round(safe_qty * safe_price, 4)
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
                # Sort and slice to requested depth safely
                if isinstance(bids, dict):
                    sorted_bids = dict(sorted([(p, q) for p, q in bids.items() if p is not None], key=lambda x: safe_float(x[0]), reverse=True)[:depth])
                else:
                    sorted_bids = bids
                if isinstance(asks, dict):
                    sorted_asks = dict(sorted([(p, q) for p, q in asks.items() if p is not None], key=lambda x: safe_float(x[0]))[:depth])
                else:
                    sorted_asks = asks
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
        margin_currency: Optional[str] = None,
        time_in_force: Optional[str] = "good_till_cancel"
    ) -> Dict[str, Any]:
        """
        Place a new Futures order (market, limit, stop_market, take_profit_market).
        """
        margin_curr = margin_currency or settings.MARGIN_CURRENCY
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
                "margin_currency_short_name": margin_curr,
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
            "margin_currency_short_name": margin_curr
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
            if isinstance(p, dict) and p.get("pair") == symbol and abs(safe_float(p.get("active_pos"), 0.0)) > 1e-6:
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
        body = {"timestamp": timestamp, "id": str(order_id)}
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/derivatives/futures/orders/cancel"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        return {"status_code": resp.status_code, "text": resp.text}

    async def cancel_all_open_orders_for_position(self, position_id: str) -> Dict[str, Any]:
        """
        Cancels all open orders (including untriggered TP & SL triggers) for a specific position.
        Ensures zero orphan trigger orders remain in CoinDCX's 'Active Orders' tab.
        """
        if self.is_paper:
            return {"message": "success", "status": 200, "code": 200}

        actual_id = position_id
        if not (len(position_id) == 36 and position_id.count("-") == 4):
            pos = await self.get_position_by_symbol(position_id)
            if pos and pos.get("id"):
                actual_id = pos["id"]

        timestamp = int(round(time.time() * 1000))
        body = {"timestamp": timestamp, "id": actual_id}
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/derivatives/futures/positions/cancel_all_open_orders_for_position"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text}

    async def cancel_all_futures_open_orders(self, margin_currencies: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Cancels all open orders across the entire futures account.
        """
        if self.is_paper:
            self._paper_orders.clear()
            return {"message": "success", "status": 200, "code": 200}

        currencies = margin_currencies or ["INR", "USDT"]
        timestamp = int(round(time.time() * 1000))
        body = {"timestamp": timestamp, "margin_currency_short_name": currencies}
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/derivatives/futures/positions/cancel_all_open_orders"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text}

    async def get_futures_active_orders(
        self,
        margin_currencies: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetches all currently open and untriggered (TP/SL) orders from CoinDCX Futures.
        """
        if self.is_paper:
            return [o for o in self._paper_orders.values() if o.get("status") in ("open", "untriggered", "init")]

        currencies = margin_currencies or ["INR", "USDT"]
        timestamp = int(round(time.time() * 1000))
        active_orders: List[Dict[str, Any]] = []

        # CoinDCX requires side ('buy' or 'sell') and status
        for side in ["buy", "sell"]:
            for status_filter in ["open", "untriggered"]:
                body = {
                    "timestamp": timestamp,
                    "status": status_filter,
                    "side": side,
                    "page": "1",
                    "size": "50",
                    "margin_currency_short_name": currencies
                }
                json_body, signature = self._generate_signature(body)
                url = f"{self.base_url}/exchange/v1/derivatives/futures/orders"
                headers = self._get_auth_headers(signature)
                try:
                    resp = await self._safe_post(url, data=json_body, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, list):
                            for o in data:
                                o["product_type"] = "FUTURES"
                                active_orders.append(o)
                except Exception as e:
                    logger.debug(f"Error fetching futures orders ({side}, {status_filter}): {e}")

        # Deduplicate by order id
        seen = set()
        deduped = []
        for o in active_orders:
            oid = o.get("id")
            if oid and oid not in seen:
                seen.add(oid)
                deduped.append(o)
        return deduped

    # ----------------------------------------------------------------------
    # SPOT ORDER MANAGEMENT (create_multiple, status, cancel, active_orders)
    # ----------------------------------------------------------------------

    async def get_spot_active_orders(self, market: str = "USDTINR") -> List[Dict[str, Any]]:
        """Fetches active orders for a specific market in CoinDCX Spot."""
        if self.is_paper:
            return []
        timestamp = int(round(time.time() * 1000))
        body = {"market": market, "timestamp": timestamp}
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/orders/active_orders"
        headers = self._get_auth_headers(signature)
        try:
            resp = await self._safe_post(url, data=json_body, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                orders = data.get("orders", []) if isinstance(data, dict) else data
                for o in orders:
                    o["product_type"] = "SPOT"
                return orders
            return []
        except Exception as e:
            logger.error(f"Error fetching spot active orders: {e}")
            return []

    async def get_spot_order_status(self, order_id: Union[int, str]) -> Dict[str, Any]:
        """Fetches status of a spot order by numeric ID or client_order_id."""
        timestamp = int(round(time.time() * 1000))
        body: Dict[str, Any] = {"timestamp": timestamp}
        if str(order_id).isdigit():
            body["id"] = int(order_id)
        else:
            body["client_order_id"] = str(order_id)

        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/orders/status"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text}

    async def get_multiple_spot_order_status(
        self,
        ids: Optional[List[Union[int, str]]] = None,
        client_order_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetches status of multiple spot orders (up to 10 IDs per request)."""
        timestamp = int(round(time.time() * 1000))
        body: Dict[str, Any] = {"timestamp": timestamp}
        if ids:
            body["ids"] = [int(i) if str(i).isdigit() else i for i in ids[:10]]
        if client_order_ids:
            body["client_order_ids"] = client_order_ids[:10]

        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/orders/status_multiple"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        try:
            return resp.json()
        except Exception:
            return [{"status_code": resp.status_code, "text": resp.text}]

    async def cancel_spot_order(self, order_id: Union[int, str]) -> Dict[str, Any]:
        """Cancels a single active spot order."""
        timestamp = int(round(time.time() * 1000))
        body: Dict[str, Any] = {"timestamp": timestamp}
        if str(order_id).isdigit():
            body["id"] = int(order_id)
        else:
            body["client_order_id"] = str(order_id)

        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/orders/cancel"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text}

    async def cancel_spot_orders_by_ids(self, ids: List[Union[int, str]]) -> Dict[str, Any]:
        """Cancels multiple active spot orders by numeric IDs (max 10 per request)."""
        timestamp = int(round(time.time() * 1000))
        body = {
            "timestamp": timestamp,
            "ids": [str(i) for i in ids[:10]]
        }
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/orders/cancel_by_ids"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text}

    async def cancel_all_spot_orders(self, market: str, side: Optional[str] = None) -> Dict[str, Any]:
        """Cancels all active spot orders in a market."""
        timestamp = int(round(time.time() * 1000))
        body: Dict[str, Any] = {"market": market, "timestamp": timestamp}
        if side:
            body["side"] = side
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/orders/cancel_all"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text}

    async def cancel_any_order(self, order_id: str) -> Dict[str, Any]:
        """
        Universal order canceler: dynamically detects whether the ID is a Futures UUID
        or a Spot numeric ID, and dispatches to the correct CoinDCX endpoint.
        """
        if len(str(order_id)) == 36 and str(order_id).count("-") == 4:
            return await self.cancel_futures_order(str(order_id))
        else:
            return await self.cancel_spot_order(order_id)

    async def get_all_active_orders(self) -> List[Dict[str, Any]]:
        """
        Unified aggregator: returns all open and untriggered orders across Futures and Spot.
        """
        futures_orders = await self.get_futures_active_orders()
        spot_orders = await self.get_spot_active_orders("USDTINR")
        return futures_orders + spot_orders

    # ----------------------------------------------------------------------
    # FUTURES TP/SL & COMPREHENSIVE POSITION EXIT
    # ----------------------------------------------------------------------

    async def create_futures_tpsl(
        self,
        position_id: str,
        tp_stop_price: float,
        sl_stop_price: float
    ) -> Dict[str, Any]:
        """
        Create Take Profit and Stop Loss triggers directly on an open position.
        Accepts either exact position UUID or symbol (e.g. 'B-BTC_USDT').
        Includes progressive retry loop (up to 10 attempts, 5 seconds) to wait
        for exchange position registration after order fill.
        """
        if self.is_paper:
            for pair, pos in self._paper_positions.items():
                if pos.get("id") == position_id or pair == position_id or pos.get("pair") == position_id:
                    pos["take_profit_trigger"] = tp_stop_price
                    pos["stop_loss_trigger"] = sl_stop_price
                    pos["tp_price"] = tp_stop_price
                    pos["sl_price"] = sl_stop_price
                    return {
                        "status": "success",
                        "take_profit": {"stop_price": tp_stop_price, "order_type": "take_profit_market"},
                        "stop_loss": {"stop_price": sl_stop_price, "order_type": "stop_market"}
                    }
            return {"status": "success"}

        actual_id = position_id
        if position_id.startswith("B-"):
            pos = None
            for attempt in range(10):
                pos = await self.get_position_by_symbol(position_id)
                if pos and pos.get("id"):
                    actual_id = pos["id"]
                    break
                await asyncio.sleep(0.5)
            if not pos or not pos.get("id"):
                logger.warning(f"No active open position found on exchange for {position_id} after 10 attempts to attach TP/SL")
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
        if resp.status_code == 200:
            logger.info(f"Attached TP/SL to position {actual_id} on CoinDCX: TP={tp_stop_price}, SL={sl_stop_price}")
        else:
            logger.warning(f"CoinDCX create_tpsl [{resp.status_code}] for {actual_id}: {resp.text}")
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text}

    async def verify_and_reconcile_position_tpsl(
        self,
        position: Dict[str, Any],
        active_orders: Optional[List[Dict[str, Any]]] = None,
        default_tp_ratio: float = 0.0085,
        default_sl_ratio: float = 0.0045
    ) -> Dict[str, Any]:
        """
        1-Minute Safety Verification & Auto-Healing Guard:
        Inspects an active futures position and verifies whether it has active Stop Loss and Take Profit triggers.
        If either is missing, automatically computes algorithmic SL (-0.45%) and TP (+0.85%) and attaches them to CoinDCX.
        """
        pair = position.get("pair") or ""
        pos_id = position.get("id") or pair
        active_qty = safe_float(position.get("active_pos"), 0.0)
        
        if abs(active_qty) < 1e-6:
            return {"status": "inactive_position", "pair": pair}
            
        is_long = active_qty > 0
        avg_price = safe_float(position.get("avg_price") or position.get("entry_price"), 0.0)
        if avg_price <= 0:
            avg_price = safe_float(position.get("mark_price"), 0.0)
        if avg_price <= 0:
            return {"status": "invalid_entry_price", "pair": pair}

        # Check existing triggers on position object
        existing_tp = safe_float(position.get("take_profit_trigger") or position.get("tp_price"), 0.0)
        existing_sl = safe_float(position.get("stop_loss_trigger") or position.get("sl_price"), 0.0)

        # Check active orders for untriggered stop/take-profit orders if not on position
        if active_orders:
            for o in active_orders:
                if not isinstance(o, dict):
                    continue
                if o.get("pair") == pair:
                    otype = str(o.get("order_type", "")).lower()
                    sprice = safe_float(o.get("stop_price") or o.get("price"), 0.0)
                    if sprice > 0:
                        if "take_profit" in otype:
                            existing_tp = sprice
                        elif "stop" in otype:
                            existing_sl = sprice

        has_tp = existing_tp > 0
        has_sl = existing_sl > 0

        if has_tp and has_sl:
            return {
                "status": "fully_protected",
                "pair": pair,
                "take_profit": existing_tp,
                "stop_loss": existing_sl
            }

        # Auto-heal: Compute algorithmic TP and SL
        if is_long:
            calc_tp = round(avg_price * (1.0 + default_tp_ratio), 6)
            calc_sl = round(avg_price * (1.0 - default_sl_ratio), 6)
        else:
            calc_tp = round(avg_price * (1.0 - default_tp_ratio), 6)
            calc_sl = round(avg_price * (1.0 + default_sl_ratio), 6)

        target_tp = existing_tp if has_tp else calc_tp
        target_sl = existing_sl if has_sl else calc_sl

        logger.info(
            f"[AUTO-HEAL 1-MIN] Missing triggers detected on {pair} (TP: {has_tp}, SL: {has_sl}). "
            f"Attaching algorithmic triggers: TP={target_tp}, SL={target_sl} (Entry: {avg_price})"
        )

        res = await self.create_futures_tpsl(
            position_id=pos_id,
            tp_stop_price=target_tp,
            sl_stop_price=target_sl
        )

        # Update position record in place
        position["take_profit_trigger"] = target_tp
        position["stop_loss_trigger"] = target_sl
        position["tp_price"] = target_tp
        position["sl_price"] = target_sl

        return {
            "status": "auto_healed",
            "pair": pair,
            "position_id": pos_id,
            "attached_tp": target_tp,
            "attached_sl": target_sl,
            "entry_price": avg_price,
            "result": res
        }

    async def exit_futures_position(self, position_id: str) -> Dict[str, Any]:
        """
        Robustly and completely exits a futures position:
        1. Resolves symbol, trade ID, or UUID to the live exchange position UUID.
        2. First cancels all open and untriggered TP/SL orders for that position so NO
           lingering orders remain in the CoinDCX app under 'Active Orders'.
        3. Calls CoinDCX's positions/exit endpoint to liquidate the contract at market.
        4. Verifies that active position reaches zero.
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
        pair_symbol = position_id if position_id.startswith("B-") else None

        # Resolve symbol or trade_id to exchange position UUID
        if not (len(position_id) == 36 and position_id.count("-") == 4):
            pos = await self.get_position_by_symbol(position_id)
            if pos and pos.get("id"):
                actual_id = pos["id"]
                pair_symbol = pos.get("pair")
            else:
                all_active = await self.get_futures_positions()
                for p in all_active:
                    if p.get("pair") == position_id or p.get("id") == position_id:
                        actual_id = p["id"]
                        pair_symbol = p.get("pair")
                        break
                if actual_id == position_id and position_id.startswith("B-"):
                    logger.warning(f"No active open position found on CoinDCX for {position_id} to exit")
                    return {"status": "no_active_position", "message": f"No active position for {position_id}"}

        # Step 1: Cancel any open or untriggered TP/SL orders on this position FIRST
        # This prevents orphan trigger orders from remaining in CoinDCX's 'Active Orders' tab!
        cancel_orders_res = await self.cancel_all_open_orders_for_position(actual_id)
        logger.info(f"Canceled open orders for position {actual_id}: {cancel_orders_res}")

        # Step 2: Liquidate position at market
        timestamp = int(round(time.time() * 1000))
        body = {"timestamp": timestamp, "id": actual_id}
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/derivatives/futures/positions/exit"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)

        exit_data = {}
        try:
            exit_data = resp.json()
        except Exception:
            exit_data = {"status_code": resp.status_code, "text": resp.text}

        if resp.status_code == 200:
            logger.info(f"CoinDCX position {actual_id} exit executed: {resp.text}")
        else:
            logger.error(f"CoinDCX position {actual_id} exit returned [{resp.status_code}]: {resp.text}")

        # Step 3: Brief verification check
        await asyncio.sleep(0.5)
        remaining_positions = await self.get_futures_positions()
        is_cleared = not any(
            p.get("id") == actual_id or (pair_symbol and p.get("pair") == pair_symbol)
            for p in remaining_positions
        )

        return {
            "status": "success" if (resp.status_code == 200 or is_cleared) else "error",
            "position_id": actual_id,
            "pair": pair_symbol,
            "is_cleared": is_cleared,
            "cancel_open_orders_response": cancel_orders_res,
            "exit_response": exit_data
        }

    async def get_futures_positions(
        self,
        margin_currency: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all currently open derivatives/futures positions across INR and USDT.
        Enriches each position with real P&L in USDT & INR and ROE percentage.
        """
        if self.is_paper:
            return [p for p in self._paper_positions.values() if abs(p.get("active_pos", 0)) > 0]

        currencies_to_check = [margin_currency] if margin_currency else ["INR", "USDT"]
        all_active_positions: List[Dict[str, Any]] = []

        timestamp = int(round(time.time() * 1000))
        for curr in currencies_to_check:
            body = {
                "timestamp": timestamp,
                "page": "1",
                "size": "50",
                "margin_currency_short_name": [curr]
            }
            json_body, signature = self._generate_signature(body)
            url = f"{self.base_url}/exchange/v1/derivatives/futures/positions"
            headers = self._get_auth_headers(signature)
            try:
                resp = await self._safe_post(url, data=json_body, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        for p in data:
                            if not isinstance(p, dict):
                                continue
                            active_qty = safe_float(p.get("active_pos"), 0.0)
                            if abs(active_qty) > 1e-6:
                                avg_price = safe_float(p.get("avg_price"), 0.0)
                                mark_price = safe_float(p.get("mark_price"), avg_price)
                                settle_rate = safe_float(p.get("settlement_currency_avg_price"), 87.5)
                                locked_margin = safe_float(p.get("locked_margin") or p.get("locked_user_margin"), 0.0)
                                
                                is_long = active_qty > 0
                                side = "BUY" if is_long else "SELL"
                                abs_qty = abs(active_qty)
                                
                                pnl_usdt = (mark_price - avg_price) * abs_qty if is_long else (avg_price - mark_price) * abs_qty
                                pnl_inr = pnl_usdt * settle_rate
                                roe_pct = (pnl_usdt / locked_margin * 100.0) if locked_margin > 0 else 0.0
                                
                                p["side"] = side
                                p["abs_quantity"] = round(abs_qty, 4)
                                p["entry_price"] = avg_price
                                p["mark_price"] = mark_price
                                p["unrealized_pnl_usdt"] = round(pnl_usdt, 4)
                                p["unrealized_pnl_inr"] = round(pnl_inr, 2)
                                p["roe_percent"] = round(roe_pct, 2)
                                p["tp_price"] = p.get("take_profit_trigger")
                                p["sl_price"] = p.get("stop_loss_trigger")
                                p["locked_margin_usdt"] = round(locked_margin, 4)
                                p["locked_margin_inr"] = round(locked_margin * settle_rate, 2)
                                all_active_positions.append(p)
            except Exception as e:
                logger.error(f"Error querying positions for {curr}: {e}")

        return all_active_positions

    async def update_position_leverage(
        self,
        pair: str,
        leverage: float,
        margin_currency: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update leverage on an instrument before or during trading."""
        if self.is_paper:
            if pair in self._paper_positions:
                self._paper_positions[pair]["leverage"] = leverage
            return {"message": "success", "status": 200}

        margin_curr = margin_currency or settings.MARGIN_CURRENCY
        timestamp = int(round(time.time() * 1000))
        body = {
            "timestamp": timestamp,
            "leverage": str(int(leverage)),
            "pair": pair,
            "margin_currency_short_name": margin_curr
        }
        json_body, signature = self._generate_signature(body)
        url = f"{self.base_url}/exchange/v1/derivatives/futures/positions/update_leverage"
        headers = self._get_auth_headers(signature)
        resp = await self._safe_post(url, data=json_body, headers=headers)
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text}

    async def close(self):
        """Close the underlying HTTP connection pool."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
