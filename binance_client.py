"""
AlphaScalper - Binance Global Market Data Client
Provides high-performance asynchronous REST and WebSocket access to Binance Futures and Spot market data:
- Universe scanning across 700+ perpetual contracts
- Multi-timeframe authentic historical candlestick ingestion (1m, 5m, 15m, 1h)
- Delta volume (Taker Buy vs Seller Volume) extraction
- L2 Order Book depth & Order Book Imbalance (OBI)
- Resilient WebSocket streaming with automatic reconnection
- Bidirectional symbol mapping: B-XXX_USDT (CoinDCX) <-> XXXUSDT (Binance)
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx
import numpy as np
import websockets

logger = logging.getLogger("AlphaScalper.BinanceClient")


def to_binance_symbol(coindcx_symbol: str) -> str:
    """
    Converts CoinDCX contract symbol to Binance symbol.
    Examples:
        'B-BTC_USDT' -> 'BTCUSDT'
        'B-1000PEPE_USDT' -> '1000PEPEUSDT'
        'B-DOGE_USDT' -> 'DOGEUSDT'
        'BTCUSDT' -> 'BTCUSDT'
    """
    clean = coindcx_symbol.strip().upper()
    if clean.startswith("B-"):
        clean = clean[2:]
    clean = clean.replace("_", "")
    return clean


def to_coindcx_symbol(binance_symbol: str) -> str:
    """
    Converts Binance symbol to CoinDCX format.
    Examples:
        'BTCUSDT' -> 'B-BTC_USDT'
        '1000PEPEUSDT' -> 'B-1000PEPE_USDT'
        'DOGEUSDT' -> 'B-DOGE_USDT'
    """
    clean = binance_symbol.strip().upper()
    if clean.startswith("B-"):
        return clean
    if clean.endswith("USDT"):
        base = clean[:-4]
        return f"B-{base}_USDT"
    return f"B-{clean}"


class BinanceClient:
    def __init__(
        self,
        fapi_base_url: str = "https://fapi.binance.com",
        api_base_url: str = "https://api.binance.com",
        ws_stream_url: str = "wss://fstream.binance.com/stream"
    ):
        self.fapi_base_url = fapi_base_url.rstrip("/")
        self.api_base_url = api_base_url.rstrip("/")
        self.ws_stream_url = ws_stream_url
        self._http_client: Optional[httpx.AsyncClient] = None
        self._active_ws = None
        
        # In-memory latest market state
        self.latest_tickers: Dict[str, Dict[str, Any]] = {}   # BinanceSymbol -> TickerDict
        self.latest_prices: Dict[str, float] = {}            # BinanceSymbol -> Price
        self.latest_klines_1m: Dict[str, Dict[str, Any]] = {} # BinanceSymbol -> KlineDict
        self.is_streaming: bool = False
        self._stream_task: Optional[asyncio.Task] = None
        self._contract_rules: Dict[str, Dict[str, Any]] = {}  # BinanceSymbol -> RuleDict
        self._monitored_symbols: List[str] = ["btcusdt", "ethusdt", "solusdt"]

    async def get_client(self) -> httpx.AsyncClient:
        """Returns active persistent httpx AsyncClient with connection pooling."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=8.0,
                limits=httpx.Limits(max_keepalive_connections=100, max_connections=200)
            )
        return self._http_client

    async def close(self):
        """Closes HTTP client and stops WebSocket streaming."""
        self.is_streaming = False
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def ping(self) -> bool:
        """Verifies connectivity to Binance Futures REST API."""
        try:
            client = await self.get_client()
            resp = await client.get(f"{self.fapi_base_url}/fapi/v1/ping", timeout=4.0)
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Binance ping failed: {e}")
            return False

    async def get_24hr_tickers(self) -> List[Dict[str, Any]]:
        """
        Fetches 24-hour ticker price change statistics for all perpetual contracts in a single call.
        Returns 700+ contracts with price, volume, quoteVolume, priceChangePercent, high, low.
        """
        client = await self.get_client()
        try:
            resp = await client.get(f"{self.fapi_base_url}/fapi/v1/ticker/24hr", timeout=6.0)
            if resp.status_code == 200:
                data = resp.json()
                for item in data:
                    sym = item.get("symbol")
                    if sym:
                        self.latest_tickers[sym] = item
                        try:
                            self.latest_prices[sym] = float(item.get("lastPrice", 0.0))
                        except (ValueError, TypeError):
                            pass
                return data
            else:
                logger.error(f"Failed to fetch Binance 24hr tickers: HTTP {resp.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching Binance 24hr tickers: {e}")
            return []

    async def get_exchange_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetches and caches Binance Futures contract specifications:
        tickSize (price filter), stepSize (lot size), minQty, pricePrecision, minNotional.
        """
        if self._contract_rules and not force_refresh:
            return self._contract_rules
        client = await self.get_client()
        try:
            resp = await client.get(f"{self.fapi_base_url}/fapi/v1/exchangeInfo", timeout=8.0)
            if resp.status_code == 200:
                data = resp.json()
                for s in data.get("symbols", []):
                    sym = s.get("symbol")
                    if not sym:
                        continue
                    price_precision = int(s.get("pricePrecision", 2))
                    qty_precision = int(s.get("quantityPrecision", 3))
                    tick_size = 0.01
                    step_size = 0.001
                    min_qty = 0.001
                    min_notional = 5.0
                    for f in s.get("filters", []):
                        ftype = f.get("filterType")
                        if ftype == "PRICE_FILTER":
                            tick_size = float(f.get("tickSize", 0.01))
                        elif ftype == "LOT_SIZE":
                            step_size = float(f.get("stepSize", 0.001))
                            min_qty = float(f.get("minQty", 0.001))
                        elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                            min_notional = float(f.get("notional", 5.0))
                    self._contract_rules[sym] = {
                        "symbol": sym,
                        "price_precision": price_precision,
                        "quantity_precision": qty_precision,
                        "tick_size": tick_size,
                        "step_size": step_size,
                        "min_qty": min_qty,
                        "min_notional": min_notional,
                    }
                logger.info(f"Cached contract rules for {len(self._contract_rules)} Binance Futures pairs")
        except Exception as e:
            logger.debug(f"Error fetching Binance exchangeInfo: {e}")
        return self._contract_rules

    def get_contract_rule(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Returns cached contract rules for symbol (supports B-BTC_USDT, BTCUSDT, etc.)."""
        b_sym = to_binance_symbol(symbol)
        return self._contract_rules.get(b_sym)


    async def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 120
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves authentic candlestick history directly from Binance Futures.
        Returns structured numpy arrays for open, high, low, close, volume, taker_buy_volume.
        """
        b_sym = to_binance_symbol(symbol)
        client = await self.get_client()
        url = f"{self.fapi_base_url}/fapi/v1/klines"
        params = {
            "symbol": b_sym,
            "interval": interval,
            "limit": limit
        }

        try:
            resp = await client.get(url, params=params, timeout=5.0)
            if resp.status_code == 200:
                raw_bars = resp.json()
                if raw_bars and len(raw_bars) >= 10:
                    opens = np.array([float(b[1]) for b in raw_bars], dtype=np.float64)
                    highs = np.array([float(b[2]) for b in raw_bars], dtype=np.float64)
                    lows = np.array([float(b[3]) for b in raw_bars], dtype=np.float64)
                    closes = np.array([float(b[4]) for b in raw_bars], dtype=np.float64)
                    volumes = np.array([float(b[5]) for b in raw_bars], dtype=np.float64)
                    quote_volumes = np.array([float(b[7]) for b in raw_bars], dtype=np.float64)
                    taker_buy_vols = np.array([float(b[9]) for b in raw_bars], dtype=np.float64)
                    open_times = np.array([int(b[0]) for b in raw_bars], dtype=np.int64)

                    # Compute Delta Volume (Taker Buy vs Taker Sell)
                    taker_sell_vols = np.maximum(0.0, volumes - taker_buy_vols)
                    delta_volumes = taker_buy_vols - taker_sell_vols

                    return {
                        "open": opens,
                        "high": highs,
                        "low": lows,
                        "close": closes,
                        "volume": volumes,
                        "quote_volume": quote_volumes,
                        "taker_buy_volume": taker_buy_vols,
                        "taker_sell_volume": taker_sell_vols,
                        "delta_volume": delta_volumes,
                        "open_time": open_times,
                        "interval": interval,
                        "last_fetch": time.time(),
                        "is_synthetic": False
                    }
            elif resp.status_code == 400:
                # Fallback to Spot API if contract not on Futures
                return await self._get_spot_klines_fallback(b_sym, interval, limit)
            else:
                logger.debug(f"Binance klines error {b_sym} {interval}: HTTP {resp.status_code}")
        except Exception as e:
            logger.debug(f"Exception fetching Binance klines {b_sym} {interval}: {e}")

        return None

    async def _get_spot_klines_fallback(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 120
    ) -> Optional[Dict[str, Any]]:
        """Spot klines fallback if contract is not listed on perpetual futures."""
        client = await self.get_client()
        url = f"{self.api_base_url}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        try:
            resp = await client.get(url, params=params, timeout=5.0)
            if resp.status_code == 200:
                raw_bars = resp.json()
                if raw_bars and len(raw_bars) >= 10:
                    opens = np.array([float(b[1]) for b in raw_bars], dtype=np.float64)
                    highs = np.array([float(b[2]) for b in raw_bars], dtype=np.float64)
                    lows = np.array([float(b[3]) for b in raw_bars], dtype=np.float64)
                    closes = np.array([float(b[4]) for b in raw_bars], dtype=np.float64)
                    volumes = np.array([float(b[5]) for b in raw_bars], dtype=np.float64)
                    taker_buy_vols = np.array([float(b[9]) for b in raw_bars], dtype=np.float64)
                    taker_sell_vols = np.maximum(0.0, volumes - taker_buy_vols)
                    return {
                        "open": opens,
                        "high": highs,
                        "low": lows,
                        "close": closes,
                        "volume": volumes,
                        "taker_buy_volume": taker_buy_vols,
                        "taker_sell_volume": taker_sell_vols,
                        "delta_volume": taker_buy_vols - taker_sell_vols,
                        "interval": interval,
                        "last_fetch": time.time(),
                        "is_synthetic": False
                    }
        except Exception:
            pass
        return None

    async def get_multi_timeframe_candles(
        self,
        symbol: str,
        intervals: Optional[List[str]] = None,
        limit: int = 120
    ) -> Dict[str, Dict[str, Any]]:
        """
        Concurrently fetches authentic historical candles across multiple timeframes (1m, 5m, 15m)
        using asyncio.gather for parallel sub-100ms execution.
        """
        intervals = intervals or ["1m", "5m", "15m"]
        tasks = [self.get_klines(symbol, interval=tf, limit=limit) for tf in intervals]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        mtf_data: Dict[str, Dict[str, Any]] = {}
        for tf, res in zip(intervals, results):
            if isinstance(res, dict) and not isinstance(res, Exception):
                mtf_data[tf] = res

        return mtf_data

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """
        Retrieves live L2 Order Book depth from Binance Futures for precise micro Order Book Imbalance (OBI).
        """
        b_sym = to_binance_symbol(symbol)
        client = await self.get_client()
        url = f"{self.fapi_base_url}/fapi/v1/depth"
        params = {"symbol": b_sym, "limit": limit}

        try:
            resp = await client.get(url, params=params, timeout=4.0)
            if resp.status_code == 200:
                data = resp.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                
                # Compute Order Book Imbalance (OBI)
                bid_vol_10 = sum(float(b[1]) for b in bids[:10])
                ask_vol_10 = sum(float(a[1]) for a in asks[:10])
                total_vol = bid_vol_10 + ask_vol_10
                obi_10 = round((bid_vol_10 - ask_vol_10) / max(1e-6, total_vol), 4)

                return {
                    "symbol": symbol,
                    "binance_symbol": b_sym,
                    "bids": bids,
                    "asks": asks,
                    "bid_vol_10": bid_vol_10,
                    "ask_vol_10": ask_vol_10,
                    "obi_10": obi_10,
                    "timestamp": data.get("E", int(time.time() * 1000))
                }
        except Exception as e:
            logger.debug(f"Error fetching Binance orderbook for {b_sym}: {e}")
        return None

    def update_monitored_symbols(self, symbols: List[str]):
        """Updates the list of symbols being streamed over WebSocket."""
        clean = []
        for s in symbols:
            bs = to_binance_symbol(s).lower()
            if bs and bs not in clean:
                clean.append(bs)
        new_symbols = clean[:25]
        diff = [s for s in new_symbols if s not in self._monitored_symbols]
        self._monitored_symbols = new_symbols

        # If WebSocket is actively connected, send dynamic JSON-RPC SUBSCRIBE frame
        if diff and self._active_ws:
            try:
                sub_msg = {
                    "method": "SUBSCRIBE",
                    "params": [f"{s}@kline_1m" for s in diff],
                    "id": int(time.time() * 1000)
                }
                asyncio.create_task(self._active_ws.send(json.dumps(sub_msg)))
                logger.info(f"BinanceStreamManager dynamic subscription sent for {len(diff)} symbols: {diff}")
            except Exception as e:
                logger.debug(f"Dynamic subscribe error: {e}")

    async def start_stream_manager(self, on_price_update=None):
        """
        Runs background WebSocket streaming task.
        Streams 1m kline updates and ticks for monitored candidates.
        """
        self.is_streaming = True
        logger.info(f"BinanceStreamManager starting live stream for {len(self._monitored_symbols)} symbols...")

        while self.is_streaming:
            try:
                streams = [f"{s}@kline_1m" for s in self._monitored_symbols]
                stream_path = "/".join(streams)
                url = f"{self.ws_stream_url}?streams={stream_path}"

                async with websockets.connect(url, ping_interval=20, close_timeout=5) as ws:
                    self._active_ws = ws
                    logger.info("BinanceStreamManager: USD-M Futures WebSocket connected successfully!")
                    while self.is_streaming:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                            payload = json.loads(msg)
                            stream_name = payload.get("stream", "")
                            data = payload.get("data", {})
                            k = data.get("k", {})

                            if k:
                                b_sym = data.get("s", "").upper()
                                c_sym = to_coindcx_symbol(b_sym)
                                close_p = float(k.get("c", 0.0))
                                vol = float(k.get("v", 0.0))
                                is_closed = bool(k.get("x", False))

                                self.latest_prices[b_sym] = close_p
                                self.latest_klines_1m[b_sym] = {
                                    "symbol": c_sym,
                                    "binance_symbol": b_sym,
                                    "close": close_p,
                                    "high": float(k.get("h", close_p)),
                                    "low": float(k.get("l", close_p)),
                                    "open": float(k.get("o", close_p)),
                                    "volume": vol,
                                    "is_closed": is_closed,
                                    "timestamp": int(data.get("E", time.time() * 1000))
                                }

                                if on_price_update and callable(on_price_update):
                                    await on_price_update(c_sym, close_p, vol, is_closed)

                        except asyncio.TimeoutError:
                            # Send WebSocket ping to keep connection alive
                            await ws.ping()
                        except asyncio.CancelledError:
                            return
                        except Exception as e:
                            logger.debug(f"Error receiving WS stream: {e}")
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"BinanceStreamManager WS error: {e}. Reconnecting in 3 seconds...")
                await asyncio.sleep(3.0)

        logger.info("BinanceStreamManager stopped.")
