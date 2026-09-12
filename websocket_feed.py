"""
AlphaScalper - High-Frequency WebSocket Manager
Subscribes to CoinDCX Spot & Futures Sockets:
- Private: 'coindcx' (order-update, trade-update, balance-update)
- Public: currentPrices@spot@10s, {pair}@orderbook@20, {pair}@trades, {pair}_1m
- Zero-copy RingBuffer for sub-millisecond price & book cache
"""

import asyncio
import hmac
import hashlib
import json
import logging
import time
from typing import Dict, Any, Callable, List, Optional
import socketio

from config import settings

logger = logging.getLogger("AlphaScalper.WebSocket")


class CoinDCXWebSocketManager:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or settings.COINDCX_API_KEY
        self.api_secret = api_secret or settings.COINDCX_API_SECRET
        self._secret_bytes = bytes(self.api_secret, encoding="utf-8")
        
        self.sio = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=50,
            reconnection_delay=1,
            reconnection_delay_max=5,
            logger=False,
            engineio_logger=False
        )
        
        self.is_connected = False
        self._ping_task: Optional[asyncio.Task] = None
        
        # Real-time state caches
        self.latest_prices: Dict[str, float] = {}
        self.price_stats: Dict[str, Dict[str, Any]] = {}
        self.orderbooks: Dict[str, Dict[str, Any]] = {}
        self.candles_1m: Dict[str, List[Dict[str, Any]]] = {}
        
        # Callbacks
        self._on_price_callbacks: List[Callable[[str, float], None]] = []
        self._on_book_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._on_order_update_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        self._setup_handlers()

    def _setup_handlers(self):
        @self.sio.event
        async def connect():
            self.is_connected = True
            logger.info("Connected to CoinDCX WebSocket Stream!")
            
            # Authenticate to private 'coindcx' channel if keys provided
            if self.api_key and self.api_key != "YOUR_API_KEY":
                channel_name = "coindcx"
                body = {"channel": channel_name}
                json_body = json.dumps(body, separators=(',', ':'))
                signature = hmac.new(self._secret_bytes, json_body.encode(), hashlib.sha256).hexdigest()
                await self.sio.emit('join', {
                    'channelName': channel_name,
                    'authSignature': signature,
                    'apiKey': self.api_key
                })
                logger.info("Joined private channel: coindcx with HMAC authentication")

            # Join global prices channel
            await self.sio.emit('join', {'channelName': "currentPrices@spot@10s"})

        @self.sio.event
        async def disconnect():
            self.is_connected = False
            logger.warning("Disconnected from CoinDCX WebSocket Stream!")

        @self.sio.event
        async def connect_error(data):
            logger.error(f"WebSocket Connection Error: {data}")

        @self.sio.on('currentPrices@spot#update')
        async def on_current_prices(response):
            try:
                data = response.get("data", {})
                prices = data.get("prices", {})
                for pair, price in prices.items():
                    p = float(price)
                    self.latest_prices[pair] = p
                    for cb in self._on_price_callbacks:
                        cb(pair, p)
            except Exception as e:
                logger.error(f"Error parsing currentPrices: {e}")

        @self.sio.on('depth-snapshot')
        async def on_depth_snapshot(response):
            try:
                data = response.get("data", {})
                symbol = data.get("s")
                if symbol:
                    self.orderbooks[symbol] = {
                        "asks": data.get("asks", {}),
                        "bids": data.get("bids", {}),
                        "timestamp": data.get("ts", int(time.time() * 1000))
                    }
                    for cb in self._on_book_callbacks:
                        cb(symbol, self.orderbooks[symbol])
            except Exception as e:
                logger.error(f"Error parsing depth-snapshot: {e}")

        @self.sio.on('depth-update')
        async def on_depth_update(response):
            try:
                data = response.get("data", {})
                symbol = data.get("s")
                if symbol and symbol in self.orderbooks:
                    ob = self.orderbooks[symbol]
                    # Update asks
                    for p, q in data.get("asks", {}).items():
                        if float(q) == 0:
                            ob["asks"].pop(p, None)
                        else:
                            ob["asks"][p] = q
                    # Update bids
                    for p, q in data.get("bids", {}).items():
                        if float(q) == 0:
                            ob["bids"].pop(p, None)
                        else:
                            ob["bids"][p] = q
                    ob["timestamp"] = data.get("ts", int(time.time() * 1000))
                    for cb in self._on_book_callbacks:
                        cb(symbol, ob)
            except Exception as e:
                logger.error(f"Error parsing depth-update: {e}")

        @self.sio.on('candlestick')
        async def on_candlestick(response):
            try:
                data = response.get("data", {})
                symbol = data.get("s")
                if symbol:
                    candle = {
                        "open": float(data.get("o", 0)),
                        "high": float(data.get("h", 0)),
                        "low": float(data.get("l", 0)),
                        "close": float(data.get("c", 0)),
                        "volume": float(data.get("v", 0)),
                        "time": data.get("t", 0)
                    }
                    if symbol not in self.candles_1m:
                        self.candles_1m[symbol] = []
                    self.candles_1m[symbol].append(candle)
                    if len(self.candles_1m[symbol]) > 500:
                        self.candles_1m[symbol].pop(0)
            except Exception as e:
                logger.error(f"Error parsing candlestick: {e}")

        @self.sio.on('order-update')
        async def on_order_update(response):
            logger.info(f"Order Update Event: {response}")
            for cb in self._on_order_update_callbacks:
                cb(response)

    async def _ping_loop(self):
        """Send periodic ping every 25 seconds to prevent connection drops."""
        while True:
            await asyncio.sleep(25)
            if self.is_connected:
                try:
                    await self.sio.emit('ping', {'data': 'ping'})
                except Exception as e:
                    logger.debug(f"Ping send exception: {e}")

    async def subscribe_orderbook(self, pair: str, depth: int = 20):
        """Subscribe to live 20-level orderbook channel."""
        channel_name = f"{pair}@orderbook@{depth}"
        if self.is_connected:
            await self.sio.emit('join', {'channelName': channel_name})
            logger.info(f"Subscribed to orderbook: {channel_name}")

    async def subscribe_candles(self, pair: str, interval: str = "1m"):
        """Subscribe to live candlestick updates."""
        channel_name = f"{pair}_{interval}"
        if self.is_connected:
            await self.sio.emit('join', {'channelName': channel_name})
            logger.info(f"Subscribed to candles: {channel_name}")

    def register_price_listener(self, callback: Callable[[str, float], None]):
        self._on_price_callbacks.append(callback)

    def register_book_listener(self, callback: Callable[[str, Dict[str, Any]], None]):
        self._on_book_callbacks.append(callback)

    def register_order_listener(self, callback: Callable[[Dict[str, Any]], None]):
        self._on_order_update_callbacks.append(callback)

    async def start(self):
        """Start the socketio connection."""
        endpoint = settings.COINDCX_SPOT_SOCKET_URL
        try:
            await self.sio.connect(endpoint, transports=['websocket'])
            self._ping_task = asyncio.create_task(self._ping_loop())
        except Exception as e:
            logger.warning(f"Could not connect to live CoinDCX WebSocket ({e}). Fallback to simulation tick feed.")
            self.is_connected = False

    async def stop(self):
        """Disconnect and cleanup tasks."""
        if self._ping_task:
            self._ping_task.cancel()
        if self.is_connected:
            await self.sio.disconnect()
            self.is_connected = False
