"""
AlphaScalper - FastAPI Real-Time Server & WebSocket Hub
Provides:
1. Bidirectional Socket.IO telemetry stream (/socket.io)
2. Raw WebSocket streaming endpoint (/api/ws and /ws)
3. CoinDCX Inbound Webhook listener (/api/v1/webhook/coindcx)
4. Dual-Mode REST controllers: Default Autonomous AI vs Custom Quant Studio
5. Continuous high-frequency screening & Win-Win execution loop
"""

import asyncio
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import socketio

from config import settings
from coindcx_client import CoinDCXClient
from websocket_feed import CoinDCXWebSocketManager
from indicators import AlphaIndicatorsEngine
from ai_engine import AlphaAIEngine
from market_screener import MarketScreener
from win_win_strategy import WinWinExecutionEngine
from risk_manager import AlphaRiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AlphaScalper.Server")

# ==========================================
# CORE SERVICES INSTANTIATION
# ==========================================
client = CoinDCXClient()
ws_manager = CoinDCXWebSocketManager()
ai_engine = AlphaAIEngine()
screener = MarketScreener(client=client, ai_engine=ai_engine)
execution_engine = WinWinExecutionEngine(client=client)
risk_manager = AlphaRiskManager(initial_capital=settings.INITIAL_SIMULATION_BALANCE)

# Engine Execution State
engine_state = {
    "is_running": False,
    "mode": "DEFAULT",  # "DEFAULT" (Auto AI) or "CUSTOM"
    "universe_size": settings.UNIVERSE_MAX_ASSETS,
    "filter_count": settings.FILTER_TOP_CANDIDATES,
    "execution_count": settings.DEFAULT_ACTIVE_ORDERS,
    "leverage": settings.DEFAULT_LEVERAGE,
    "risk_per_trade_pct": settings.MAX_RISK_PER_TRADE_PERCENT,
    "selected_indicators": ["RSI", "VWAP", "Bollinger", "SuperTrend", "OBI", "MACD"],
    "price_action_rules": ["OrderBlocks", "FVG", "LiquiditySweeps"],
    "latency_history": []
}

# Socket.IO ASGI Server
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# Active Raw WebSockets (/api/ws, /ws)
active_raw_websockets: List[WebSocket] = []

# Background Tasks
background_tasks = []


async def broadcast_event(event_name: str, data: Any):
    """Broadcasts event to BOTH Socket.IO clients and Raw WebSocket connections."""
    # 1. Socket.IO broadcast
    try:
        await sio.emit(event_name, data)
    except Exception as e:
        logger.debug(f"Socket.IO emit error: {e}")

    # 2. Raw WebSockets broadcast
    if active_raw_websockets:
        message_str = json.dumps({"event": event_name, "data": data})
        disconnected = []
        for ws in list(active_raw_websockets):
            try:
                await ws.send_text(message_str)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            if ws in active_raw_websockets:
                active_raw_websockets.remove(ws)


async def scalper_orchestrator_loop():
    """Continuous high-frequency loop running market scanning & execution."""
    logger.info("AlphaScalper autonomous orchestrator loop started!")
    while True:
        try:
            if engine_state["is_running"] and not risk_manager.is_kill_switch_active:
                loop_start = time.perf_counter()

                # 1. Run Market Screener (500 -> 100 -> Top N)
                scan_res = await screener.scan_and_filter_market(
                    universe_size=engine_state["universe_size"],
                    filter_count=engine_state["filter_count"],
                    execution_count=engine_state["execution_count"]
                )

                # 2. Check for actionable scalps on top candidates
                for candidate in scan_res.get("ranked_targets", []):
                    # In default mode or custom mode, execute if high confidence
                    if candidate["confidence"] >= 72.0 or candidate["confidence"] <= 28.0:
                        symbol = candidate["symbol"]
                        
                        # Avoid duplicate trade on same symbol
                        already_open = any(t.symbol == symbol for t in execution_engine.active_trades.values())
                        if not already_open and len(execution_engine.active_trades) < engine_state["execution_count"]:
                            # Size position using Kelly Criterion
                            pos_size = risk_manager.calculate_kelly_position_size(candidate["confidence"])
                            can_trade, reason = risk_manager.can_open_new_trade(pos_size)
                            
                            if can_trade:
                                trade = await execution_engine.execute_win_win_entry(
                                    symbol=symbol,
                                    signal=candidate["signal"],
                                    entry_price=candidate["entry_price"],
                                    tp1_price=candidate["tp1_price"],
                                    tp2_price=candidate["tp2_price"],
                                    sl_price=candidate["sl_price"],
                                    breakeven_sl=candidate["breakeven_sl"],
                                    allocated_usdt=pos_size,
                                    leverage=engine_state["leverage"]
                                )
                                if trade:
                                    await broadcast_event("execution_event", {
                                        "type": "ENTRY",
                                        "trade_id": trade.trade_id,
                                        "symbol": symbol,
                                        "side": trade.side,
                                        "price": trade.entry_price,
                                        "size_usdt": pos_size,
                                        "message": f"Executed {trade.side} Scalp on {symbol} @ ${trade.entry_price}"
                                    })
                            else:
                                # Throttle Balance Guard notification to prevent spamming
                                if "BALANCE GUARD" in reason:
                                    now_ts = time.time()
                                    if now_ts - engine_state.get("_last_guard_broadcast", 0.0) > 20.0:
                                        engine_state["_last_guard_broadcast"] = now_ts
                                        await broadcast_event("execution_event", {
                                            "type": "BALANCE_GUARD_HOLD",
                                            "symbol": symbol,
                                            "message": f"[BALANCE GUARD] AI Scalp signal ({candidate['signal']}) spotted on {symbol}, but order held. Balance (${risk_manager.current_capital:.4f} USDT) < Min $6.00 USDT."
                                        })

                # 3. Simulate or update price movements on active trades
                for symbol, trade in list(execution_engine.active_trades.items()):
                    # Get latest price or drift slightly in paper mode
                    curr_p = screener._candle_cache.get(trade.symbol, {}).get("close", [trade.entry_price])[-1]
                    events = execution_engine.update_ticks(trade.symbol, float(curr_p), obi_10=0.2)
                    for ev in events:
                        # Ensure real position is flattened on CoinDCX if in live mode
                        if not client.is_paper and ev.get("event") in ("WIN_WIN_TP2_MAX_PROFIT", "STOP_EXECUTED", "MICRO_TIMEOUT_SCRATCH_EXIT"):
                            try:
                                await client.exit_futures_position(trade.symbol)
                            except Exception as exit_err:
                                logger.error(f"Error executing live position exit on CoinDCX for {trade.symbol}: {exit_err}")
                        risk_manager.update_daily_pnl(ev.get("realized_pnl", 0.0))
                        await broadcast_event("execution_event", ev)

                # 4. Broadcast real-time telemetry to Next.js frontend
                loop_latency_ms = round((time.perf_counter() - loop_start) * 1000.0, 2)
                engine_state["latency_history"].append(loop_latency_ms)
                if len(engine_state["latency_history"]) > 30:
                    engine_state["latency_history"].pop(0)

                stats = execution_engine.get_summary_stats()
                telemetry = {
                    "timestamp": int(time.time() * 1000),
                    "is_running": engine_state["is_running"],
                    "mode": engine_state["mode"],
                    "trading_mode": settings.TRADING_MODE,
                    "latency_ms": loop_latency_ms,
                    "avg_latency_ms": round(sum(engine_state["latency_history"]) / len(engine_state["latency_history"]), 2),
                    "daily_pnl": round(risk_manager.daily_pnl, 2),
                    "current_capital": round(risk_manager.current_capital, 4),
                    "is_balance_sufficient": risk_manager.is_balance_sufficient,
                    "min_required_margin": risk_manager.min_order_notional,
                    "win_rate_pct": stats["win_rate_pct"],
                    "profit_factor": stats["profit_factor"],
                    "total_trades": stats["total_trades"],
                    "active_trades_count": stats["active_trades_count"],
                    "risk_free_count": stats["risk_free_active_count"],
                    "kill_switch_active": risk_manager.is_kill_switch_active,
                    "top_ranked": scan_res.get("ranked_targets", [])[:5],
                    "filtered_100_summary": [
                        {
                            "symbol": c["symbol"],
                            "price": c["price"],
                            "confidence": c["confidence"],
                            "signal": c["signal"],
                            "volume_24h": c["volume_24h"],
                            "spread_pct": c["spread_pct"],
                            "obi_10": c["obi_10"],
                            "regime": c["regime"]
                        }
                        for c in scan_res.get("top_100_filtered", [])[:20]
                    ]
                }
                await broadcast_event("telemetry_update", telemetry)

            await asyncio.sleep(1.5)  # Fast 1.5s scan interval
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in scalper loop: {e}", exc_info=True)
            await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting AlphaScalper Services...")
    
    # 1. Verify CoinDCX External API Connectivity and Live Account Balances
    verification = await client.verify_api_connectivity(force_refresh=True)
    logger.info("==================================================")
    logger.info("     ALPHASCALPER COINDCX API VERIFICATION        ")
    logger.info("==================================================")
    logger.info(f"Trading Mode:             {verification['trading_mode']}")
    logger.info(f"Public REST Market Data:  {verification['public_api']}")
    logger.info(f"Authenticated Auth API:   {verification['auth_api']}")
    logger.info(f"Active Futures Contracts: {verification.get('active_instruments_count', 0)}")
    logger.info(f"Live Balances:            {verification.get('balances', {})}")
    logger.info(f"Total Usable USDT Margin: ${verification.get('total_usdt_balance', 0.0):.4f} USDT")
    logger.info(f"Sufficient for Live Exec: {verification.get('is_balance_sufficient', False)} (Min required: ${verification.get('min_required_usdt', 6.0):.2f})")
    logger.info(f"Strategy Status:          {verification.get('strategy_status', 'UNKNOWN')}")
    logger.info(f"Strategy Guidance:        {verification.get('strategy_message', '')}")
    logger.info("==================================================")

    # 2. Synchronize Risk Manager with Real Account Balance
    risk_manager.sync_live_balance(
        usable_balance_usdt=verification.get("total_usdt_balance", 0.0),
        is_live=(verification.get("trading_mode") == "LIVE")
    )

    # 3. Initialize Market Universe and Feeds
    await screener.initialize_universe()
    asyncio.create_task(ws_manager.start())
    task = asyncio.create_task(scalper_orchestrator_loop())
    background_tasks.append(task)
    yield
    # Shutdown
    logger.info("Stopping AlphaScalper Services...")
    for t in background_tasks:
        t.cancel()
    await ws_manager.stop()
    await client.close()


fastapi_app = FastAPI(
    title=f"{settings.BRAND_NAME} API",
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# RAW WEBSOCKET ENDPOINTS (/api/ws & /ws)
# ==========================================

@fastapi_app.websocket("/api/ws")
@fastapi_app.websocket("/ws")
async def raw_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_raw_websockets.append(websocket)
    logger.info(f"Raw WebSocket client connected on {websocket.url.path}! Total active clients: {len(active_raw_websockets)}")
    try:
        # Send initial state immediately
        stats = execution_engine.get_summary_stats()
        await websocket.send_text(json.dumps({
            "event": "initial_state",
            "data": {
                "is_running": engine_state["is_running"],
                "mode": engine_state["mode"],
                "config": engine_state,
                "stats": stats
            }
        }))
        while True:
            # Handle incoming client messages / ping-pong
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping" or msg.get("event") == "ping":
                    await websocket.send_text(json.dumps({"event": "pong", "data": {"ts": int(time.time() * 1000)}}))
            except Exception:
                pass
    except WebSocketDisconnect:
        logger.info("Raw WebSocket client disconnected.")
    except Exception as e:
        logger.debug(f"Raw WebSocket session ended: {e}")
    finally:
        if websocket in active_raw_websockets:
            active_raw_websockets.remove(websocket)


# ==========================================
# SOCKET.IO EVENT HANDLERS
# ==========================================

@sio.event
async def connect(sid, environ):
    logger.info(f"Socket.IO Frontend Client connected: {sid}")
    stats = execution_engine.get_summary_stats()
    await sio.emit("initial_state", {
        "is_running": engine_state["is_running"],
        "mode": engine_state["mode"],
        "config": engine_state,
        "stats": stats
    }, room=sid)


@sio.event
async def disconnect(sid):
    logger.info(f"Socket.IO Frontend Client disconnected: {sid}")


# ==========================================
# REST API CONTROLLERS
# ==========================================

class StrategyConfigRequest(BaseModel):
    mode: str = "DEFAULT"
    universe_size: int = 500
    filter_count: int = 100
    execution_count: int = 10
    leverage: float = 10.0
    risk_per_trade_pct: float = 1.0
    selected_indicators: Optional[List[str]] = None
    price_action_rules: Optional[List[str]] = None


@fastapi_app.get("/api/v1/health")
async def health():
    return {
        "brand": settings.BRAND_NAME,
        "version": settings.VERSION,
        "status": "HEALTHY",
        "trading_mode": settings.TRADING_MODE,
        "is_running": engine_state["is_running"],
        "timestamp": int(time.time() * 1000)
    }


@fastapi_app.get("/api/v1/coindcx/verify")
async def verify_coindcx_api_endpoint():
    """
    Comprehensive diagnostic endpoint verifying:
    1. CoinDCX Public Market Data API connectivity
    2. CoinDCX Authenticated Private API credentials
    3. User live wallet balances (USDT, INR, Crypto)
    4. Futures margin sufficiency (Min $6.00 USDT)
    5. Balance-aware execution strategy status
    """
    verification = await client.verify_api_connectivity(force_refresh=True)
    risk_manager.sync_live_balance(
        usable_balance_usdt=verification.get("total_usdt_balance", 0.0),
        is_live=(verification.get("trading_mode") == "LIVE")
    )
    return verification


@fastapi_app.post("/api/v1/coindcx/sync_balance")
async def sync_coindcx_balance_endpoint():
    """
    On-demand synchronization of CoinDCX wallet balance into risk manager.
    """
    verification = await client.verify_api_connectivity(force_refresh=True)
    risk_manager.sync_live_balance(
        usable_balance_usdt=verification.get("total_usdt_balance", 0.0),
        is_live=(verification.get("trading_mode") == "LIVE")
    )
    return {
        "status": "success",
        "total_usdt_balance": verification.get("total_usdt_balance", 0.0),
        "is_balance_sufficient": verification.get("is_balance_sufficient", False),
        "strategy_status": verification.get("strategy_status", "UNKNOWN"),
        "balances": verification.get("balances", {})
    }


@fastapi_app.get("/api/exchange/positions")
async def get_exchange_positions_alias():
    """
    Compatibility alias for exchange positions query.
    """
    return await get_positions()


@fastapi_app.post("/api/v1/engine/start")
async def start_engine():
    risk_manager.is_kill_switch_active = False
    engine_state["is_running"] = True
    logger.info("AlphaScalper Execution Engine STARTED!")
    return {"status": "success", "is_running": True}


@fastapi_app.post("/api/v1/engine/stop")
async def stop_engine():
    engine_state["is_running"] = False
    logger.info("AlphaScalper Execution Engine STOPPED!")
    return {"status": "success", "is_running": False}


@fastapi_app.post("/api/v1/engine/kill_switch")
async def trigger_kill_switch():
    engine_state["is_running"] = False
    risk_manager.trigger_emergency_kill_switch()
    # Flatten active trades
    for trade_id, trade in list(execution_engine.active_trades.items()):
        await client.exit_futures_position(trade.symbol)
        trade.remaining_qty = 0.0
        execution_engine.closed_trades.append(trade)
    execution_engine.active_trades.clear()
    
    await broadcast_event("execution_event", {
        "type": "KILL_SWITCH",
        "message": "EMERGENCY KILL-SWITCH TRIGGERED. ALL POSITIONS CLOSED & TRADING HALTED!"
    })
    return {"status": "success", "kill_switch_active": True, "message": "All positions flattened."}


@fastapi_app.post("/api/v1/strategy/configure")
async def configure_strategy(req: StrategyConfigRequest):
    engine_state["mode"] = req.mode.upper()
    engine_state["universe_size"] = req.universe_size
    engine_state["filter_count"] = req.filter_count
    engine_state["execution_count"] = req.execution_count
    engine_state["leverage"] = req.leverage
    engine_state["risk_per_trade_pct"] = req.risk_per_trade_pct
    if req.selected_indicators:
        engine_state["selected_indicators"] = req.selected_indicators
    if req.price_action_rules:
        engine_state["price_action_rules"] = req.price_action_rules

    logger.info(f"Updated Strategy Configuration: Mode={engine_state['mode']}, Universe={req.universe_size}, Filter={req.filter_count}, Exec={req.execution_count}")
    return {"status": "success", "config": engine_state}


@fastapi_app.get("/api/v1/markets/screener")
async def get_screener_results():
    scan_res = await screener.scan_and_filter_market(
        universe_size=engine_state["universe_size"],
        filter_count=engine_state["filter_count"],
        execution_count=engine_state["execution_count"]
    )
    return scan_res


@fastapi_app.get("/api/v1/markets/orderbook")
async def get_live_orderbook(symbol: str = "B-BTC_USDT"):
    """
    Returns live L2 Order Book depth directly from CoinDCX for a given symbol.
    """
    ob = await client.get_futures_orderbook(symbol, depth=20)
    return ob


@fastapi_app.get("/api/v1/positions")
async def get_positions():
    trades = [
        {
            "trade_id": t.trade_id,
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": t.entry_price,
            "tp1_price": t.tp1_price,
            "tp2_price": t.tp2_price,
            "sl_price": t.sl_price,
            "breakeven_sl": t.breakeven_sl,
            "is_risk_free": t.is_risk_free,
            "remaining_qty": round(t.remaining_qty, 4),
            "unrealized_pnl": round(t.unrealized_pnl, 2),
            "realized_pnl": round(t.realized_pnl, 2),
            "state": t.current_state,
            "elapsed_seconds": int(time.time() - (t.entry_time or time.time()))
        }
        for t in execution_engine.active_trades.values()
    ]
    return {"active_trades": trades, "count": len(trades)}


@fastapi_app.post("/api/v1/positions/exit")
async def exit_single_position(payload: Dict[str, str]):
    trade_id = payload.get("trade_id")
    if trade_id in execution_engine.active_trades:
        trade = execution_engine.active_trades.pop(trade_id)
        await client.exit_futures_position(trade.symbol)
        trade.remaining_qty = 0.0
        execution_engine.closed_trades.append(trade)
        return {"status": "success", "trade_id": trade_id}
    return {"status": "not_found"}


@fastapi_app.get("/api/v1/analytics/stats")
async def get_analytics():
    stats = execution_engine.get_summary_stats()
    stats["capital"] = round(risk_manager.current_capital, 2)
    stats["daily_pnl"] = round(risk_manager.daily_pnl, 2)
    stats["latency_history"] = engine_state["latency_history"]
    return stats


# ==========================================
# COINDCX INBOUND WEBHOOK ENDPOINT
# ==========================================

@fastapi_app.post("/api/v1/webhook/coindcx")
async def coindcx_webhook(request: Request):
    """
    Receives inbound order fill notifications and liquidation alerts from CoinDCX.
    """
    try:
        body = await request.json()
        logger.info(f"Inbound CoinDCX Webhook Event: {body}")
        
        event_type = body.get("event") or body.get("type", "UNKNOWN")
        order_data = body.get("data", body)
        
        # Broadcast fill to frontend
        await broadcast_event("webhook_event", {
            "type": event_type,
            "data": order_data,
            "timestamp": int(time.time() * 1000)
        })
        
        return {"status": "received", "event": event_type}
    except Exception as e:
        logger.error(f"Error processing CoinDCX webhook: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Mount Socket.IO to FastAPI app
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=settings.HOST, port=settings.PORT, reload=True)
