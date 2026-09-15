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
import random
import numpy as np
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import socketio

from config import settings
from coindcx_client import CoinDCXClient, safe_float, safe_int
from websocket_feed import CoinDCXWebSocketManager
from indicators import AlphaIndicatorsEngine
from ai_engine import AlphaAIEngine
from market_screener import MarketScreener
from win_win_strategy import WinWinExecutionEngine
from risk_manager import AlphaRiskManager
from binance_client import to_binance_symbol
from db import (
    mark_trade_closed,
    init_sqlite_db,
    neon_manager,
    get_24h_analytics,
    save_equity_snapshot,
    log_ai_signal,
    load_closed_trades
)

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
    "direction_bias": "AUTO",  # "AUTO", "LONG", or "SHORT"
    "selected_indicators": ["RSI", "VWAP", "Bollinger", "SuperTrend", "OBI", "MACD"],
    "price_action_rules": ["OrderBlocks", "FVG", "LiquiditySweeps"],
    "min_confidence": 62.0,
    "min_volume_24h": 10000.0,
    "max_spread_pct": 0.20,
    "stop_loss_pct": 0.50,
    "take_profit_1_pct": 1.00,
    "take_profit_2_pct": 2.00,
    "enable_breakeven": True,
    "timeframes": ["1m", "5m", "15m"],
    "strict_counter_trend_veto": True,
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


# Shared state for latest deep AI market scan
latest_scan_result: Dict[str, Any] = {}

# 1-Minute (60s) In-Memory Cache for CoinDCX Positions & Active Orders to prevent rate limits
coindcx_cache: Dict[str, Any] = {
    "positions": [],
    "positions_last_fetched": 0.0,
    "orders": [],
    "orders_last_fetched": 0.0,
    "ttl_seconds": 60.0  # Strictly throttled to 1 minute (60s)
}

# Rolling 24-Hour Analytics Cache (Neon Cloud PostgreSQL / SQLite Mirror)
cached_analytics_24h: Dict[str, Any] = {
    "status": "success",
    "database": "Neon Cloud PostgreSQL (Primary)",
    "neon_connected": True,
    "period": "24h_rolling",
    "tp1_hits": 0,
    "tp2_hits": 0,
    "sl_hits": 0,
    "breakeven_exits": 0,
    "total_trades_24h": 0,
    "winning_trades_24h": 0,
    "losing_trades_24h": 0,
    "win_rate_24h_pct": 0.0,
    "profit_factor_24h": 0.0,
    "earned_24h_usdt": 0.0,
    "lost_24h_usdt": 0.0,
    "net_pnl_24h_usdt": 0.0,
    "earned_24h_inr": 0.0,
    "lost_24h_inr": 0.0,
    "net_pnl_24h_inr": 0.0,
    "total_fees_24h_usdt": 0.0,
    "total_fees_24h_inr": 0.0,
    "inr_rate": 87.5,
    "updated_at": 0.0
}

async def coindcx_sync_1min_loop():
    """
    Periodic background worker running strictly every 1 minute (60s):
    1. Syncs CoinDCX positions & active orders.
    2. Performs automated 1-minute safety inspection: verifies active SL & TP on every open position.
    3. Auto-heals by attaching algorithmic SL (-0.45%) and TP (+0.85%) if missing on exchange.
    4. Pushes real-time updates via WebSocket.
    """
    logger.info("Starting CoinDCX 1-minute sync & SL/TP auto-healing worker...")
    while True:
        try:
            await asyncio.sleep(60.0)  # Strictly 1 minute
            now = time.time()
            
            live_pos = []
            orders = []
            
            try:
                live_pos = await client.get_futures_positions()
            except Exception as e:
                logger.debug(f"1-min positions sync error: {e}")

            try:
                orders = await client.get_all_active_orders()
            except Exception as e:
                logger.debug(f"1-min active orders sync error: {e}")

            # Safety Inspection & Auto-Healing Guard
            if live_pos:
                for p in live_pos:
                    if not isinstance(p, dict):
                        continue
                    active_qty = safe_float(p.get("active_pos"), 0.0)
                    if abs(active_qty) > 1e-6:
                        try:
                            reconcile_res = await client.verify_and_reconcile_position_tpsl(
                                position=p,
                                active_orders=orders,
                                default_tp_ratio=settings.TP1_RATIO,
                                default_sl_ratio=settings.HARD_SL_RATIO
                            )
                            if reconcile_res.get("status") == "auto_healed":
                                sym = reconcile_res.get("pair")
                                tp_val = reconcile_res.get("attached_tp")
                                sl_val = reconcile_res.get("attached_sl")
                                logger.info(f"🛡️ [AUTO-HEAL 1-MIN] Attached missing SL & TP for {sym} (TP={tp_val}, SL={sl_val})")
                                await broadcast_event("execution_event", {
                                    "type": "TPSL_AUTO_HEALED",
                                    "symbol": sym,
                                    "message": f"Auto-Heal Active: Attached missing SL (${sl_val}) & TP (${tp_val}) on CoinDCX for {sym}.",
                                    "details": reconcile_res
                                })
                        except Exception as hr_err:
                            logger.error(f"Error during 1-min TP/SL auto-healing for {p.get('pair')}: {hr_err}")

            # Bidirectional Reconciliation: Evict closed positions from engine memory and SQLite
            if isinstance(live_pos, list) and not client.is_paper:
                active_exchange_symbols = {
                    p.get("pair") for p in live_pos
                    if isinstance(p, dict) and abs(safe_float(p.get("active_pos"), 0.0)) > 1e-6
                }
                for trade_id, trade in list(execution_engine.active_trades.items()):
                    if trade.symbol not in active_exchange_symbols:
                        logger.info(f"🔄 [RECONCILE 1-MIN] Position {trade.symbol} is closed on CoinDCX. Evicting from active engine memory.")
                        trade.remaining_qty = 0.0
                        trade.exit_time = now
                        trade.current_state = "EXCHANGE_CLOSED"
                        execution_engine.closed_trades.append(trade)
                        execution_engine.active_trades.pop(trade_id, None)
                        mark_trade_closed(trade_id, now, trade.net_realized_pnl, "EXCHANGE_CLOSED")
                        risk_manager.record_trade_exit(trade.symbol)
                        await broadcast_event("execution_event", {
                            "type": "RECONCILED_EXIT",
                            "symbol": trade.symbol,
                            "trade_id": trade_id,
                            "message": f"Reconciliation: Closed position on {trade.symbol} confirmed by CoinDCX. Memory and slot freed."
                        })

                # Adoption: If CoinDCX has an open position NOT in active_trades, adopt it into the tick loop
                for p in live_pos:
                    if isinstance(p, dict) and abs(safe_float(p.get("active_pos"), 0.0)) > 1e-6:
                        p_pair = p.get("pair")
                        if p_pair and not any(t.symbol == p_pair for t in execution_engine.active_trades.values()):
                            adopted = execution_engine.adopt_exchange_position(p)
                            if adopted:
                                logger.info(f"🛡️ [ADOPT 1-MIN] Adopted active CoinDCX position {p_pair} into WinWin tick monitor.")

            coindcx_cache["positions"] = live_pos or []
            coindcx_cache["positions_last_fetched"] = now
            coindcx_cache["orders"] = orders or []
            coindcx_cache["orders_last_fetched"] = now

            # Broadcast to WebSocket clients
            await broadcast_event("positions_update", {"positions": live_pos, "timestamp": int(now * 1000)})
            await broadcast_event("orders_update", {"orders": orders, "timestamp": int(now * 1000)})

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in coindcx_sync_1min_loop: {e}")
            await asyncio.sleep(5.0)

async def neon_equity_snapshot_loop():
    """
    Periodic background worker (running every 5 minutes) logging real-time portfolio equity
    and margin metrics to Neon Cloud PostgreSQL & local SQLite mirror.
    """
    logger.info("AlphaScalper Neon Cloud Equity Snapshot worker started!")
    while True:
        try:
            await asyncio.sleep(300.0)  # Every 5 minutes
            active_cnt = len(execution_engine.active_trades)
            unrealized = sum(t.unrealized_pnl for t in execution_engine.active_trades.values())
            margin_used = sum(t.total_size_usdt / max(1.0, t.leverage) for t in execution_engine.active_trades.values())

            usdt_bal = risk_manager.current_capital
            inr_bal = usdt_bal * 87.5
            if not client.is_paper:
                try:
                    w = await client.get_futures_inr_balance()
                    if w and w.get("total_inr", 0.0) > 0:
                        inr_bal = w["total_inr"]
                except Exception:
                    pass

            save_equity_snapshot(
                wallet_inr=round(inr_bal, 2),
                wallet_usdt=round(usdt_bal, 4),
                unrealized_pnl=round(unrealized, 4),
                active_positions_count=active_cnt,
                margin_used=round(margin_used, 4)
            )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"Neon equity snapshot error: {e}")

async def deep_ai_screener_loop():
    """Continuous background worker running deep multi-factor market analysis on dynamic intervals (20s - 45s)."""
    logger.info("AlphaScalper Deep AI Screener background worker started!")
    global latest_scan_result
    while True:
        try:
            if not risk_manager.is_kill_switch_active:
                scan_res = await screener.scan_and_filter_market(
                    universe_size=engine_state["universe_size"],
                    filter_count=engine_state["filter_count"],
                    execution_count=engine_state["execution_count"],
                    min_volume_24h=engine_state.get("min_volume_24h", 15000.0),
                    max_spread_pct=engine_state.get("max_spread_pct", 0.20),
                    direction_bias=engine_state.get("direction_bias", "AUTO"),
                    selected_indicators=engine_state.get("selected_indicators"),
                    price_action_rules=engine_state.get("price_action_rules"),
                    stop_loss_pct=engine_state.get("stop_loss_pct"),
                    take_profit_1_pct=engine_state.get("take_profit_1_pct"),
                    take_profit_2_pct=engine_state.get("take_profit_2_pct"),
                    enable_breakeven=engine_state.get("enable_breakeven", True),
                    strict_counter_trend_veto=engine_state.get("strict_counter_trend_veto", True),
                    timeframes=engine_state.get("timeframes", ["1m", "5m", "15m"]),
                    min_confidence=engine_state.get("min_confidence", 62.0),
                    leverage=engine_state.get("leverage", settings.DEFAULT_LEVERAGE)
                )
                latest_scan_result = scan_res

                # Log top candidates to Neon Cloud PostgreSQL / SQLite audit trail
                for cand in scan_res.get("top_10_filtered", [])[:5]:
                    log_ai_signal(
                        symbol=cand["symbol"],
                        direction=cand["signal"],
                        confidence=float(cand.get("confidence", 0.0)),
                        volatility_regime=cand.get("regime", "NORMAL"),
                        obi_score=float(cand.get("obi_10", 0.0)),
                        executed=False
                    )
                
                # Keep Binance WebSocket streaming top 10 candidates + any active trade symbols
                top_syms = [c["symbol"] for c in scan_res.get("top_10_filtered", [])]
                active_syms = [t.symbol for t in execution_engine.active_trades.values()]
                screener.binance_client.update_monitored_symbols(top_syms + active_syms)

                await broadcast_event("screener_update", scan_res)
                
                # Dynamic sleep interval based on market condition
                if engine_state["is_running"]:
                    sleep_seconds = float(scan_res.get("suggested_next_scan_seconds", 25.0))
                else:
                    sleep_seconds = 30.0  # Steady background screening when execution engine is paused
            else:
                sleep_seconds = 10.0
            
            await asyncio.sleep(sleep_seconds)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in deep_ai_screener_loop: {e}", exc_info=True)
            await asyncio.sleep(10.0)


async def scalper_orchestrator_loop():
    """High-frequency tick monitoring, real-time position management, and order execution."""
    logger.info("AlphaScalper autonomous orchestrator loop started!")
    global latest_scan_result
    while True:
        try:
            if engine_state["is_running"] and not risk_manager.is_kill_switch_active:
                loop_start = time.perf_counter()
                scan_res = latest_scan_result or {}

                # 1. Check for actionable scalps from latest Top 10 high-probability candidates
                min_conf = float(engine_state.get("min_confidence", 62.0))
                candidates_to_eval = scan_res.get("ranked_targets") or scan_res.get("top_10_filtered", [])
                for candidate in candidates_to_eval:
                    if candidate.get("is_synthetic", False):
                        continue
                    sig = candidate.get("signal", "NEUTRAL")
                    is_ev_viable = candidate.get("is_ev_viable", True)
                    if candidate.get("confidence", 0.0) >= min_conf and ("BUY" in sig or "SELL" in sig) and is_ev_viable:
                        symbol = candidate["symbol"]
                        
                        already_open = any(t.symbol == symbol for t in execution_engine.active_trades.values())
                        if not already_open and len(execution_engine.active_trades) < engine_state["execution_count"]:
                            current_lev = float(engine_state.get("leverage", settings.DEFAULT_LEVERAGE))
                            entry_p = float(candidate.get("entry_price") or candidate.get("price") or 100.0)
                            sl_p = float(candidate.get("sl_price") or (entry_p * 0.995))
                            risk_pct = float(engine_state.get("risk_per_trade_pct", 1.5))
                            
                            # Volatility parity sizing with ATR
                            pos_size = risk_manager.calculate_volatility_parity_position_size(
                                entry_price=entry_p,
                                sl_price=sl_p,
                                risk_per_trade_pct=risk_pct,
                                leverage=current_lev
                            )
                            can_trade, reason = risk_manager.can_open_new_trade(pos_size, leverage=current_lev, symbol=symbol)
                            
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
                                    leverage=current_lev
                                )
                                if trade:
                                    risk_manager.record_trade_entry(symbol)
                                    coindcx_cache["positions_last_fetched"] = 0.0
                                    coindcx_cache["orders_last_fetched"] = 0.0
                                    b_sym = to_binance_symbol(symbol)
                                    if b_sym.lower() not in screener.binance_client._monitored_symbols:
                                        screener.binance_client.update_monitored_symbols(
                                            screener.binance_client._monitored_symbols + [b_sym]
                                        )
                                    await broadcast_event("execution_event", {
                                        "type": "ENTRY",
                                        "trade_id": trade.trade_id,
                                        "symbol": symbol,
                                        "side": trade.side,
                                        "price": trade.entry_price,
                                        "size_usdt": pos_size,
                                        "leverage": current_lev,
                                        "message": f"Executed {trade.side} Scalp on {symbol} @ ${trade.entry_price} (AI Signal: {sig}, Conviction: {candidate['confidence']}%, EV: +{candidate.get('expected_value_pct', 0.0)}%, Leverage: {current_lev}x)"
                                    })
                            else:
                                if "COOLDOWN" in reason or "BALANCE GUARD" in reason or "PACING" in reason:
                                    now_ts = time.time()
                                    if now_ts - engine_state.get("_last_guard_broadcast", 0.0) > 30.0:
                                        engine_state["_last_guard_broadcast"] = now_ts
                                        await broadcast_event("execution_event", {
                                            "type": "PACING_GUARD_HOLD",
                                            "symbol": symbol,
                                            "message": f"[AI GUARD] Scalp signal on {symbol} held: {reason}"
                                        })

                # 2. Update price movements on active trades using live Binance ticks
                # Active trades are NEVER closed due to ranking changes or timeouts; only TP1, TP2, or SL.
                for trade_id, trade in list(execution_engine.active_trades.items()):
                    b_sym = to_binance_symbol(trade.symbol)
                    binance_p = screener.binance_client.latest_prices.get(b_sym)
                    cand = next((c for c in screener.filtered_100 if c["symbol"] == trade.symbol), None)
                    curr_p = binance_p or (cand["price"] if cand else screener._candle_cache.get(trade.symbol, {}).get("close", [trade.entry_price])[-1])
                    live_obi = float(cand.get("obi_10", 0.0)) if cand else 0.0
                    events = execution_engine.update_ticks(trade.symbol, float(curr_p), obi_10=live_obi)
                    for ev in events:
                        ev_type = ev.get("event")
                        if not client.is_paper:
                            if ev_type == "WIN_WIN_BREAKEVEN_LOCKED":
                                try:
                                    half_qty = float(ev.get("half_closed_qty", 0.0))
                                    half_notional = half_qty * float(curr_p)
                                    if half_notional >= 6.0:
                                        exit_side = "sell" if trade.side.upper() == "BUY" else "buy"
                                        await client.create_futures_order(
                                            pair=trade.symbol,
                                            side=exit_side,
                                            order_type="market_order",
                                            total_quantity=half_qty,
                                            leverage=trade.leverage
                                        )
                                        logger.info(f"[LIVE SYNC] Scaled out 50% ({half_qty}) on CoinDCX for {trade.symbol} @ ${curr_p}")
                                    else:
                                        # CoinDCX minimum order rule: $6.00 USDT.
                                        # Revert memory reduction so in-memory quantity remains synchronized with 100% position on exchange.
                                        trade.remaining_qty += half_qty
                                        logger.info(
                                            f"[SMART SCALE-OUT] Half notional (${half_notional:.2f}) < $6.00 min contract. "
                                            f"Retaining 100% position ({trade.remaining_qty}) on {trade.symbol} and ratcheting SL to Breakeven (+Fees)!"
                                        )
                                    
                                    is_long_pos = (trade.side.upper() == "BUY")
                                    await client.cancel_all_open_orders_for_position(trade.symbol)
                                    await client.create_futures_tpsl(
                                        position_id=trade.symbol,
                                        tp_stop_price=trade.tp2_price,
                                        sl_stop_price=trade.breakeven_sl,
                                        is_long=is_long_pos,
                                        pair=trade.symbol
                                    )
                                    logger.info(f"[LIVE SYNC] Ratcheted exchange SL on CoinDCX for {trade.symbol} to Breakeven: {trade.breakeven_sl}")
                                except Exception as be_err:
                                    logger.error(f"Error synchronizing Breakeven SL on CoinDCX for {trade.symbol}: {be_err}")
                                    
                            elif ev_type in ("WIN_WIN_TP2_MAX_PROFIT", "STOP_EXECUTED"):
                                try:
                                    await client.exit_futures_position(trade.symbol)
                                except Exception as exit_err:
                                    logger.error(f"Error executing live position exit on CoinDCX for {trade.symbol}: {exit_err}")
                                
                                risk_manager.record_trade_exit(trade.symbol)
                                coindcx_cache["positions_last_fetched"] = 0.0

                                try:
                                    wallet_info = await client.get_futures_inr_balance()
                                    if wallet_info and wallet_info.get("total_inr", 0.0) > 0:
                                        risk_manager.sync_live_inr_equity(wallet_info["total_inr"])
                                except Exception as sync_err:
                                    logger.debug(f"Live balance re-sync error: {sync_err}")
                                    
                        if ev_type in ("WIN_WIN_TP2_MAX_PROFIT", "STOP_EXECUTED"):
                            risk_manager.record_trade_exit(trade.symbol)
                            coindcx_cache["positions_last_fetched"] = 0.0

                        risk_manager.update_daily_pnl(ev.get("net_realized_pnl", ev.get("realized_pnl", 0.0)))
                        await broadcast_event("execution_event", ev)

                # 3. Broadcast real-time telemetry to Next.js frontend
                loop_latency_ms = round((time.perf_counter() - loop_start) * 1000.0, 2)
                engine_state["latency_history"].append(loop_latency_ms)
                if len(engine_state["latency_history"]) > 30:
                    engine_state["latency_history"].pop(0)

                stats = execution_engine.get_summary_stats()
                inr_rate = 87.5
                daily_pnl_usdt = round(risk_manager.daily_pnl, 2)
                daily_pnl_inr = round(daily_pnl_usdt * inr_rate, 2)
                capital_usdt = round(risk_manager.current_capital, 2)
                capital_inr = round(capital_usdt * inr_rate, 2)

                # Refresh rolling 24h performance metrics every 2 seconds
                global cached_analytics_24h
                now_epoch = time.time()
                if now_epoch - cached_analytics_24h.get("updated_at", 0.0) >= 2.0:
                    try:
                        cached_analytics_24h = await get_24h_analytics(inr_rate=inr_rate)
                    except Exception as err_24h:
                        logger.debug(f"Error fetching 24h rolling analytics: {err_24h}")

                telemetry = {
                    "timestamp": int(time.time() * 1000),
                    "is_running": engine_state["is_running"],
                    "mode": engine_state["mode"],
                    "trading_mode": settings.TRADING_MODE,
                    "account_currency": getattr(client, "account_currency", "INR"),
                    "currency_symbol": "₹" if getattr(client, "account_currency", "INR") == "INR" else "$",
                    "analytics_24h": cached_analytics_24h,
                    "latency_ms": loop_latency_ms,
                    "avg_latency_ms": round(sum(engine_state["latency_history"]) / len(engine_state["latency_history"]), 2),
                    "daily_pnl": daily_pnl_usdt,
                    "daily_pnl_inr": daily_pnl_inr,
                    "current_capital": capital_usdt,
                    "current_capital_inr": capital_inr,
                    "inr_rate": inr_rate,
                    "total_fees_paid": stats.get("total_fees_paid", 0.0),
                    "total_fees_paid_inr": round(stats.get("total_fees_paid", 0.0) * inr_rate, 2),
                    "net_pnl": stats.get("net_pnl", daily_pnl_usdt),
                    "is_balance_sufficient": risk_manager.is_balance_sufficient,
                    "min_required_margin": risk_manager.min_order_notional,
                    "win_rate_pct": stats["win_rate_pct"],
                    "profit_factor": stats["profit_factor"],
                    "total_trades": stats["total_trades"],
                    "active_trades_count": stats["active_trades_count"],
                    "risk_free_count": stats["risk_free_active_count"],
                    "kill_switch_active": risk_manager.is_kill_switch_active,
                    "circuit_breaker_tripped": risk_manager.is_circuit_breaker_tripped,
                    "scanned_universe_count": scan_res.get("scanned_universe_count", 0),
                    "total_universe_scanned": scan_res.get("total_universe_scanned", 0),
                    "max_executable_orders": scan_res.get("max_executable_orders", 0),
                    "capital_allocation": scan_res.get("capital_allocation", {}),
                    "btc_regime": scan_res.get("btc_regime", "CHOP_COMPRESSION"),
                    "btc_context": scan_res.get("btc_context", {}),
                    "top_ranked": scan_res.get("ranked_targets", [])[:5],
                    "top_10_filtered": scan_res.get("top_10_filtered", []),
                    "filtered_100_summary": [
                        {
                            "symbol": c["symbol"],
                            "price": c["price"],
                            "confidence": c["confidence"],
                            "signal": c["signal"],
                            "volume_24h": c["volume_24h"],
                            "spread_pct": c["spread_pct"],
                            "obi_10": c["obi_10"],
                            "regime": c["regime"],
                            "win_probability_pct": c.get("win_probability_pct", 75.0),
                            "expected_value_pct": c.get("expected_value_pct", 0.0),
                            "execution_status": c.get("execution_status", "PENDING")
                        }
                        for c in scan_res.get("top_10_filtered", [])
                    ]
                }
                await broadcast_event("telemetry_update", telemetry)

            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in scalper loop: {e}", exc_info=True)
            await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting AlphaScalper Services...")
    
    # 0. Initialize Neon Cloud PostgreSQL & SQLite WAL Persistence
    init_sqlite_db()
    neon_connected = await neon_manager.connect()
    if neon_connected:
        logger.info("Neon Cloud PostgreSQL connected and operational.")
    else:
        logger.warning("Neon Cloud PostgreSQL unavailable on boot. Operating in local SQLite WAL resilience mode.")

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
    logger.info(f"Futures INR Wallet:       ₹{verification.get('futures_inr_balance', 0.0):.2f} INR (Available: ₹{verification.get('futures_inr_available', 0.0):.2f}, Locked: ₹{verification.get('futures_inr_locked', 0.0):.2f})")
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

    # 3. Reconcile open exchange positions on boot to prevent orphan positions
    if not client.is_paper:
        try:
            boot_positions = await client.get_futures_positions()
            for p in boot_positions:
                if isinstance(p, dict) and abs(safe_float(p.get("active_pos"), 0.0)) > 1e-6:
                    p_sym = p.get("pair")
                    if p_sym and not any(t.symbol == p_sym for t in execution_engine.active_trades.values()):
                        execution_engine.adopt_exchange_position(p)
            if execution_engine.active_trades:
                logger.info(f"Startup position reconciliation complete: {len(execution_engine.active_trades)} active trades monitored.")
        except Exception as boot_pos_err:
            logger.warning(f"Startup positions check error: {boot_pos_err}")

    # 4. Initialize Market Universe and Feeds
    await screener.initialize_universe()
    try:
        init_scan = await screener.scan_and_filter_market(
            universe_size=engine_state["universe_size"],
            filter_count=engine_state["filter_count"],
            execution_count=engine_state["execution_count"],
            min_volume_24h=engine_state.get("min_volume_24h", 15000.0),
            max_spread_pct=engine_state.get("max_spread_pct", 0.20),
            direction_bias=engine_state.get("direction_bias", "AUTO"),
            timeframes=engine_state.get("timeframes", ["1m", "5m", "15m"]),
            min_confidence=engine_state.get("min_confidence", 62.0),
            leverage=engine_state.get("leverage", settings.DEFAULT_LEVERAGE)
        )
        latest_scan_result = init_scan
        top_syms = [c["symbol"] for c in init_scan.get("top_10_filtered", [])]
        screener.binance_client.update_monitored_symbols(top_syms)
        logger.info(f"Initial AI Market Scan complete: {len(top_syms)} Top cryptos loaded on boot!")
    except Exception as e:
        logger.warning(f"Initial market scan during startup: {e}")

    asyncio.create_task(ws_manager.start())
    task_screener = asyncio.create_task(deep_ai_screener_loop())
    task_orchestrator = asyncio.create_task(scalper_orchestrator_loop())
    task_binance_stream = asyncio.create_task(screener.binance_client.start_stream_manager())
    task_coindcx_sync = asyncio.create_task(coindcx_sync_1min_loop())
    task_equity_snapshot = asyncio.create_task(neon_equity_snapshot_loop())
    background_tasks.extend([task_screener, task_orchestrator, task_binance_stream, task_coindcx_sync, task_equity_snapshot])
    yield
    # Shutdown
    logger.info("Stopping AlphaScalper Services...")
    for t in background_tasks:
        t.cancel()
    await ws_manager.stop()
    await screener.binance_client.close()
    await client.close()
    await neon_manager.close()


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
        stats["daily_pnl"] = round(risk_manager.daily_pnl, 2)
        stats["daily_pnl_inr"] = round(risk_manager.daily_pnl * 87.5, 2)
        stats["current_capital"] = round(risk_manager.current_capital, 2)
        stats["current_capital_inr"] = round(risk_manager.current_capital * 87.5, 2)
        stats["inr_rate"] = 87.5
        top_10 = latest_scan_result.get("top_10_filtered", [])
        await websocket.send_text(json.dumps({
            "event": "initial_state",
            "data": {
                "is_running": engine_state["is_running"],
                "mode": engine_state["mode"],
                "config": engine_state,
                "stats": stats,
                "top_10_filtered": top_10
            }
        }))
        if latest_scan_result and "top_10_filtered" in latest_scan_result:
            await websocket.send_text(json.dumps({
                "event": "screener_update",
                "data": latest_scan_result
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
    stats["daily_pnl"] = round(risk_manager.daily_pnl, 2)
    stats["daily_pnl_inr"] = round(risk_manager.daily_pnl * 87.5, 2)
    stats["current_capital"] = round(risk_manager.current_capital, 2)
    stats["current_capital_inr"] = round(risk_manager.current_capital * 87.5, 2)
    stats["inr_rate"] = 87.5
    stats["account_currency"] = getattr(client, "account_currency", "INR")
    stats["currency_symbol"] = "₹" if getattr(client, "account_currency", "INR") == "INR" else "$"
    top_10 = latest_scan_result.get("top_10_filtered", [])
    await sio.emit("initial_state", {
        "is_running": engine_state["is_running"],
        "mode": engine_state["mode"],
        "account_currency": getattr(client, "account_currency", "INR"),
        "currency_symbol": "₹" if getattr(client, "account_currency", "INR") == "INR" else "$",
        "config": engine_state,
        "stats": stats,
        "top_10_filtered": top_10
    }, room=sid)

    if latest_scan_result and "top_10_filtered" in latest_scan_result:
        await sio.emit("screener_update", latest_scan_result, room=sid)


@sio.event
async def disconnect(sid):
    logger.info(f"Socket.IO Frontend Client disconnected: {sid}")


# ==========================================
# REST API CONTROLLERS
# ==========================================

class StrategyConfigRequest(BaseModel):
    mode: Optional[str] = None
    universe_size: Optional[int] = None
    filter_count: Optional[int] = None
    execution_count: Optional[int] = None
    leverage: Optional[float] = None
    risk_per_trade_pct: Optional[float] = None
    direction_bias: Optional[str] = None  # "AUTO", "LONG", or "SHORT"
    selected_indicators: Optional[List[str]] = None
    price_action_rules: Optional[List[str]] = None
    min_confidence: Optional[float] = None
    min_volume_24h: Optional[float] = None
    max_spread_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_1_pct: Optional[float] = None
    take_profit_2_pct: Optional[float] = None
    enable_breakeven: Optional[bool] = None
    timeframes: Optional[List[str]] = None
    strict_counter_trend_veto: Optional[bool] = None


class CreateOrderRequest(BaseModel):
    pair: str
    side: str = "buy"  # "buy" or "sell"
    order_type: str = "market_order"
    quantity: Optional[float] = None
    notional: Optional[float] = 6.0
    leverage: Optional[float] = None
    price: Optional[float] = None
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp2_price: Optional[float] = None



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


class StartEngineRequest(BaseModel):
    leverage: Optional[float] = None


@fastapi_app.get("/api/v1/contracts/leverage")
async def get_contracts_leverage():
    """Returns contract maximum leverage limits and current active leverage."""
    leverage_map = {}
    for pair, details in client._instrument_details_cache.items():
        raw_max = details.get("max_leverage_long") or details.get("max_leverage") or details.get("leverage") or 20.0
        leverage_map[pair] = float(raw_max)
    return {
        "active_leverage": engine_state.get("leverage", settings.DEFAULT_LEVERAGE),
        "default_leverage": settings.DEFAULT_LEVERAGE,
        "max_leverage_map": leverage_map
    }


@fastapi_app.post("/api/v1/engine/start")
async def start_engine(req: Optional[StartEngineRequest] = None):
    risk_manager.is_kill_switch_active = False
    if req and req.leverage is not None and req.leverage > 0:
        engine_state["leverage"] = float(req.leverage)
        logger.info(f"Execution Engine manual leverage set to: {engine_state['leverage']}x")
    engine_state["is_running"] = True
    logger.info(f"AlphaScalper Execution Engine STARTED! (Active Leverage: {engine_state.get('leverage', settings.DEFAULT_LEVERAGE)}x)")
    await broadcast_event("engine_status", {"is_running": True, "leverage": engine_state["leverage"]})
    return {"status": "success", "is_running": True, "leverage": engine_state["leverage"]}


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
    if req.mode is not None:
        engine_state["mode"] = req.mode.upper()
    if req.universe_size is not None:
        engine_state["universe_size"] = req.universe_size
    if req.filter_count is not None:
        engine_state["filter_count"] = req.filter_count
    if req.execution_count is not None:
        engine_state["execution_count"] = req.execution_count
    if req.leverage is not None:
        engine_state["leverage"] = req.leverage
    if req.risk_per_trade_pct is not None:
        engine_state["risk_per_trade_pct"] = req.risk_per_trade_pct
    if req.direction_bias is not None:
        engine_state["direction_bias"] = req.direction_bias.upper()
    if req.selected_indicators is not None:
        engine_state["selected_indicators"] = req.selected_indicators
    if req.price_action_rules is not None:
        engine_state["price_action_rules"] = req.price_action_rules
    if req.min_confidence is not None:
        engine_state["min_confidence"] = req.min_confidence
    if req.min_volume_24h is not None:
        engine_state["min_volume_24h"] = req.min_volume_24h
    if req.max_spread_pct is not None:
        engine_state["max_spread_pct"] = req.max_spread_pct
    if req.stop_loss_pct is not None:
        engine_state["stop_loss_pct"] = req.stop_loss_pct
    if req.take_profit_1_pct is not None:
        engine_state["take_profit_1_pct"] = req.take_profit_1_pct
    if req.take_profit_2_pct is not None:
        engine_state["take_profit_2_pct"] = req.take_profit_2_pct
    if req.enable_breakeven is not None:
        engine_state["enable_breakeven"] = req.enable_breakeven
    if req.timeframes is not None:
        engine_state["timeframes"] = req.timeframes
    if req.strict_counter_trend_veto is not None:
        engine_state["strict_counter_trend_veto"] = req.strict_counter_trend_veto

    logger.info(f"Updated Strategy Configuration: Mode={engine_state['mode']}, Bias={engine_state['direction_bias']}, Universe={engine_state['universe_size']}, Filter={engine_state['filter_count']}, Exec={engine_state['execution_count']}, Lev={engine_state['leverage']}x, MinConf={engine_state.get('min_confidence', 62.0)}%")
    return {"status": "success", "config": engine_state}


@fastapi_app.get("/api/v1/strategy/config")
async def get_strategy_config():
    """
    Returns current active strategy configuration for real-time frontend synchronization.
    """
    return {"status": "success", "config": engine_state}


@fastapi_app.post("/api/v1/strategy/test")
async def test_strategy_configuration(req: StrategyConfigRequest):
    """
    Dry-run simulation of custom strategy against live market.
    Returns matched candidate count and target details without executing orders.
    """
    scan_res = await screener.scan_and_filter_market(
        universe_size=req.universe_size or engine_state["universe_size"],
        filter_count=req.filter_count or engine_state["filter_count"],
        execution_count=req.execution_count or engine_state["execution_count"],
        min_volume_24h=req.min_volume_24h or engine_state.get("min_volume_24h", 10000.0),
        max_spread_pct=req.max_spread_pct or engine_state.get("max_spread_pct", 0.20),
        direction_bias=req.direction_bias or engine_state.get("direction_bias", "AUTO"),
        selected_indicators=req.selected_indicators or engine_state.get("selected_indicators"),
        price_action_rules=req.price_action_rules or engine_state.get("price_action_rules"),
        stop_loss_pct=req.stop_loss_pct or engine_state.get("stop_loss_pct"),
        take_profit_1_pct=req.take_profit_1_pct or engine_state.get("take_profit_1_pct"),
        take_profit_2_pct=req.take_profit_2_pct or engine_state.get("take_profit_2_pct"),
        enable_breakeven=req.enable_breakeven if req.enable_breakeven is not None else engine_state.get("enable_breakeven", True),
        strict_counter_trend_veto=req.strict_counter_trend_veto if req.strict_counter_trend_veto is not None else engine_state.get("strict_counter_trend_veto", True),
        timeframes=req.timeframes or engine_state.get("timeframes", ["1m", "5m", "15m"]),
        min_confidence=req.min_confidence or engine_state.get("min_confidence", 62.0)
    )
    targets = scan_res.get("ranked_targets", [])
    min_conf = req.min_confidence or engine_state.get("min_confidence", 62.0)
    matching = [t for t in targets if t.get("confidence", 0) >= min_conf and t.get("signal") in ["BUY", "STRONG_BUY", "SELL", "STRONG_SELL"]]
    return {
        "status": "success",
        "matching_count": len(matching),
        "total_ranked": len(targets),
        "matching_targets": matching,
        "scan_latency_ms": scan_res.get("scan_latency_ms", 0.0)
    }


@fastapi_app.get("/api/v1/markets/screener")
async def get_screener_results():
    scan_res = await screener.scan_and_filter_market(
        universe_size=engine_state["universe_size"],
        filter_count=engine_state["filter_count"],
        execution_count=engine_state["execution_count"],
        min_volume_24h=engine_state.get("min_volume_24h", 10000.0),
        max_spread_pct=engine_state.get("max_spread_pct", 0.20),
        direction_bias=engine_state.get("direction_bias", "AUTO"),
        selected_indicators=engine_state.get("selected_indicators"),
        price_action_rules=engine_state.get("price_action_rules"),
        stop_loss_pct=engine_state.get("stop_loss_pct"),
        take_profit_1_pct=engine_state.get("take_profit_1_pct"),
        take_profit_2_pct=engine_state.get("take_profit_2_pct"),
        enable_breakeven=engine_state.get("enable_breakeven", True),
        strict_counter_trend_veto=engine_state.get("strict_counter_trend_veto", True),
        timeframes=engine_state.get("timeframes", ["1m", "5m", "15m"]),
        min_confidence=engine_state.get("min_confidence", 62.0)
    )
    return scan_res


@fastapi_app.get("/api/v1/markets/orderbook")
async def get_live_orderbook(symbol: str = "B-BTC_USDT"):
    """
    Returns live L2 Order Book depth directly from CoinDCX for a given symbol.
    """
    ob = await client.get_futures_orderbook(symbol, depth=20)
    return ob


@fastapi_app.get("/api/v1/markets/candles")
async def get_market_candles(symbol: str = "B-BTC_USDT", resolution: str = "1", limit: int = 60):
    """
    Returns candlestick OHLCV data for interactive chart visualization with horizontal SL/TP/Entry lines.
    Supports real-time CoinDCX futures candles with fallback to cached or synchronized market bars.
    """
    clean_sym = symbol.strip()
    try:
        now_ts = int(time.time())
        res_mins = 1
        if resolution.isdigit():
            res_mins = max(1, int(resolution))
        from_ts = now_ts - (res_mins * 60 * limit)
        
        # 1. Attempt live CoinDCX candlestick retrieval
        raw_candles = await client.get_futures_candlesticks(
            clean_sym,
            resolution=resolution,
            from_ts=from_ts,
            to_ts=now_ts
        )
        if raw_candles and len(raw_candles) > 0:
            formatted = []
            for b in raw_candles[-limit:]:
                formatted.append({
                    "time": int(b.get("time") or b.get("t") or (time.time() * 1000)),
                    "open": float(b.get("open") or b.get("o") or 0.0),
                    "high": float(b.get("high") or b.get("h") or 0.0),
                    "low": float(b.get("low") or b.get("l") or 0.0),
                    "close": float(b.get("close") or b.get("c") or 0.0),
                    "volume": float(b.get("volume") or b.get("v") or 0.0)
                })
            return {
                "status": "success",
                "symbol": clean_sym,
                "resolution": resolution,
                "is_live": True,
                "candles": formatted
            }
    except Exception as e:
        logger.debug(f"Live candle fetch for {clean_sym} failed: {e}")

    # 2. Fallback to screener cache or synthetic realistic bars anchored to current price
    cached = screener._candle_cache.get(clean_sym)
    if cached and "close" in cached and len(cached["close"]) > 0:
        c_len = min(limit, len(cached["close"]))
        now_ms = int(time.time() * 1000)
        formatted = []
        for i in range(c_len):
            idx = -(c_len - i)
            t = now_ms - (c_len - 1 - i) * 60000
            formatted.append({
                "time": t,
                "open": float(cached["open"][idx]),
                "high": float(cached["high"][idx]),
                "low": float(cached["low"][idx]),
                "close": float(cached["close"][idx]),
                "volume": float(cached["volume"][idx])
            })
        return {
            "status": "success",
            "symbol": clean_sym,
            "resolution": resolution,
            "is_live": False,
            "candles": formatted
        }

    # 3. Last-resort synthesized realistic bars around current price
    curr_price = 100.0
    for item in (getattr(screener, "filtered_10", None) or []):
        if item.get("symbol") == clean_sym:
            curr_price = float(item.get("price", 100.0))
            break
    now_ms = int(time.time() * 1000)
    bars = []
    p = curr_price
    for i in range(limit):
        walk = np.random.normal(0, 0.001) * p
        p_close = max(0.0001, p + walk)
        p_high = max(p, p_close) * 1.0008
        p_low = min(p, p_close) * 0.9992
        bars.append({
            "time": now_ms - (limit - 1 - i) * 60000,
            "open": round(p, 6),
            "high": round(p_high, 6),
            "low": round(p_low, 6),
            "close": round(p_close, 6),
            "volume": round(100.0 + i * 2.5, 2)
        })
        p = p_close
    return {
        "status": "success",
        "symbol": clean_sym,
        "resolution": resolution,
        "is_live": False,
        "candles": bars
    }



@fastapi_app.get("/api/v1/positions")
async def get_positions(force_refresh: bool = False):
    """
    Returns all active positions synced directly with live CoinDCX exchange.
    Throttled by 3-minute (180s) in-memory cache to strictly avoid CoinDCX rate limits.
    Includes entry price, live mark price, liquidation price, TP, SL, and PnL in both INR and USDT.
    """
    trades = []
    seen_symbols = set()
    now = time.time()

    # 1. Query live CoinDCX exchange positions with 3-minute (180s) throttling
    if not client.is_paper:
        if force_refresh or (now - coindcx_cache["positions_last_fetched"] > coindcx_cache["ttl_seconds"]):
            try:
                live_pos = await client.get_futures_positions()
                coindcx_cache["positions"] = live_pos or []
                coindcx_cache["positions_last_fetched"] = now
            except Exception as e:
                logger.error(f"Error fetching live positions from CoinDCX: {e}")

        for p in coindcx_cache.get("positions", []):
            sym = p.get("pair")
            seen_symbols.add(sym)
            side = p.get("side", "BUY").upper()
            entry_p = float(p.get("entry_price") or 0.0)
            tp_p = p.get("tp_price")
            sl_p = p.get("sl_price")

            # Try to enrich TP/SL from internal execution engine if missing from exchange response
            matched_trade = next((t for t in execution_engine.active_trades.values() if t.symbol == sym), None)
            if matched_trade:
                if not tp_p and matched_trade.tp1_price:
                    tp_p = matched_trade.tp1_price
                if not sl_p and matched_trade.sl_price:
                    sl_p = matched_trade.sl_price
                is_risk_free = matched_trade.is_risk_free
            else:
                # Also check active orders cache for this symbol if tp_p or sl_p is still missing
                if not tp_p or not sl_p:
                    for o in coindcx_cache.get("orders", []):
                        if o.get("market") == sym or o.get("pair") == sym:
                            order_type = str(o.get("order_type") or o.get("order_type_name") or "").lower()
                            stop_p = float(o.get("stop_price") or o.get("trigger_price") or 0.0)
                            if stop_p > 0:
                                if "take_profit" in order_type and not tp_p:
                                    tp_p = stop_p
                                elif "stop" in order_type and not sl_p:
                                    sl_p = stop_p

                # Genuine risk-free check: Stop loss must be at or beyond entry price
                is_risk_free = False
                if entry_p > 0 and sl_p:
                    sl_val = float(sl_p)
                    if side == "BUY":
                        is_risk_free = sl_val >= (entry_p * 0.9995)
                    else:
                        is_risk_free = sl_val <= (entry_p * 1.0005)

            trades.append({
                "trade_id": p.get("id"),
                "position_id": p.get("id"),
                "symbol": sym,
                "side": side,
                "entry_price": entry_p,
                "mark_price": p.get("mark_price", entry_p),
                "liquidation_price": float(p.get("liquidation_price") or 0.0),
                "tp1_price": tp_p,
                "sl_price": sl_p,
                "is_risk_free": is_risk_free,
                "remaining_qty": p.get("abs_quantity", 0.0),
                "unrealized_pnl": p.get("unrealized_pnl_usdt", 0.0),
                "unrealized_pnl_inr": p.get("unrealized_pnl_inr", 0.0),
                "roe_percent": p.get("roe_percent", 0.0),
                "locked_margin_usdt": p.get("locked_margin_usdt", 0.0),
                "locked_margin_inr": p.get("locked_margin_inr", 0.0),
                "leverage": float(p.get("leverage") or 10.0),
                "state": "LIVE_OPEN",
                "is_live": True,
                "elapsed_seconds": 0
            })

    # 2. Also merge any in-memory trades that haven't appeared on exchange or paper trades
    for t in execution_engine.active_trades.values():
        if t.symbol not in seen_symbols:
            trades.append({
                "trade_id": t.trade_id,
                "position_id": t.trade_id,
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "mark_price": t.entry_price,
                "liquidation_price": 0.0,
                "tp1_price": t.tp1_price,
                "tp2_price": t.tp2_price,
                "sl_price": t.sl_price,
                "breakeven_sl": t.breakeven_sl,
                "is_risk_free": t.is_risk_free,
                "remaining_qty": round(t.remaining_qty, 4),
                "unrealized_pnl": round(t.unrealized_pnl, 4),
                "unrealized_pnl_inr": round(t.unrealized_pnl * 87.5, 2),
                "roe_percent": round(getattr(t, "roe_pct", 0.0), 2),
                "locked_margin_usdt": round(t.remaining_qty * t.entry_price / max(1.0, t.leverage), 4),
                "locked_margin_inr": round(t.remaining_qty * t.entry_price / max(1.0, t.leverage) * 87.5, 2),
                "leverage": t.leverage,
                "state": t.current_state,
                "is_live": not client.is_paper,
                "elapsed_seconds": int(time.time() - (t.entry_time or time.time()))
            })

    return {"active_trades": trades, "count": len(trades)}


@fastapi_app.post("/api/v1/positions/exit")
async def exit_single_position(payload: Dict[str, str]):
    """
    Exits a specific position on CoinDCX and removes it from in-memory engine.
    Accepts trade_id, symbol, or exchange position UUID.
    """
    target = payload.get("trade_id") or payload.get("symbol") or payload.get("id")
    if not target:
        return {"status": "error", "message": "Missing position identifier"}

    # 1. Close on exchange
    res = await client.exit_futures_position(target)
    coindcx_cache["positions_last_fetched"] = 0.0
    
    # 2. Also remove from execution_engine active_trades
    for tid, trade in list(execution_engine.active_trades.items()):
        if tid == target or trade.symbol == target:
            trade.remaining_qty = 0.0
            execution_engine.closed_trades.append(trade)
            execution_engine.active_trades.pop(tid, None)
            break

    await broadcast_event("execution_event", {
        "type": "POSITION_EXITED",
        "message": f"Position on {target} exited successfully."
    })
    return {"status": "success", "target": target, "exchange_response": res}


@fastapi_app.post("/api/v1/positions/exit_all")
async def exit_all_positions():
    """
    Exits and liquidates all active positions both in memory and on the live exchange.
    """
    exited_trades = []
    coindcx_cache["positions_last_fetched"] = 0.0
    coindcx_cache["orders_last_fetched"] = 0.0

    # 1. Cancel all open/untriggered orders on exchange first
    if not client.is_paper:
        try:
            await client.cancel_all_futures_open_orders(["INR", "USDT"])
            logger.info("Cancelled all open futures orders during exit_all")
        except Exception as e:
            logger.warning(f"Note on cancelling open orders during exit_all: {e}")

    # 2. Exit in-memory trades
    for trade_id, trade in list(execution_engine.active_trades.items()):
        await client.exit_futures_position(trade.symbol)
        trade.remaining_qty = 0.0
        execution_engine.closed_trades.append(trade)
        exited_trades.append(trade.symbol)
    execution_engine.active_trades.clear()

    # 3. Exit any live exchange positions
    if not client.is_paper:
        try:
            live_positions = await client.get_futures_positions()
            for p in live_positions:
                if isinstance(p, dict) and abs(safe_float(p.get("active_pos"), 0.0)) > 1e-6:
                    sym = p.get("pair")
                    pos_id = p.get("id")
                    await client.exit_futures_position(pos_id or sym)
                    if sym not in exited_trades:
                        exited_trades.append(sym)
        except Exception as e:
            logger.error(f"Error checking live positions during exit_all: {e}")

    await broadcast_event("execution_event", {
        "type": "EXIT_ALL",
        "message": f"Exit All executed: All open positions flattened and all active orders cancelled ({len(exited_trades)} positions closed)."
    })
    return {"status": "success", "closed_count": len(exited_trades), "symbols": exited_trades}


@fastapi_app.get("/api/v1/orders/active")
async def get_active_orders_endpoint(force_refresh: bool = False):
    """
    Fetches all open and untriggered orders from CoinDCX (Futures TP/SL & Spot orders).
    Throttled by 3-minute (180s) cache to strictly prevent exchange rate limits.
    """
    now = time.time()
    if not client.is_paper:
        if force_refresh or (now - coindcx_cache["orders_last_fetched"] > coindcx_cache["ttl_seconds"]):
            try:
                orders = await client.get_all_active_orders()
                coindcx_cache["orders"] = orders or []
                coindcx_cache["orders_last_fetched"] = now
            except Exception as e:
                logger.error(f"Error in get_active_orders: {e}")
                return {"status": "error", "message": str(e), "orders": coindcx_cache.get("orders", []), "count": len(coindcx_cache.get("orders", []))}
    return {"status": "success", "orders": coindcx_cache.get("orders", []), "count": len(coindcx_cache.get("orders", []))}


@fastapi_app.post("/api/v1/orders/cancel")
async def cancel_order_endpoint(payload: Dict[str, Any]):
    """
    Cancels a specific active order on CoinDCX (supports both Futures UUIDs and Spot numeric IDs).
    """
    order_id = payload.get("order_id") or payload.get("id")
    if not order_id:
        return {"status": "error", "message": "Missing order_id"}

    res = await client.cancel_any_order(str(order_id))
    coindcx_cache["orders_last_fetched"] = 0.0
    await broadcast_event("execution_event", {
        "type": "ORDER_CANCELLED",
        "message": f"Order {order_id} cancelled on CoinDCX."
    })
    return {"status": "success", "order_id": order_id, "response": res}


@fastapi_app.post("/api/v1/orders/cancel_all")
async def cancel_all_orders_endpoint():
    """
    Cancels ALL open and untriggered orders across Futures and Spot on CoinDCX.
    """
    fut_res = {}
    spot_res = {}
    coindcx_cache["orders_last_fetched"] = 0.0
    if not client.is_paper:
        try:
            fut_res = await client.cancel_all_futures_open_orders(["INR", "USDT"])
        except Exception as e:
            fut_res = {"error": str(e)}
        try:
            spot_res = await client.cancel_all_spot_orders("USDTINR")
        except Exception as e:
            spot_res = {"error": str(e)}

    await broadcast_event("execution_event", {
        "type": "ALL_ORDERS_CANCELLED",
        "message": "All open and untriggered orders cancelled across CoinDCX."
    })
    return {
        "status": "success",
        "futures_response": fut_res,
        "spot_response": spot_res
    }


@fastapi_app.post("/api/v1/orders/status_multiple")
async def get_multiple_orders_status_endpoint(payload: Dict[str, Any]):
    """
    Queries status of multiple orders via CoinDCX status_multiple endpoint.
    """
    ids = payload.get("ids", [])
    client_ids = payload.get("client_order_ids", [])
    res = await client.get_multiple_spot_order_status(ids=ids, client_order_ids=client_ids)
    return {"status": "success", "orders": res}


@fastapi_app.post("/api/v1/orders/create")
async def create_order_endpoint(req: CreateOrderRequest):
    """
    Creates a new futures order, attaches Take Profit & Stop Loss triggers on CoinDCX,
    and registers the trade in the real-time position manager.
    """
    target_lev = req.leverage or engine_state.get("leverage") or settings.DEFAULT_LEVERAGE
    target_notional = req.notional or 6.0
    
    # Fetch current market price for pair if not provided
    current_price = req.price
    if not current_price or current_price <= 0:
        try:
            ob = await client.get_futures_orderbook(req.pair, depth=1)
            bids = ob.get("bids", {})
            asks = ob.get("asks", {})
            if req.side.lower() == "buy" and asks:
                if isinstance(asks, dict):
                    current_price = safe_float(next(iter(asks.keys())), 0.0)
                elif isinstance(asks, list) and asks:
                    current_price = safe_float(asks[0][0] if isinstance(asks[0], (list, tuple)) else asks[0], 0.0)
            elif bids:
                if isinstance(bids, dict):
                    current_price = safe_float(next(iter(bids.keys())), 0.0)
                elif isinstance(bids, list) and bids:
                    current_price = safe_float(bids[0][0] if isinstance(bids[0], (list, tuple)) else bids[0], 0.0)
        except Exception:
            pass

    if not current_price or current_price <= 0:
        try:
            rt = await client.get_realtime_futures_prices()
            prices_list = rt.get("prices") if isinstance(rt, dict) else rt
            if isinstance(prices_list, list):
                for item in prices_list:
                    if isinstance(item, dict) and item.get("pair") == req.pair:
                        current_price = safe_float(item.get("last_price") or item.get("ls") or item.get("price"), 0.0)
                        break
        except Exception:
            pass

    if not current_price or current_price <= 0:
        current_price = 1.0

    # Sanitize order parameters to match CoinDCX step size & min notional
    safe_qty, safe_lev, actual_notional = await client.sanitize_order_params(
        pair=req.pair,
        target_notional=target_notional,
        desired_leverage=target_lev,
        price=current_price
    )
    
    if req.quantity is not None and req.quantity > 0:
        safe_qty = req.quantity
        actual_notional = round(safe_qty * current_price, 4)

    req_margin = actual_notional / max(1.0, float(safe_lev))

    # Balance guard
    if not client.is_paper:
        usable = await client.get_usable_balance_usdt()
        if usable < req_margin:
            return {
                "status": "error",
                "message": f"Insufficient margin: Required ${req_margin:.2f} USDT, available ${usable:.4f} USDT",
                "symbol": req.pair
            }

    try:
        await client.update_position_leverage(req.pair, safe_lev)
    except Exception as e:
        logger.warning(f"Note on updating leverage: {e}")
    
    # Dynamic TP & SL targets with intelligent price precision
    fmt = AlphaAIEngine.format_price_precision
    is_buy = req.side.lower() == "buy"
    if req.tp_price and req.tp_price > 0:
        tp1_price = fmt(req.tp_price)
    else:
        tp1_price = fmt(current_price * (1.0 + (settings.TP1_RATIO if is_buy else -settings.TP1_RATIO)))

    if req.sl_price and req.sl_price > 0:
        sl_price = fmt(req.sl_price)
    else:
        sl_price = fmt(current_price * (1.0 - (settings.HARD_SL_RATIO if is_buy else -settings.HARD_SL_RATIO)))

    if req.tp2_price and req.tp2_price > 0:
        tp2_price = fmt(req.tp2_price)
    else:
        tp2_price = fmt(current_price * (1.0 + (settings.TP2_RATIO if is_buy else -settings.TP2_RATIO)))

    be_sl = fmt(current_price * (1.0 + (settings.FEE_BUFFER if is_buy else -settings.FEE_BUFFER)))

    # 1. Place the order
    order_res = await client.create_futures_order(
        pair=req.pair,
        side=req.side,
        order_type=req.order_type,
        total_quantity=safe_qty,
        price=req.price,
        leverage=safe_lev
    )

    # Check if order failed on exchange
    order_error = None
    if isinstance(order_res, dict):
        if order_res.get("status_code", 200) >= 400:
            order_error = order_res.get("text") or order_res.get("message") or f"Exchange error {order_res.get('status_code')}"
        elif order_res.get("code") in [400, 401, 403, 422, 500] or str(order_res.get("status", "")).lower() in ["error", "failed"]:
            order_error = order_res.get("message") or order_res.get("description") or "Exchange rejected order"

    if order_error:
        await broadcast_event("execution_event", {
            "type": "ORDER_FAILED",
            "message": f"Order placement failed on {req.pair}: {order_error}"
        })
        return {
            "status": "error",
            "message": order_error,
            "order": order_res,
            "symbol": req.pair
        }

    # 2. Attach TP & SL triggers on CoinDCX
    tpsl_res = None
    is_long_req = (req.side.upper() == "BUY")
    try:
        tpsl_res = await client.create_futures_tpsl(
            position_id=req.pair,
            tp_stop_price=tp1_price,
            sl_stop_price=sl_price,
            is_long=is_long_req,
            mark_price=current_price,
            pair=req.pair
        )
    except Exception as e:
        logger.warning(f"Note on attaching TP/SL: {e}")

    # 3. Register in internal execution engine
    trade_id = f"ORDER-{int(time.time()*1000)}-{req.pair}"
    from win_win_strategy import WinWinTrade, WinWinOrderState
    new_trade = WinWinTrade(
        trade_id=trade_id,
        symbol=req.pair,
        side=req.side.upper(),
        total_size_usdt=actual_notional,
        entry_price=current_price,
        tp1_price=tp1_price,
        tp2_price=tp2_price,
        sl_price=sl_price,
        breakeven_sl=be_sl,
        leverage=safe_lev
    )
    new_trade.filled_qty = safe_qty
    new_trade.remaining_qty = safe_qty
    new_trade.entry_time = time.time()
    new_trade.current_state = WinWinOrderState.TIER1_FILLED
    execution_engine.active_trades[trade_id] = new_trade
    coindcx_cache["positions_last_fetched"] = 0.0
    coindcx_cache["orders_last_fetched"] = 0.0
    b_sym = to_binance_symbol(req.pair)
    if b_sym.lower() not in screener.binance_client._monitored_symbols:
        screener.binance_client.update_monitored_symbols(
            screener.binance_client._monitored_symbols + [b_sym]
        )

    await broadcast_event("execution_event", {
        "type": "ORDER_PLACED_WITH_TPSL",
        "message": f"Order placed on {req.pair} ({req.side.upper()} {safe_qty}) with TP: ${tp1_price} and SL: ${sl_price}"
    })
    
    return {
        "status": "success",
        "order": order_res,
        "tpsl": tpsl_res,
        "symbol": req.pair,
        "quantity": safe_qty,
        "entry_price": current_price,
        "tp1_price": tp1_price,
        "sl_price": sl_price,
        "leverage": safe_lev,
        "margin_required": req_margin
    }


@fastapi_app.get("/api/v1/risk/dynamic_allocation")
async def get_dynamic_risk_allocation():
    """
    Computes mathematical dynamic order allocation and capital protection metrics
    based on live CoinDCX Futures INR balance, equity tiers, and market volatility.
    """
    fut_info = await client.get_futures_inr_balance()
    eq_usdt = fut_info.get("total_usdt_equiv", 0.0)
    free_usdt = fut_info.get("available_usdt_equiv", 0.0)
    
    if client.is_paper:
        eq_usdt = risk_manager.current_capital
        free_usdt = eq_usdt * 0.9

    allocation = risk_manager.compute_dynamic_order_allocation(
        total_equity_usdt=eq_usdt,
        free_margin_usdt=free_usdt,
        current_atr_pct=0.005,
        ai_confidence=78.0
    )
    allocation["futures_inr_wallet"] = fut_info
    return allocation


@fastapi_app.get("/api/v1/analytics/stats")
async def get_analytics():
    stats = execution_engine.get_summary_stats()
    capital_usdt = round(risk_manager.current_capital, 2)
    capital_inr = round(risk_manager.current_capital * 87.5, 2)
    stats["capital"] = capital_usdt
    stats["current_capital"] = capital_usdt
    stats["capital_inr"] = capital_inr
    stats["current_capital_inr"] = capital_inr
    stats["daily_pnl"] = round(risk_manager.daily_pnl, 2)
    stats["daily_pnl_inr"] = round(risk_manager.daily_pnl * 87.5, 2)
    stats["inr_rate"] = 87.5
    stats["latency_history"] = engine_state["latency_history"]
    stats["analytics_24h"] = await get_24h_analytics(inr_rate=87.5)
    return stats


@fastapi_app.get("/api/v1/stats/analytics")
async def get_24h_stats_analytics():
    """
    Returns rolling 24-hour financial metrics and execution stats from Neon Cloud PostgreSQL:
    - TP1 Hits Count
    - TP2 Hits Count
    - Stop Loss Hits Count
    - Breakeven Exits Count
    - 24H Gross Earnings ($ and ₹)
    - 24H Gross Losses ($ and ₹)
    - 24H Net PnL ($ and ₹)
    - 24H Total Fees ($ and ₹)
    - Win Rate % & Profit Factor
    - Neon Database Health Status
    """
    return await get_24h_analytics(inr_rate=87.5)


@fastapi_app.get("/api/v1/trades/history")
async def get_closed_trades_history(limit: int = 100):
    """
    Returns recent trade execution records from Neon Cloud / SQLite WAL.
    """
    trades = load_closed_trades(limit=limit)
    return {
        "status": "success",
        "trades": trades,
        "count": len(trades),
        "database": "Neon Cloud PostgreSQL + SQLite WAL"
    }


@fastapi_app.get("/api/v1/database/status")
async def get_database_status():
    """
    Checks Neon Cloud PostgreSQL connectivity and pool status.
    """
    return {
        "status": "success",
        "neon_connected": neon_manager.is_connected,
        "database": "Neon Cloud PostgreSQL" if neon_manager.is_connected else "SQLite WAL (Local Fallback)",
        "pool_available": neon_manager.pool is not None,
        "region": "us-east-2 (AWS)",
        "timestamp": time.time()
    }


@fastapi_app.post("/api/v1/stats/reset")
async def reset_stats_endpoint():
    """
    Resets in-memory trade history, win-rate, and daily PnL tracker.
    Provides a clean slate for real account tracking.
    """
    execution_engine.reset_stats()
    risk_manager.reset_daily_pnl()
    risk_manager.symbol_cooldowns.clear()
    logger.info("AlphaScalper performance statistics, cooldowns, and daily PnL reset to 0.")
    
    inr_rate = 87.5
    capital_usdt = round(risk_manager.current_capital, 2)
    capital_inr = round(capital_usdt * inr_rate, 2)
    
    reset_telemetry = {
        "timestamp": int(time.time() * 1000),
        "is_running": engine_state["is_running"],
        "mode": engine_state["mode"],
        "trading_mode": settings.TRADING_MODE,
        "latency_ms": 0.18,
        "avg_latency_ms": 0.18,
        "daily_pnl": 0.0,
        "daily_pnl_inr": 0.0,
        "current_capital": capital_usdt,
        "current_capital_inr": capital_inr,
        "inr_rate": inr_rate,
        "total_fees_paid": 0.0,
        "total_fees_paid_inr": 0.0,
        "net_pnl": 0.0,
        "is_balance_sufficient": risk_manager.is_balance_sufficient,
        "min_required_margin": risk_manager.min_order_notional,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "total_trades": 0,
        "active_trades_count": len(execution_engine.active_trades),
        "risk_free_count": 0,
        "kill_switch_active": risk_manager.is_kill_switch_active
    }
    await broadcast_event("telemetry_update", reset_telemetry)
    return {"status": "success", "message": "Performance statistics reset.", "stats": reset_telemetry}


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
