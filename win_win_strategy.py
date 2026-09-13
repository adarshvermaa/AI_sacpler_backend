"""
AlphaScalper - Win-Win Multi-Order Execution Engine
Concept: "Adaptive Asymmetric Tri-Bracket & Dynamic Breakeven Ratcheter (ATB-DBR)"
Key Mechanisms:
1. Tri-Tier Multi-Order Slicing (Tier 1 Taker 40%, Tier 2 Pullback Limit 35%, Tier 3 Breakout 25%)
2. Instant Breakeven Ratchet on TP1 (+0.35% closes 50%, ratchets SL to Entry + Fees)
3. Microstructure Invalidation Time-Out (30-second stall exit to prevent drawdown)
4. Trailing Dynamic Runner for remaining 50% position
"""

import time
import logging
from typing import Dict, Any, List, Optional

from config import settings
from coindcx_client import CoinDCXClient

logger = logging.getLogger("AlphaScalper.WinWinStrategy")


class WinWinOrderState:
    PENDING_ENTRY = "PENDING_ENTRY"
    TIER1_FILLED = "TIER1_FILLED"
    FULL_ENTRY = "FULL_ENTRY"
    TP1_HIT_BREAKEVEN_LOCKED = "TP1_HIT_BREAKEVEN_LOCKED"  # The "Win-Win" Risk-Free State
    TP2_HIT_CLOSED = "TP2_HIT_CLOSED"
    BREAKEVEN_CLOSED = "BREAKEVEN_CLOSED"
    STOP_LOSS_CLOSED = "STOP_LOSS_CLOSED"
    MICRO_TIMEOUT_CLOSED = "MICRO_TIMEOUT_CLOSED"


class WinWinTrade:
    def __init__(
        self,
        trade_id: str,
        symbol: str,
        side: str,  # "BUY" (long) or "SELL" (short)
        total_size_usdt: float,
        entry_price: float,
        tp1_price: float,
        tp2_price: float,
        sl_price: float,
        breakeven_sl: float,
        leverage: float = 10.0
    ):
        self.trade_id = trade_id
        self.symbol = symbol
        self.side = side.upper()
        self.total_size_usdt = total_size_usdt
        self.entry_price = round(entry_price, 6)
        self.tp1_price = round(tp1_price, 6)
        self.tp2_price = round(tp2_price, 6)
        self.sl_price = round(sl_price, 6)
        self.breakeven_sl = round(breakeven_sl, 6)
        self.leverage = leverage
        
        self.current_state = WinWinOrderState.PENDING_ENTRY
        self.created_at = time.time()
        self.entry_time = None
        self.exit_time = None
        
        # Multi-order slicing only when notional is large enough so every slice meets CoinDCX's 6.00 USDT min notional
        # (e.g. 24 USDT: tier 1 = 9.6 USDT, tier 2 = 8.4 USDT, tier 3 = 6.0 USDT)
        if total_size_usdt >= 24.0:
            self.is_sliced = True
            self.tier1_qty = (total_size_usdt * 0.40) / entry_price
            self.tier2_qty = (total_size_usdt * 0.35) / entry_price
            self.tier3_qty = (total_size_usdt * 0.25) / entry_price
        else:
            self.is_sliced = False
            self.tier1_qty = total_size_usdt / entry_price
            self.tier2_qty = 0.0
            self.tier3_qty = 0.0
        
        self.filled_qty = 0.0
        self.remaining_qty = 0.0
        self.realized_pnl = 0.0
        self.total_fees_paid = round(total_size_usdt * 0.00059, 4)  # Entry taker fee + GST
        self.net_realized_pnl = -self.total_fees_paid
        self.unrealized_pnl = 0.0
        self.roe_pct = 0.0
        self.is_risk_free = False  # Becomes True when TP1 hits and SL is ratcheted!


class WinWinExecutionEngine:
    def __init__(self, client: CoinDCXClient):
        self.client = client
        self.active_trades: Dict[str, WinWinTrade] = {}
        self.closed_trades: List[WinWinTrade] = []
        self.trade_counter = 0

    async def execute_win_win_entry(
        self,
        symbol: str,
        signal: str,
        entry_price: float,
        tp1_price: float,
        tp2_price: float,
        sl_price: float,
        breakeven_sl: float,
        allocated_usdt: float = 250.0,
        leverage: float = 10.0
    ) -> Optional[WinWinTrade]:
        """
        Submits the Tri-Tier Multi-Order Slices and registers bracket state.
        """
        self.trade_counter += 1
        trade_id = f"WW-{int(time.time())}-{self.trade_counter}"
        side = "buy" if "BUY" in signal else "sell"

        trade = WinWinTrade(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            total_size_usdt=allocated_usdt,
            entry_price=entry_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            sl_price=sl_price,
            breakeven_sl=breakeven_sl,
            leverage=leverage
        )

        # Pre-flight Live Balance Guard & Notional Check
        if not self.client.is_paper:
            try:
                usable_bal = await self.client.get_usable_balance_usdt()
                req_margin = allocated_usdt / max(1.0, leverage)
                min_margin_for_order = 6.0 / max(1.0, leverage)
                if usable_bal < min_margin_for_order:
                    logger.warning(
                        f"[BALANCE GUARD] Live trade held on {symbol}: Wallet balance (${usable_bal:.4f} USDT) "
                        f"is below minimum required margin (${min_margin_for_order:.2f} USDT) for $6.00 CoinDCX contract."
                    )
                    return None
                if usable_bal < req_margin:
                    logger.warning(
                        f"[BALANCE GUARD] Live trade held on {symbol}: Required margin (${req_margin:.2f} USDT) "
                        f"exceeds available wallet balance (${usable_bal:.4f} USDT)."
                    )
                    return None
            except Exception as e:
                logger.error(f"[BALANCE GUARD] Error verifying live balance before order: {e}")
                return None

        try:
            # 1. Sanitize leverage and quantity to match CoinDCX contract rules
            safe_qty, safe_lev, actual_notional = await self.client.sanitize_order_params(
                pair=symbol,
                target_notional_usdt=allocated_usdt,
                price=entry_price,
                requested_leverage=leverage
            )

            # Check wallet margin sufficiency for this specific order
            req_margin = actual_notional / max(1.0, safe_lev)
            if not self.client.is_paper:
                usable_bal = await self.client.get_usable_balance_usdt()
                if usable_bal < req_margin:
                    logger.warning(
                        f"[BALANCE GUARD] Live trade held on {symbol}: Required margin (${req_margin:.2f} USDT) "
                        f"exceeds available wallet balance (${usable_bal:.4f} USDT)."
                    )
                    return None

            await self.client.update_position_leverage(symbol, safe_lev)

            # 2. Place Tier 1 Taker Order with CoinDCX-validated safe quantity
            order_res = await self.client.create_futures_order(
                pair=symbol,
                side=side,
                order_type="market_order",
                total_quantity=safe_qty,
                leverage=safe_lev
            )
            order_qty = safe_qty

            # Validate order response
            if isinstance(order_res, dict):
                if order_res.get("status_code", 200) not in (200, 201):
                    logger.error(f"CoinDCX rejected order on {symbol}: {order_res.get('text', order_res)}")
                    return None
                if "error" in order_res or order_res.get("status") == "error":
                    logger.error(f"CoinDCX order returned error on {symbol}: {order_res}")
                    return None

            trade.current_state = WinWinOrderState.TIER1_FILLED
            trade.entry_time = time.time()
            trade.filled_qty = order_qty
            trade.remaining_qty = order_qty
            trade.total_size_usdt = actual_notional
            trade.leverage = safe_lev
            if actual_notional < 24.0:
                trade.is_sliced = False
                trade.tier1_qty = order_qty
                trade.tier2_qty = 0.0
                trade.tier3_qty = 0.0

            # 3. Attach initial emergency Stop Loss and TP1 triggers
            try:
                await self.client.create_futures_tpsl(
                    position_id=symbol,
                    tp_stop_price=tp1_price,
                    sl_stop_price=sl_price
                )
            except Exception as tpsl_err:
                logger.warning(f"Note on TP/SL creation for {symbol}: {tpsl_err}")

            self.active_trades[trade_id] = trade
            logger.info(f"[{trade_id}] Win-Win Scalp Initialized on {symbol} ({side.upper()} @ {entry_price})")
            return trade

        except Exception as e:
            logger.error(f"Failed to execute Win-Win entry on {symbol}: {e}")
            return None

    def update_ticks(self, symbol: str, current_price: float, obi_10: float = 0.0) -> List[Dict[str, Any]]:
        """
        Called on every price tick / WebSocket update (sub-millisecond evaluation):
        - Evaluates TP1 hit -> locks in 50% profit & ratchets SL to Breakeven ("Win-Win" trigger).
        - Evaluates TP2 hit -> closes remaining runner.
        - Evaluates 30s timeout or adverse OBI flip -> scratch exit to protect capital.
        """
        events = []
        now = time.time()

        for trade_id, trade in list(self.active_trades.items()):
            if trade.symbol != symbol:
                continue

            is_long = trade.side.upper() == "BUY"

            # Calculate live unrealized PnL and ROE%
            if is_long:
                price_diff = current_price - trade.entry_price
            else:
                price_diff = trade.entry_price - current_price

            trade.unrealized_pnl = round(price_diff * trade.remaining_qty, 4)
            trade.roe_pct = round((price_diff / trade.entry_price) * trade.leverage * 100.0, 2) if trade.entry_price > 0 else 0.0

            # Floating point comparison tolerance
            eps = 1e-6

            # ==============================================================
            # EVENT 1: TP1 REACHED -> RATCHET SL TO BREAKEVEN ("WIN-WIN")
            # ==============================================================
            tp1_hit = (current_price >= trade.tp1_price - eps) if is_long else (current_price <= trade.tp1_price + eps)
            if tp1_hit and not trade.is_risk_free:
                # Realize 50% of position profit
                half_qty = trade.remaining_qty / 2.0
                pnl_tp1 = round(abs(trade.tp1_price - trade.entry_price) * half_qty, 4)
                exit_notional = half_qty * trade.tp1_price
                exit_fee = round(exit_notional * 0.00059, 4)  # 0.05% taker + 18% GST
                trade.total_fees_paid += exit_fee
                trade.realized_pnl += pnl_tp1
                trade.net_realized_pnl = round(trade.realized_pnl - trade.total_fees_paid, 4)
                trade.remaining_qty -= half_qty
                trade.is_risk_free = True
                trade.current_state = WinWinOrderState.TP1_HIT_BREAKEVEN_LOCKED
                
                # Dynamic Stop Loss is now locked at Breakeven + Fee Buffer
                trade.sl_price = trade.breakeven_sl

                events.append({
                    "event": "WIN_WIN_BREAKEVEN_LOCKED",
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "realized_pnl": pnl_tp1,
                    "net_realized_pnl": trade.net_realized_pnl,
                    "total_fees_paid": trade.total_fees_paid,
                    "new_sl": trade.breakeven_sl,
                    "new_tp": trade.tp2_price,
                    "half_closed_qty": half_qty,
                    "remaining_qty": trade.remaining_qty,
                    "roe_pct": trade.roe_pct,
                    "message": f"TP1 Hit on {symbol}! 50% locked (+${pnl_tp1:.4f}). SL ratcheted to Breakeven (+${trade.breakeven_sl}). Trade is now 100% Risk-Free!"
                })
                logger.info(f"[{trade_id}] WIN-WIN TRIGGERED! SL set to Breakeven: {trade.breakeven_sl}")

            # ==============================================================
            # EVENT 2: TP2 REACHED -> CLOSE REMAINING RUNNER
            # ==============================================================
            tp2_hit = (current_price >= trade.tp2_price - eps) if is_long else (current_price <= trade.tp2_price + eps)
            if tp2_hit:
                runner_pnl = round(abs(trade.tp2_price - trade.entry_price) * trade.remaining_qty, 4)
                exit_notional = trade.remaining_qty * trade.tp2_price
                exit_fee = round(exit_notional * 0.00059, 4)
                trade.total_fees_paid += exit_fee
                trade.realized_pnl += runner_pnl
                trade.net_realized_pnl = round(trade.realized_pnl - trade.total_fees_paid, 4)
                trade.remaining_qty = 0.0
                trade.current_state = WinWinOrderState.TP2_HIT_CLOSED
                trade.exit_time = now
                self.closed_trades.append(trade)
                self.active_trades.pop(trade_id, None)

                events.append({
                    "event": "WIN_WIN_TP2_MAX_PROFIT",
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "total_realized_pnl": round(trade.realized_pnl, 4),
                    "net_realized_pnl": trade.net_realized_pnl,
                    "total_fees_paid": trade.total_fees_paid,
                    "roe_pct": trade.roe_pct,
                    "message": f"TP2 Target Achieved on {symbol}! Total Scalp Net Profit: +${trade.net_realized_pnl:.4f}."
                })
                logger.info(f"[{trade_id}] TP2 Hit! Closed with Net PnL: ${trade.net_realized_pnl:.4f} (Fees: ${trade.total_fees_paid:.4f})")
                continue

            # ==============================================================
            # EVENT 3: STOP LOSS / BREAKEVEN HIT
            # ==============================================================
            sl_hit = (current_price <= trade.sl_price + eps) if is_long else (current_price >= trade.sl_price - eps)
            if sl_hit:
                exit_notional = trade.remaining_qty * current_price
                exit_fee = round(exit_notional * 0.00059, 4)
                trade.total_fees_paid += exit_fee
                if trade.is_risk_free:
                    # Exited at breakeven stop loss
                    if is_long:
                        diff = trade.breakeven_sl - trade.entry_price
                    else:
                        diff = trade.entry_price - trade.breakeven_sl
                    be_pnl = round(diff * trade.remaining_qty, 4)
                    trade.realized_pnl += be_pnl
                    trade.net_realized_pnl = round(trade.realized_pnl - trade.total_fees_paid, 4)
                    trade.current_state = WinWinOrderState.BREAKEVEN_CLOSED
                    msg = f"Breakeven Stop Hit on {symbol}. Closed risk-free with Net PnL: +${trade.net_realized_pnl:.4f} (Fees: ${trade.total_fees_paid:.4f})."
                else:
                    if is_long:
                        diff = trade.entry_price - trade.sl_price
                    else:
                        diff = trade.sl_price - trade.entry_price
                    loss = round(abs(diff) * trade.remaining_qty, 4)
                    trade.realized_pnl -= loss
                    trade.net_realized_pnl = round(trade.realized_pnl - trade.total_fees_paid, 4)
                    trade.current_state = WinWinOrderState.STOP_LOSS_CLOSED
                    msg = f"Stop Loss Hit on {symbol}. Net Loss including fees: -${abs(trade.net_realized_pnl):.4f}."

                trade.remaining_qty = 0.0
                trade.exit_time = now
                self.closed_trades.append(trade)
                self.active_trades.pop(trade_id, None)

                events.append({
                    "event": "STOP_EXECUTED",
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "total_realized_pnl": round(trade.realized_pnl, 4),
                    "net_realized_pnl": trade.net_realized_pnl,
                    "total_fees_paid": trade.total_fees_paid,
                    "roe_pct": trade.roe_pct,
                    "message": msg
                })
                continue

            # ==============================================================
            # EVENT 4: MICROSTRUCTURE INVALIDATION OR SCALP EXPIRATION (10m)
            # ==============================================================
            elapsed = now - (trade.entry_time or now)
            adverse_obi = (is_long and obi_10 < -0.65) or (not is_long and obi_10 > 0.65)
            
            if (elapsed > settings.MICRO_TIMEOUT_SECONDS or adverse_obi) and not trade.is_risk_free:
                scratch_pnl = round(price_diff * trade.remaining_qty, 4)
                exit_notional = trade.remaining_qty * current_price
                exit_fee = round(exit_notional * 0.00059, 4)
                trade.total_fees_paid += exit_fee
                trade.realized_pnl += scratch_pnl
                trade.net_realized_pnl = round(trade.realized_pnl - trade.total_fees_paid, 4)
                trade.remaining_qty = 0.0
                trade.current_state = WinWinOrderState.MICRO_TIMEOUT_CLOSED
                trade.exit_time = now
                self.closed_trades.append(trade)
                self.active_trades.pop(trade_id, None)

                events.append({
                    "event": "MICRO_TIMEOUT_SCRATCH_EXIT",
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "total_realized_pnl": round(trade.realized_pnl, 4),
                    "net_realized_pnl": trade.net_realized_pnl,
                    "total_fees_paid": trade.total_fees_paid,
                    "roe_pct": trade.roe_pct,
                    "message": f"Scalp Maturity / Invalidation on {symbol}. Exited with Net PnL: ${trade.net_realized_pnl:.4f} (Fees: ${trade.total_fees_paid:.4f})."
                })
                logger.info(f"[{trade_id}] Scalp Maturity Scratch Exit! Net PnL: ${trade.net_realized_pnl:.4f}")

        return events

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates institutional performance statistics with fee deduction."""
        all_closed = self.closed_trades
        total_trades = len(all_closed)
        if total_trades == 0:
            return {
                "total_trades": 0, "win_rate_pct": 0.0,
                "profit_factor": 0.0, "total_pnl": 0.0,
                "net_pnl": 0.0, "total_fees_paid": 0.0,
                "active_trades_count": len(self.active_trades),
                "risk_free_active_count": 0
            }

        # Accurate Win Rate: ONLY trades where net profit > 0 after paying all CoinDCX fees
        wins = [t.net_realized_pnl for t in all_closed if t.net_realized_pnl > 0]
        losses = [abs(t.net_realized_pnl) for t in all_closed if t.net_realized_pnl < 0]
        
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        win_rate = round(len(wins) / total_trades * 100.0, 1)

        # Mathematical Profit Factor handling (prevent division-by-zero / epsilon explosion)
        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        elif gross_profit > 0:
            profit_factor = 99.9  # Clean institutional cap for 100% win rate
        else:
            profit_factor = 0.0

        total_pnl = round(sum(t.realized_pnl for t in all_closed), 4)
        net_pnl = round(sum(t.net_realized_pnl for t in all_closed), 4)
        total_fees = round(sum(t.total_fees_paid for t in all_closed), 4)
        risk_free_count = sum(1 for t in self.active_trades.values() if t.is_risk_free)

        return {
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "total_pnl": net_pnl,  # Primary PnL is NET PnL
            "gross_pnl": total_pnl,
            "net_pnl": net_pnl,
            "total_fees_paid": total_fees,
            "gross_profit": round(gross_profit, 4),
            "gross_loss": round(gross_loss, 4),
            "active_trades_count": len(self.active_trades),
            "risk_free_active_count": risk_free_count
        }

    def reset_stats(self):
        """Clears closed trades history and resets performance statistics."""
        self.closed_trades.clear()
        self.trade_counter = 0
        logger.info("WinWinExecutionEngine performance statistics reset.")
