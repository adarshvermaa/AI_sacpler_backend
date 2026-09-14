"""
AlphaScalper - Institutional Risk Management & Circuit Breaker Engine
Key Responsibilities:
1. Daily Loss Limit Circuit Breaker (Auto-Halt on -5% Drawdown)
2. Fractional Kelly Criterion Position Sizing based on AI Confidence
3. CoinDCX Dynamic Leverage Limit Validator
4. Margin Ratio & Liquidation Buffer Guard
5. Emergency Kill-Switch
"""

import logging
from typing import Dict, Any, Optional

from config import settings

logger = logging.getLogger("AlphaScalper.RiskManager")


class AlphaRiskManager:
    def __init__(self, initial_capital: float = 10000.0, is_live_mode: bool = False):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_pnl = 0.0
        self.is_circuit_breaker_tripped = False
        self.is_kill_switch_active = False
        self.is_live_mode = is_live_mode
        self.min_order_notional = 6.0  # CoinDCX futures minimum notional in USDT
        min_margin_default = self.min_order_notional / settings.DEFAULT_LEVERAGE
        self.is_balance_sufficient = not is_live_mode or (initial_capital >= min_margin_default)
        
        # Anti-Churn & Re-entry Structure Cooldowns
        self.symbol_cooldowns: Dict[str, float] = {}  # symbol -> last_exit_timestamp
        self.symbol_cooldown_seconds: float = 480.0    # 8 minutes (allows 1-2 5m candles to form fresh structure)
        self.last_trade_execution_time: float = 0.0
        self.min_inter_trade_seconds: float = 45.0     # Minimum spacing between any orders
        
        # Real CoinDCX INR Collateral Equity Circuit Breaker
        self.starting_session_inr_equity: Optional[float] = None
        self.max_session_drawdown_inr: float = 100.0   # Strict ₹100 INR drawdown cap (~$1.15 USDT)

    def record_trade_entry(self, symbol: str):
        """Records timestamp of an open order to prevent burst orders."""
        import time
        self.last_trade_execution_time = time.time()

    def record_trade_exit(self, symbol: str):
        """Enforces a mandatory structure cooldown on the specific symbol."""
        import time
        self.symbol_cooldowns[symbol] = time.time()
        logger.info(f"[COOLDOWN] {symbol} locked on structure cooldown for {int(self.symbol_cooldown_seconds)}s.")

    def sync_live_inr_equity(self, total_inr: float, inr_usd_rate: float = 87.5):
        """
        Synchronizes real CoinDCX INR futures equity.
        Accurately calculates real daily PnL and trips circuit breaker on real account loss.
        """
        if total_inr <= 0.0:
            return
        if self.starting_session_inr_equity is None:
            self.starting_session_inr_equity = total_inr

        delta_inr = total_inr - self.starting_session_inr_equity
        delta_usdt = delta_inr / inr_usd_rate
        self.daily_pnl = round(delta_usdt, 2)
        self.current_capital = round(total_inr / inr_usd_rate, 2)

        # Check strict INR drawdown circuit breaker
        if delta_inr <= -self.max_session_drawdown_inr:
            self.is_circuit_breaker_tripped = True
            logger.critical(
                f"[CIRCUIT BREAKER TRIPPED] Live CoinDCX equity loss (-₹{abs(delta_inr):.2f} INR) "
                f"exceeded maximum allowed loss of ₹{self.max_session_drawdown_inr:.2f} INR! ALL TRADING SUSPENDED!"
            )

    def sync_live_balance(self, usable_balance_usdt: float, is_live: bool = True):
        """
        Synchronizes real wallet balance from CoinDCX API and updates risk limits.
        """
        self.is_live_mode = is_live
        if is_live:
            self.current_capital = usable_balance_usdt
            self.initial_capital = max(usable_balance_usdt, 0.01)
            min_margin_default = self.min_order_notional / settings.DEFAULT_LEVERAGE
            self.is_balance_sufficient = usable_balance_usdt >= min_margin_default
            logger.info(
                f"[RISK MANAGER] Synced CoinDCX Live Balance: ${usable_balance_usdt:.4f} USDT | "
                f"Sufficient for Live Execution: {self.is_balance_sufficient} (Min Margin for 10x: ${min_margin_default:.2f} USDT)"
            )

    def can_open_new_trade(self, requested_allocation_usdt: float, leverage: float = 10.0, symbol: Optional[str] = None) -> tuple[bool, str]:
        """Validates if a new trade can be placed under current risk rules."""
        import time
        if self.is_kill_switch_active:
            return False, "Emergency Kill-Switch is ACTIVE. Trading halted."

        if self.is_circuit_breaker_tripped:
            return False, "Daily Drawdown Circuit Breaker TRIPPED. Trading suspended."

        # Anti-churn symbol cooldown check
        if symbol and symbol in self.symbol_cooldowns:
            elapsed = time.time() - self.symbol_cooldowns[symbol]
            if elapsed < self.symbol_cooldown_seconds:
                remaining = int(self.symbol_cooldown_seconds - elapsed)
                return False, f"[COOLDOWN] {symbol} in structure cooldown ({remaining}s remaining). Allowing 5m candle formation."

        # Global pacing check to prevent rapid burst executions
        time_since_last = time.time() - self.last_trade_execution_time
        if self.last_trade_execution_time > 0 and time_since_last < self.min_inter_trade_seconds:
            remaining = int(self.min_inter_trade_seconds - time_since_last)
            return False, f"[PACING] Global scalp pacing active ({remaining}s remaining to prevent burst execution)."

        if requested_allocation_usdt <= 0:
            return False, "Kelly position sizing returned 0 (no positive statistical edge or insufficient capital)."

        req_margin = requested_allocation_usdt / max(1.0, leverage)
        min_margin_needed = self.min_order_notional / max(1.0, leverage)

        # Balance Guard for LIVE trading
        if self.is_live_mode:
            if not self.is_balance_sufficient or self.current_capital < min_margin_needed:
                return (
                    False,
                    f"[BALANCE GUARD] Account balance (${self.current_capital:.4f} USDT) is below minimum margin (${min_margin_needed:.2f} USDT) required for CoinDCX minimum contract (${self.min_order_notional:.2f} USDT). Live orders held."
                )
            if requested_allocation_usdt < self.min_order_notional:
                return (
                    False,
                    f"[ORDER GUARD] Requested trade notional (${requested_allocation_usdt:.2f} USDT) is below CoinDCX minimum required notional (${self.min_order_notional:.2f} USDT)."
                )
            if req_margin > (self.current_capital * 0.85):
                return (
                    False,
                    f"Required trade margin (${req_margin:.2f} USDT) exceeds 85% of available balance (${self.current_capital:.2f} USDT)."
                )

        # Check daily drawdown limit
        max_loss_usdt = self.initial_capital * (settings.MAX_DAILY_DRAWDOWN_PERCENT / 100.0)
        if self.daily_pnl <= -max_loss_usdt and self.initial_capital > 10.0:
            self.is_circuit_breaker_tripped = True
            logger.critical(f"Circuit Breaker Tripped! Daily loss -${abs(self.daily_pnl)} exceeded limit -${max_loss_usdt}")
            return False, f"Daily loss limit (-${max_loss_usdt}) breached."

        # In paper mode, check 50% max margin allocation
        if not self.is_live_mode and req_margin > (self.current_capital * 0.5):
            return False, "Required trade margin exceeds 50% of available simulation capital."

        return True, "Approved"

    def calculate_kelly_position_size(
        self,
        ai_confidence: float,
        reward_ratio: float = 2.0,  # TP / SL ratio (e.g. 0.85% / 0.45%)
        fractional_factor: float = 0.25,  # Quarter-Kelly for conservative institutional risk
        leverage: float = 10.0
    ) -> float:
        """
        Fractional Kelly Criterion for Leveraged Futures:
        f* = (p * (b + 1) - 1) / b
        Where:
        p = probability of win (from AI confidence, e.g. 0.72)
        b = payoff ratio (reward / risk)
        
        Leveraged Futures Math:
        1. If f* <= 0 (negative edge), return 0.0 (prohibit trade).
        2. Safe margin fraction = clamp(f* * fractional_factor, 0.05, 0.25).
        3. Allocated margin = current_capital * safe_fraction.
        4. Target notional = Allocated margin * leverage.
        5. Enforce CoinDCX minimum notional ($6.00 USDT) and max 85% margin floor.
        Returns: target_notional_usdt
        """
        p = max(0.01, min(0.99, ai_confidence / 100.0))
        b = max(0.1, reward_ratio)
        
        # Kelly formula
        kelly_fraction = (p * (b + 1.0) - 1.0) / b
        
        # Zero allocation on non-positive edge
        if kelly_fraction <= 0:
            return 0.0
        
        # Scale by fractional factor (Quarter-Kelly: 5% - 25% of margin)
        safe_fraction = min(0.25, max(0.05, kelly_fraction * fractional_factor))
        
        # In leveraged futures: Margin allocated = current_capital * safe_fraction, Notional = Margin * Leverage
        allocated_margin = round(self.current_capital * safe_fraction, 4)
        target_notional = round(allocated_margin * leverage, 2)
        min_margin_req = self.min_order_notional / max(1.0, leverage)
        
        if self.is_live_mode:
            # Check if wallet can support at least CoinDCX min notional
            if target_notional < self.min_order_notional:
                if min_margin_req <= self.current_capital * 0.85:
                    target_notional = self.min_order_notional
                else:
                    return 0.0  # Cannot safely afford even the minimum contract
            
            # Cap notional so required margin never exceeds 85% of available capital
            max_notional = round(self.current_capital * 0.85 * leverage, 2)
            target_notional = min(target_notional, max_notional)
            return target_notional
        else:
            # Paper trading simulation
            target_notional = max(50.0, target_notional)
            max_notional = round(self.current_capital * 0.50 * leverage, 2)
            return min(target_notional, max_notional)

    def calculate_volatility_parity_position_size(
        self,
        entry_price: float,
        sl_price: float,
        risk_per_trade_pct: float = 1.5,
        leverage: float = 10.0
    ) -> float:
        """
        Volatility Parity Position Sizing (Fixed Dollar Risk per Trade):
        Ensures that if Stop Loss is hit, the exact predetermined dollar amount is lost,
        regardless of whether the asset is a high-volatility meme coin or low-volatility Bitcoin.
        
        Dollar Risk = Current Capital * (risk_per_trade_pct / 100)
        Stop Distance % = abs(entry_price - sl_price) / entry_price
        Target Notional = Dollar Risk / max(Stop Distance %, 0.003)
        """
        if entry_price <= 0 or sl_price <= 0:
            return self.min_order_notional
        
        dollar_risk = self.current_capital * (risk_per_trade_pct / 100.0)
        stop_dist_pct = abs(entry_price - sl_price) / entry_price
        stop_dist_pct = max(0.003, stop_dist_pct)
        
        target_notional = round(dollar_risk / stop_dist_pct, 2)
        min_margin_req = self.min_order_notional / max(1.0, leverage)
        
        if self.is_live_mode:
            if target_notional < self.min_order_notional:
                if min_margin_req <= self.current_capital * 0.85:
                    target_notional = self.min_order_notional
                else:
                    return 0.0
            max_notional = round(self.current_capital * 0.85 * leverage, 2)
            return min(target_notional, max_notional)
        else:
            return max(self.min_order_notional, min(target_notional, round(self.current_capital * 0.50 * leverage, 2)))

    def validate_leverage(self, requested_leverage: float, notional_value: float) -> float:
        """Enforces CoinDCX tiered maximum leverage based on position notional size."""
        # Tiered limits as per CoinDCX documentation:
        # < 50k: up to 25x, 50k-100k: 20x, 100k-500k: 15x, 500k-1M: 10x
        if notional_value > 500000.0:
            max_allowed = 10.0
        elif notional_value > 100000.0:
            max_allowed = 15.0
        elif notional_value > 50000.0:
            max_allowed = 20.0
        else:
            max_allowed = settings.MAX_LEVERAGE

        return min(requested_leverage, max_allowed)

    def update_daily_pnl(self, realized_delta: float):
        """Updates internal capital and daily PnL tracker."""
        self.daily_pnl += realized_delta
        self.current_capital += realized_delta
        if self.daily_pnl <= -(self.initial_capital * (settings.MAX_DAILY_DRAWDOWN_PERCENT / 100.0)):
            self.is_circuit_breaker_tripped = True

    def reset_daily_pnl(self):
        """Resets daily PnL tracker and circuit breaker state."""
        self.daily_pnl = 0.0
        self.is_circuit_breaker_tripped = False
        logger.info("Risk manager daily PnL reset to 0.0.")

    def trigger_emergency_kill_switch(self):
        """Activates immediate halt on all execution."""
        self.is_kill_switch_active = True
        logger.warning("EMERGENCY KILL SWITCH ACTIVATED BY USER OR RISK ENGINE!")

    def reset_kill_switch(self):
        """Resets the kill-switch and circuit breaker after manual user review."""
        self.is_kill_switch_active = False
        self.is_circuit_breaker_tripped = False
        logger.info("Kill switch reset. System ready.")

    def compute_dynamic_order_allocation(
        self,
        total_equity_usdt: float,
        free_margin_usdt: float,
        current_atr_pct: float = 0.005,
        ai_confidence: float = 75.0,
        inr_usd_rate: float = 87.5
    ) -> Dict[str, Any]:
        """
        Dynamically calculates concurrent orders, leverage, margin exposure,
        and capital protection floors based on total account balance and volatility.
        """
        eq = max(total_equity_usdt, 0.01)
        free = max(free_margin_usdt, 0.0)
        
        # 1. Total Account Equity in INR and USDT
        total_inr = round(eq * inr_usd_rate, 2)
        free_inr = round(free * inr_usd_rate, 2)

        # 2. Capital Protection Floor (15% Reserve Buffer, 85% Max Exposure)
        reserve_buffer_pct = 0.15
        max_exposure_ratio = 0.85  # 85% allocatable capital, preserving 15% buffer
        max_margin_budget_usdt = round(eq * max_exposure_ratio, 2)
        max_margin_budget_inr = round(max_margin_budget_usdt * inr_usd_rate, 2)

        if eq < 20.0:
            account_tier = "MICRO_STARTER"
        elif eq < 50.0:
            account_tier = "GROWTH_TIER_1"
        elif eq < 100.0:
            account_tier = "GROWTH_TIER_2"
        elif eq < 250.0:
            account_tier = "PRO_TRADER"
        else:
            account_tier = "INSTITUTIONAL"

        # 3. Dynamic Volatility-Scaled Leverage (ATR-based)
        atr_clamped = max(0.003, min(0.03, current_atr_pct))
        dynamic_leverage = max(5.0, min(10.0, float(int(0.045 / atr_clamped))))

        # 4. Single Order Notional and Margin Requirement
        order_notional_usdt = self.min_order_notional  # $6.00 USDT
        order_notional_inr = round(order_notional_usdt * inr_usd_rate, 2)
        margin_per_order_usdt = round(order_notional_usdt / dynamic_leverage, 4)
        margin_per_order_inr = round(margin_per_order_usdt * inr_usd_rate, 2)

        # 5. Dynamic Order Count Allocation Formula (up to 10 concurrent orders)
        effective_budget_usdt = min(max_margin_budget_usdt, free)
        calculated_orders = int(effective_budget_usdt // margin_per_order_usdt) if margin_per_order_usdt > 0 else 0
        dynamic_orders_count = max(1 if free >= margin_per_order_usdt else 0, min(10, calculated_orders))
        max_concurrent_cap = 10

        # 6. Dynamic Risk per Trade (Fractional Kelly)
        p = min(0.95, max(0.50, ai_confidence / 100.0))
        b = 1.88  # TP / SL payoff ratio
        kelly_full = (p * (b + 1.0) - 1.0) / b
        safe_risk_pct = max(0.005, min(0.015, kelly_full * 0.25))  # Quarter-Kelly (0.5% - 1.5%)
        
        max_risk_dollars = round(eq * safe_risk_pct, 4)
        max_risk_inr = round(max_risk_dollars * inr_usd_rate, 2)
        
        # Stop-Loss Distance (Hard Stop -0.45%)
        stop_loss_pct = 0.0045
        actual_loss_at_sl_usdt = round(order_notional_usdt * stop_loss_pct, 4)
        actual_loss_at_sl_inr = round(actual_loss_at_sl_usdt * inr_usd_rate, 2)

        return {
            "account_tier": account_tier,
            "total_equity_usdt": round(eq, 4),
            "total_equity_inr": total_inr,
            "free_margin_usdt": round(free, 4),
            "free_margin_inr": free_inr,
            "inr_usd_rate": inr_usd_rate,
            "max_exposure_ratio": max_exposure_ratio,
            "reserve_floor_pct": round((1.0 - max_exposure_ratio) * 100, 1),
            "max_margin_budget_usdt": max_margin_budget_usdt,
            "max_margin_budget_inr": max_margin_budget_inr,
            "dynamic_leverage": dynamic_leverage,
            "order_notional_usdt": order_notional_usdt,
            "order_notional_inr": order_notional_inr,
            "margin_per_order_usdt": margin_per_order_usdt,
            "margin_per_order_inr": margin_per_order_inr,
            "max_concurrent_orders": dynamic_orders_count,
            "max_concurrent_cap": max_concurrent_cap,
            "safe_risk_pct": round(safe_risk_pct * 100, 2),
            "max_risk_dollars": max_risk_dollars,
            "max_risk_inr": max_risk_inr,
            "stop_loss_pct": round(stop_loss_pct * 100, 2),
            "actual_loss_at_sl_usdt": actual_loss_at_sl_usdt,
            "actual_loss_at_sl_inr": actual_loss_at_sl_inr,
            "can_trade": dynamic_orders_count > 0 and free >= margin_per_order_usdt
        }


# Convenience alias
DynamicRiskManager = AlphaRiskManager

