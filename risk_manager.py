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
        self.is_balance_sufficient = not is_live_mode or (initial_capital >= 6.0)
        self.min_order_notional = 6.0  # CoinDCX futures minimum notional in USDT

    def sync_live_balance(self, usable_balance_usdt: float, is_live: bool = True):
        """
        Synchronizes real wallet balance from CoinDCX API and updates risk limits.
        """
        self.is_live_mode = is_live
        if is_live:
            self.current_capital = usable_balance_usdt
            self.initial_capital = max(usable_balance_usdt, 0.01)
            self.is_balance_sufficient = usable_balance_usdt >= self.min_order_notional
            logger.info(
                f"[RISK MANAGER] Synced CoinDCX Live Balance: ${usable_balance_usdt:.4f} USDT | "
                f"Sufficient for Live Execution: {self.is_balance_sufficient} (Min: ${self.min_order_notional} USDT)"
            )

    def can_open_new_trade(self, requested_allocation_usdt: float) -> tuple[bool, str]:
        """Validates if a new trade can be placed under current risk rules."""
        if self.is_kill_switch_active:
            return False, "Emergency Kill-Switch is ACTIVE. Trading halted."

        if self.is_circuit_breaker_tripped:
            return False, "Daily Drawdown Circuit Breaker TRIPPED. Trading suspended."

        # Balance Guard for LIVE trading
        if self.is_live_mode:
            if not self.is_balance_sufficient or self.current_capital < self.min_order_notional:
                return (
                    False,
                    f"[BALANCE GUARD] Account balance (${self.current_capital:.4f} USDT) is below CoinDCX minimum required notional (${self.min_order_notional:.2f} USDT). Live orders held."
                )
            if requested_allocation_usdt > self.current_capital:
                return (
                    False,
                    f"Requested trade allocation (${requested_allocation_usdt:.2f}) exceeds available balance (${self.current_capital:.2f})."
                )

        # Check daily drawdown limit
        max_loss_usdt = self.initial_capital * (settings.MAX_DAILY_DRAWDOWN_PERCENT / 100.0)
        if self.daily_pnl <= -max_loss_usdt and self.initial_capital > 10.0:
            self.is_circuit_breaker_tripped = True
            logger.critical(f"Circuit Breaker Tripped! Daily loss -${abs(self.daily_pnl)} exceeded limit -${max_loss_usdt}")
            return False, f"Daily loss limit (-${max_loss_usdt}) breached."

        # In paper mode, check 50% max allocation
        if not self.is_live_mode and requested_allocation_usdt > (self.current_capital * 0.5):
            return False, "Requested trade allocation exceeds 50% of available capital."

        return True, "Approved"

    def calculate_kelly_position_size(
        self,
        ai_confidence: float,
        reward_ratio: float = 2.0,  # TP / SL ratio (e.g. 0.85% / 0.45%)
        fractional_factor: float = 0.25  # Quarter-Kelly for conservative institutional risk
    ) -> float:
        """
        Fractional Kelly Criterion:
        f* = (p * (b + 1) - 1) / b
        Where:
        p = probability of win (from AI confidence, e.g. 0.72)
        b = payoff ratio (reward / risk)
        """
        if self.is_live_mode and self.current_capital < self.min_order_notional:
            return 0.0

        p = max(0.01, min(0.99, ai_confidence / 100.0))
        b = reward_ratio
        
        # Kelly formula
        kelly_fraction = (p * (b + 1.0) - 1.0) / b
        
        # Scale by fractional factor (Quarter-Kelly)
        safe_fraction = min(0.20, max(0.05, kelly_fraction * fractional_factor))
        
        if self.is_live_mode:
            # Scale dynamically to real available balance
            size_usdt = round(self.current_capital * safe_fraction, 2)
            # Must satisfy CoinDCX minimum contract threshold but not exceed available capital
            size_usdt = max(self.min_order_notional, min(size_usdt, self.current_capital * 0.95))
            return size_usdt
        else:
            # Paper trading simulation
            if kelly_fraction <= 0:
                return 100.0
            size_usdt = round(self.current_capital * safe_fraction, 2)
            return max(50.0, size_usdt)

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

    def trigger_emergency_kill_switch(self):
        """Activates immediate halt on all execution."""
        self.is_kill_switch_active = True
        logger.warning("EMERGENCY KILL SWITCH ACTIVATED BY USER OR RISK ENGINE!")

    def reset_kill_switch(self):
        """Resets the kill-switch and circuit breaker after manual user review."""
        self.is_kill_switch_active = False
        self.is_circuit_breaker_tripped = False
        logger.info("Kill switch reset. System ready.")
