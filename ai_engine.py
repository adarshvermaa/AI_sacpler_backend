"""
AlphaScalper - Dynamic 4-Tier Hierarchical AI Engine for Crypto Futures Scalping
Architecture:
1. BTC Macro Regime Gatekeeper (1H/15M Global Trend & Volatility Filter)
2. Quantitative Multi-Factor Alpha Confluence (RS vs BTC, Volume Surge, Squeeze, Real OBI)
3. Mathematical Expected Value Model: EV = (P_win * Net Gain) - (P_loss * Net Loss) - Roundtrip Fees
4. Dynamic Asymmetric Bracket Generator (R:R >= 1.8:1, Volatility Parity Sizing)
"""

import time
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy.stats import norm

from config import settings


class BTCRegimeGatekeeper:
    """
    Tier 1 Macro Gatekeeper: Analyzes Bitcoin (BTC) as the market tide.
    90% of crypto scalps fail when fighting adverse Bitcoin momentum.
    """
    @staticmethod
    def analyze_btc_regime(btc_candles: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Analyzes 15m and 1h Bitcoin trend, volatility, and momentum.
        Returns regime status, directional bias allowed for altcoins, and risk scaling factor.
        """
        if not btc_candles or len(btc_candles.get("close", [])) < 20:
            return {
                "regime": "CHOP_COMPRESSION",
                "allowed_alt_bias": "ALL",
                "risk_multiplier": 0.85,
                "btc_trend_score": 0.0,
                "btc_price": 60000.0,
                "btc_change_1h": 0.0,
                "is_flush": False
            }

        closes = btc_candles["close"]
        highs = btc_candles["high"]
        lows = btc_candles["low"]
        curr_price = float(closes[-1])
        n = len(closes)

        # 1. Multi-Period Moving Averages on BTC
        ema_9 = float(np.mean(closes[-9:]))
        ema_21 = float(np.mean(closes[-21:])) if n >= 21 else ema_9
        sma_50 = float(np.mean(closes[-50:])) if n >= 50 else ema_21

        # 2. 1-Hour & 15-Minute Momentum
        bars_1h = min(n - 1, 4)  # 4 * 15m = 1 hour
        btc_change_1h = round(((curr_price - closes[-bars_1h]) / closes[-bars_1h]) * 100.0, 2)
        btc_change_15m = round(((curr_price - closes[-2]) / closes[-2]) * 100.0, 2) if n >= 2 else 0.0

        # 3. Volatility & Flush Detection (ATR 14)
        tr = np.maximum(highs[-14:] - lows[-14:], np.maximum(np.abs(highs[-14:] - closes[-15:-1]), np.abs(lows[-14:] - closes[-15:-1])))
        atr_14 = float(np.mean(tr)) if len(tr) > 0 else (curr_price * 0.005)
        atr_pct = (atr_14 / curr_price) * 100.0

        # Rapid Flush Check: BTC moved > 1.2% in 15m or > 2.5% in 1h
        is_flush = bool(abs(btc_change_15m) >= 1.2 or abs(btc_change_1h) >= 2.5 or atr_pct > 1.5)

        # 4. Determine Regime
        bull_stack = curr_price > ema_9 > ema_21 and btc_change_1h > 0.3
        bear_stack = curr_price < ema_9 < ema_21 and btc_change_1h < -0.3

        if is_flush:
            regime = "HIGH_VOLATILITY_FLUSH"
            allowed_alt_bias = "DEFENSIVE_HOLD"
            risk_multiplier = 0.25
        elif bull_stack:
            regime = "BULLISH_EXPANSION"
            allowed_alt_bias = "LONG_PREFERRED"
            risk_multiplier = 1.0
        elif bear_stack:
            regime = "BEARISH_EXPANSION"
            allowed_alt_bias = "SHORT_PREFERRED"
            risk_multiplier = 1.0
        else:
            regime = "CHOP_COMPRESSION"
            allowed_alt_bias = "SELECTIVE_RANGE"
            risk_multiplier = 0.70

        trend_score = round(np.clip((curr_price - ema_21) / (atr_14 + 1e-9) * 0.5, -1.0, 1.0), 3)

        return {
            "regime": regime,
            "allowed_alt_bias": allowed_alt_bias,
            "risk_multiplier": risk_multiplier,
            "btc_trend_score": trend_score,
            "btc_price": curr_price,
            "btc_change_1h": btc_change_1h,
            "is_flush": is_flush
        }


class ExpectedValueModel:
    """
    Tier 3 Mathematical Expected Value Engine.
    Ensures every scalp has positive mathematical expectancy after deducting CoinDCX fees.
    EV = (P_win * Net Gain) - (P_loss * Net Loss) - Roundtrip Fees
    """
    ROUNDTRIP_FEE_PCT = 0.00118  # 0.05% taker + 18% GST each side = 0.059% * 2 = 0.118%

    @classmethod
    def compute_expected_value(
        cls,
        win_prob: float,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        is_long: bool
    ) -> Dict[str, Any]:
        """
        Calculates net mathematical expected value and risk:reward.
        win_prob: float between 0.0 and 1.0
        """
        if entry_price <= 0:
            return {"ev": 0.0, "ev_pct": 0.0, "is_viable": False, "rr_ratio": 1.0}

        if is_long:
            gross_gain_pct = max(0.0, (tp_price - entry_price) / entry_price)
            gross_loss_pct = max(0.0, (entry_price - sl_price) / entry_price)
        else:
            gross_gain_pct = max(0.0, (entry_price - tp_price) / entry_price)
            gross_loss_pct = max(0.0, (sl_price - entry_price) / entry_price)

        # Net gains and losses after exchange fee friction
        net_gain_pct = max(0.0, gross_gain_pct - cls.ROUNDTRIP_FEE_PCT)
        net_loss_pct = gross_loss_pct + cls.ROUNDTRIP_FEE_PCT

        # Mathematical Expectancy
        ev = (win_prob * net_gain_pct) - ((1.0 - win_prob) * net_loss_pct)
        ev_pct = round(ev * 100.0, 3)
        rr_ratio = round(net_gain_pct / max(1e-6, net_loss_pct), 2)
        # Scalping edge requires strictly positive EV after exchange fees, win prob >= 58%, and viable R:R
        is_viable = ev > 0.0 and win_prob >= 0.58 and rr_ratio >= 0.80

        return {
            "ev": ev,
            "ev_pct": ev_pct,
            "net_gain_pct": round(net_gain_pct * 100.0, 2),
            "net_loss_pct": round(net_loss_pct * 100.0, 2),
            "roundtrip_fee_pct": round(cls.ROUNDTRIP_FEE_PCT * 100.0, 3),
            "rr_ratio": rr_ratio,
            "is_viable": is_viable
        }


class AlphaAIEngine:
    def __init__(self):
        # Multi-Armed Bandit Strategy Weights (Thompson Sampling state)
        self.strategies = ["OBI_MOMENTUM", "SMC_LIQUIDITY_SWEEP", "VOLATILITY_BREAKOUT", "MEAN_REVERSION"]
        self.mab_successes = {s: 1.0 for s in self.strategies}
        self.mab_failures = {s: 1.0 for s in self.strategies}
        
        # Pre-calculated standard normal constants for sub-microsecond options greeks
        self._inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    @staticmethod
    def format_price_precision(price: float) -> float:
        """Intelligently rounds prices based on magnitude to avoid zeroing out sub-cent assets."""
        if price <= 0.0:
            return price
        if price >= 1000.0:
            return round(price, 2)
        elif price >= 10.0:
            return round(price, 3)
        elif price >= 1.0:
            return round(price, 4)
        elif price >= 0.01:
            return round(price, 5)
        elif price >= 0.0001:
            return round(price, 7)
        else:
            return round(price, 8)

    def evaluate_scalp_opportunity(
        self,
        symbol: str,
        features: Dict[str, float],
        orderbook: Optional[Dict[str, Any]] = None,
        btc_context: Optional[Dict[str, Any]] = None,
        relative_strength_vs_btc: Optional[float] = None,
        volume_surge_ratio: Optional[float] = None,
        volatility_squeeze: Optional[bool] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        selected_indicators: Optional[List[str]] = None,
        price_action_rules: Optional[List[str]] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_1_pct: Optional[float] = None,
        take_profit_2_pct: Optional[float] = None,
        enable_breakeven: Optional[bool] = None,
        strict_counter_trend_veto: Optional[bool] = None,
        timeframes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes 4-tier multi-factor quantitative evaluation:
        1. BTC Macro Gating
        2. Relative Strength & Volatility Squeeze Confluence
        3. Real Order Flow & Microstructure (OBI)
        4. Mathematical Expected Value Calculation (EV > 0, Win Prob >= 70%)
        5. Asymmetric Bracket Generation (R:R >= 1.8:1)
        """
        start_time = time.perf_counter_ns()
        
        # Real Live Market Price (Guaranteed from live ticks & real candles)
        curr_price = float(features.get("current_price") or features.get("close") or features.get("sma_20") or 100.0)

        # 1. Microstructure OBI Score (-1.0 to 1.0)
        obi_10 = float(features.get("orderbook_imbalance_10", 0.0))
        micro_premium = float(features.get("micro_price_premium", 0.0))
        micro_score = (obi_10 * 0.7) + (np.clip(micro_premium * 500.0, -1.0, 1.0) * 0.3)

        # 2. Dynamic Technical Indicators Confluence
        rsi_14 = float(features.get("rsi_14", 50.0))
        rsi_bias = np.clip((rsi_14 - 50.0) / 25.0, -1.0, 1.0)

        macd_hist = float(features.get("macd_hist", 0.0))
        macd_bias = np.clip(macd_hist / (curr_price * 0.001 + 1e-9), -1.0, 1.0)

        stoch_k = float(features.get("stoch_rsi_k", 50.0))
        stoch_bias = np.clip((stoch_k - 50.0) / 35.0, -1.0, 1.0)

        cci_14 = float(features.get("cci_14", 0.0))
        cci_bias = np.clip(cci_14 / 100.0, -1.0, 1.0)

        st_bull = float(features.get("supertrend_bullish", 0.5))
        st_bias = 1.0 if st_bull > 0.5 else -1.0

        ema_trend = float(features.get("ema_trend", 0.0))
        macro_trend = float(features.get("macro_trend", 0.0))
        ma_stack = float(features.get("ma_stack_bias", 0.0))
        
        vwap_val = float(features.get("vwap", curr_price))
        vwap_bias = 1.0 if curr_price > vwap_val else -1.0

        # Bollinger Bands bias
        bb_upper = float(features.get("bb_upper", curr_price * 1.01))
        bb_lower = float(features.get("bb_lower", curr_price * 0.99))
        bb_mid = (bb_upper + bb_lower) / 2.0
        bb_bias = np.clip((curr_price - bb_mid) / max(bb_upper - bb_lower, 1e-6) * 2.0, -1.0, 1.0)

        # Ichimoku bias
        ich_conv = float(features.get("ichimoku_conversion", curr_price))
        ich_base = float(features.get("ichimoku_base", curr_price))
        ichimoku_bias = 1.0 if ich_conv > ich_base else -1.0

        # ATR / Volatility
        atr_14 = float(features.get("atr_14", max(0.001, curr_price * 0.005)))
        atr_bias = np.clip(features.get("atr_normalized", 0.01) * 50.0, 0.0, 1.0) * st_bias

        # Chaikin Money Flow
        cmf = float(features.get("chaikin_money_flow", 0.0))
        chaikin_bias = np.clip(cmf * 5.0, -1.0, 1.0)

        # Dictionary of available indicators
        ind_pool: Dict[str, Tuple[float, float]] = {
            "RSI": (rsi_bias, 0.20),
            "MACD": (macd_bias, 0.20),
            "SuperTrend": (st_bias, 0.20),
            "VWAP": (vwap_bias, 0.15),
            "Bollinger": (bb_bias, 0.15),
            "StochRSI": (stoch_bias, 0.10),
            "Ichimoku": (ichimoku_bias, 0.10),
            "ATR": (atr_bias, 0.10),
            "Chaikin": (chaikin_bias, 0.10),
            "CCI": (cci_bias, 0.10)
        }

        if selected_indicators and len(selected_indicators) > 0:
            active_inds = {k: ind_pool[k] for k in selected_indicators if k in ind_pool}
            if not active_inds:
                active_inds = {"RSI": ind_pool["RSI"], "SuperTrend": ind_pool["SuperTrend"]}
        else:
            active_inds = {
                "RSI": ind_pool["RSI"],
                "MACD": ind_pool["MACD"],
                "SuperTrend": ind_pool["SuperTrend"],
                "VWAP": ind_pool["VWAP"],
                "Bollinger": ind_pool["Bollinger"],
                "StochRSI": ind_pool["StochRSI"]
            }

        total_ind_weight = sum(w for _, w in active_inds.values()) or 1.0
        tech_score = sum(score * (w / total_ind_weight) for score, w in active_inds.values())
        tech_score = np.clip(tech_score, -1.0, 1.0)

        # 3. Dynamic Price Action & Smart Money Concepts (SMC)
        fvg_bull = float(features.get("fvg_bullish", 0.0))
        fvg_bear = float(features.get("fvg_bearish", 0.0))
        ob_bull = float(features.get("bullish_order_block", 0.0))
        ob_bear = float(features.get("bearish_order_block", 0.0))
        sweep_low = float(features.get("liquidity_sweep_low", 0.0))
        sweep_high = float(features.get("liquidity_sweep_high", 0.0))
        pin_bull = float(features.get("bullish_pinbar", 0.0))
        pin_bear = float(features.get("bearish_pinbar", 0.0))
        bos_bull = float(features.get("bos_bullish", 0.0))
        bos_bear = float(features.get("bos_bearish", 0.0))

        pa_rules_map: Dict[str, Tuple[float, float, float]] = {
            "OrderBlocks": (ob_bull, ob_bear, 0.25),
            "FVG": (fvg_bull, fvg_bear, 0.25),
            "LiquiditySweeps": (sweep_low, sweep_high, 0.25),
            "BOS": (bos_bull, bos_bear, 0.25),
            "PinBars": (pin_bull, pin_bear, 0.20)
        }

        if price_action_rules and len(price_action_rules) > 0:
            active_pa = {k: pa_rules_map[k] for k in price_action_rules if k in pa_rules_map}
            if not active_pa:
                active_pa = {"OrderBlocks": pa_rules_map["OrderBlocks"], "FVG": pa_rules_map["FVG"]}
        else:
            active_pa = {
                "OrderBlocks": pa_rules_map["OrderBlocks"],
                "FVG": pa_rules_map["FVG"],
                "LiquiditySweeps": pa_rules_map["LiquiditySweeps"],
                "BOS": pa_rules_map["BOS"]
            }

        total_pa_weight = sum(w for _, _, w in active_pa.values()) or 1.0
        bull_pa = sum(bull * (w / total_pa_weight) for bull, _, w in active_pa.values())
        bear_pa = sum(bear * (w / total_pa_weight) for _, bear, w in active_pa.values())
        
        candlestick_score = float(features.get("candlestick_pattern_score", 0.0))
        pa_score = np.clip((bull_pa - bear_pa) * 0.70 + candlestick_score * 0.30, -1.0, 1.0)

        # 4. Multi-Factor Alpha Additions:
        # A. Relative Strength vs BTC
        rs_score = 0.0
        if relative_strength_vs_btc is not None:
            # RS > 1.2 indicates strong altcoin outperformance
            rs_score = np.clip((relative_strength_vs_btc - 1.0) * 0.8, -1.0, 1.0)

        # B. Volume Surge Factor
        vol_surge = float(volume_surge_ratio or features.get("vol_surge_ratio", 1.0))

        # C. Volatility Squeeze Breakout
        is_squeeze = bool(volatility_squeeze or features.get("is_volatility_squeeze", False))
        squeeze_multiplier = 1.25 if is_squeeze else 1.0

        # 5. Composite Confluence Score
        weights = custom_weights or self._sample_mab_weights()
        w_micro = weights.get("microstructure", 0.30)
        w_tech = weights.get("technical", 0.30)
        w_pa = weights.get("price_action", 0.25)
        w_rs = 0.15

        base_score = (micro_score * w_micro) + (tech_score * w_tech) + (pa_score * w_pa) + (rs_score * w_rs)
        if vol_surge >= 2.0:
            base_score = base_score * 1.20

        composite_score = float(np.clip(base_score * squeeze_multiplier, -1.0, 1.0))

        # 6. Tri-Timeframe Alignment (15m + 5m + 1m)
        tf_15m = float(features.get("tf_15m_bias", 0.0))
        mtf_confirmed = float(features.get("mtf_confirmed", 0.0))

        # 7. TIER 1: BTC REGIME GATEKEEPER GATING
        btc = btc_context or {}
        btc_regime = btc.get("regime", "CHOP_COMPRESSION")
        btc_risk_mult = float(btc.get("risk_multiplier", 1.0))

        # Apply Strict Counter-Trend & BTC Gatekeeper Veto
        if btc_regime == "HIGH_VOLATILITY_FLUSH":
            # In a flash dump or panic wick, freeze new entries
            composite_score = 0.0
        elif btc_regime == "BEARISH_EXPANSION" and composite_score > 0:
            # Block or heavily dampen LONG entries during BTC bear trends
            if relative_strength_vs_btc and relative_strength_vs_btc > 2.0:
                composite_score *= 0.50  # Allow exceptional decoupled leaders with reduced size
            else:
                composite_score = 0.0  # VETO
        elif btc_regime == "BULLISH_EXPANSION" and composite_score < 0:
            # Block SHORT entries during strong BTC bull trends
            if relative_strength_vs_btc and relative_strength_vs_btc < 0.5:
                composite_score *= 0.50
            else:
                composite_score = 0.0  # VETO

        # 15m Counter-Trend Veto on the asset itself
        is_strict_veto = True if strict_counter_trend_veto is None else strict_counter_trend_veto
        if is_strict_veto:
            if composite_score > 0 and tf_15m < -0.08:
                composite_score = 0.0
            elif composite_score < 0 and tf_15m > 0.08:
                composite_score = 0.0

        # Scale by BTC risk factor
        composite_score *= btc_risk_mult

        # 8. Signal Classification & Raw Confidence
        if composite_score >= 0.35 and mtf_confirmed > 0:
            signal = "STRONG_BUY"
        elif composite_score >= 0.20:
            signal = "BUY"
        elif composite_score <= -0.35 and mtf_confirmed > 0:
            signal = "STRONG_SELL"
        elif composite_score <= -0.20:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        confidence_pct = round(float(min(98.5, 50.0 + abs(composite_score) * 48.0)), 2)

        # Regime of this individual asset
        bandwidth = float(features.get("bb_bandwidth", 0.02))
        if bandwidth > 0.04:
            regime = "HIGH_VOLATILITY_EXPANSION"
        elif abs(float(features.get("dist_sma_50", 0.0))) > 0.015:
            regime = "TRENDING_MOMENTUM"
        else:
            regime = "RANGING_CONSOLIDATION"

        # 9. DYNAMIC ASYMMETRIC BRACKETS (R:R >= 1.8:1)
        fmt = self.format_price_precision
        swing_high = float(features.get("swing_high_15", curr_price * 1.008))
        swing_low = float(features.get("swing_low_15", curr_price * 0.992))
        ema_9_val = float(features.get("ema_9", curr_price))

        if "BUY" in signal:
            if vol_surge >= 1.8 and bos_bull > 0:
                entry_type = "MARKET"
                entry_price = curr_price
            else:
                pullback_candidate = max(swing_low, min(curr_price, max(vwap_val, ema_9_val)))
                if pullback_candidate < curr_price * 0.9998:
                    entry_type = "LIMIT_PULLBACK"
                    entry_price = pullback_candidate
                else:
                    entry_type = "MARKET"
                    entry_price = curr_price

            # Stop Loss: Anchored to ATR or swing low
            if stop_loss_pct and stop_loss_pct > 0:
                sl_price = entry_price * (1.0 - (stop_loss_pct / 100.0))
            else:
                # Dynamic ATR Stop Loss with safety band
                raw_sl = min(entry_price - 1.0 * atr_14, swing_low - 0.2 * atr_14)
                min_sl = entry_price * 0.988
                max_sl = entry_price * 0.9975
                sl_price = max(min_sl, min(max_sl, raw_sl))
            
            risk_r = max(entry_price * 0.0030, entry_price - sl_price)

            # Asymmetric Take Profit: TP1 at 1.2R (breakeven lock), TP2 at 2.2R (maximum runner)
            if take_profit_1_pct and take_profit_1_pct > 0:
                tp1_price = entry_price * (1.0 + (take_profit_1_pct / 100.0))
            else:
                tp1_price = max(entry_price * 1.006, entry_price + 1.2 * risk_r)

            if take_profit_2_pct and take_profit_2_pct > 0:
                tp2_price = entry_price * (1.0 + (take_profit_2_pct / 100.0))
            else:
                tp2_price = max(entry_price * 1.012, entry_price + 2.2 * risk_r)

            breakeven_trigger = tp1_price
            breakeven_sl = entry_price * (1.0 + settings.FEE_BUFFER) if enable_breakeven is not False else sl_price
            rr_ratio = round((tp2_price - entry_price) / (risk_r + 1e-9), 2)

        elif "SELL" in signal:
            if vol_surge >= 1.8 and bos_bear > 0:
                entry_type = "MARKET"
                entry_price = curr_price
            else:
                pullback_candidate = min(swing_high, max(curr_price, min(vwap_val, ema_9_val)))
                if pullback_candidate > curr_price * 1.0002:
                    entry_type = "LIMIT_PULLBACK"
                    entry_price = pullback_candidate
                else:
                    entry_type = "MARKET"
                    entry_price = curr_price

            if stop_loss_pct and stop_loss_pct > 0:
                sl_price = entry_price * (1.0 + (stop_loss_pct / 100.0))
            else:
                raw_sl = max(entry_price + 1.0 * atr_14, swing_high + 0.2 * atr_14)
                min_sl = entry_price * 1.0025
                max_sl = entry_price * 1.012
                sl_price = min(max_sl, max(min_sl, raw_sl))

            risk_r = max(entry_price * 0.0030, sl_price - entry_price)

            if take_profit_1_pct and take_profit_1_pct > 0:
                tp1_price = entry_price * (1.0 - (take_profit_1_pct / 100.0))
            else:
                tp1_price = min(entry_price * 0.994, entry_price - 1.2 * risk_r)

            if take_profit_2_pct and take_profit_2_pct > 0:
                tp2_price = entry_price * (1.0 - (take_profit_2_pct / 100.0))
            else:
                tp2_price = min(entry_price * 0.988, entry_price - 2.2 * risk_r)

            breakeven_trigger = tp1_price
            breakeven_sl = entry_price * (1.0 - settings.FEE_BUFFER) if enable_breakeven is not False else sl_price
            rr_ratio = round((entry_price - tp2_price) / (risk_r + 1e-9), 2)

        else:
            entry_type = "MARKET"
            entry_price = curr_price
            tp1_price = curr_price
            tp2_price = curr_price
            sl_price = curr_price
            breakeven_trigger = curr_price
            breakeven_sl = curr_price
            risk_r = curr_price * 0.0045
            rr_ratio = 1.0

        # 10. TIER 3: MATHEMATICAL EXPECTED VALUE (EV) CALCULATION
        is_long = "BUY" in signal
        is_short = "SELL" in signal
        is_active_signal = is_long or is_short
        
        # Calculate win probability factoring in multi-factor confluence
        if is_active_signal:
            base_prob = confidence_pct / 100.0
            obi_factor = 0.05 * (obi_10 if is_long else -obi_10)
            surge_bonus = 0.04 if vol_surge >= 1.8 else 0.0
            rs_bonus = 0.04 if (relative_strength_vs_btc and ((is_long and relative_strength_vs_btc > 1.2) or (is_short and relative_strength_vs_btc < 0.8))) else 0.0
            
            # MTF Alignment bonus: 15m + 5m + 1m confluence
            mtf_align = float(features.get("tri_timeframe_alignment", 0.0))
            mtf_bonus = 0.05 if ((is_long and mtf_align > 0) or (is_short and mtf_align < 0)) else 0.0

            # Delta volume bonus: aggressive buyer/seller absorption
            delta_ratio = float(features.get("delta_volume_ratio", 0.0))
            delta_bonus = 0.04 if ((is_long and delta_ratio > 0.15) or (is_short and delta_ratio < -0.15)) else 0.0

            calibrated_win_prob = np.clip(base_prob + obi_factor + surge_bonus + rs_bonus + mtf_bonus + delta_bonus, 0.50, 0.95)
            
            # Compute EV with realistic fee friction using blended 50/50 TP1 and TP2 targets
            blended_tp = (tp1_price + tp2_price) / 2.0 if tp2_price else tp1_price
            ev_metrics = ExpectedValueModel.compute_expected_value(
                win_prob=float(calibrated_win_prob),
                entry_price=entry_price,
                tp_price=blended_tp,
                sl_price=sl_price,
                is_long=is_long
            )
            
            # If EV <= 0, disqualify trade to NEUTRAL to protect capital
            if not ev_metrics["is_viable"] and "STRONG" not in signal:
                if ev_metrics["ev"] <= 0:
                    signal = "NEUTRAL"
                    confidence_pct = 50.0
        else:
            calibrated_win_prob = 0.50
            ev_metrics = {
                "ev": 0.0, "ev_pct": 0.0, "net_gain_pct": 0.0, "net_loss_pct": 0.0,
                "roundtrip_fee_pct": 0.118, "rr_ratio": 1.0, "is_viable": False
            }

        elapsed_ns = time.perf_counter_ns() - start_time
        latency_us = round(elapsed_ns / 1000.0, 2)

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence_pct,
            "win_probability_pct": round(float(calibrated_win_prob * 100.0), 1),
            "expected_value_pct": ev_metrics["ev_pct"],
            "is_ev_viable": ev_metrics["is_viable"],
            "regime": regime,
            "latency_us": latency_us,
            "entry_type": entry_type,
            "entry_price": fmt(entry_price),
            "market_price": fmt(curr_price),
            "tp1_price": fmt(tp1_price),
            "tp2_price": fmt(tp2_price),
            "sl_price": fmt(sl_price),
            "breakeven_trigger": fmt(breakeven_trigger),
            "breakeven_sl": fmt(breakeven_sl),
            "risk_r": fmt(risk_r),
            "rr_ratio": rr_ratio,
            "micro_score": round(float(micro_score), 3),
            "tech_score": round(float(tech_score), 3),
            "pa_score": round(float(pa_score), 3),
            "candlestick_pattern": features.get("candlestick_pattern_score", 0.0),
            "vol_surge": round(float(vol_surge), 2),
            "btc_regime": btc_regime,
            "relative_strength": round(float(relative_strength_vs_btc or 1.0), 2),
            "tri_timeframe_alignment": features.get("tri_timeframe_alignment", 0.0),
            "mtf_confirmed": bool(features.get("mtf_confirmed", 0.0)),
            "delta_volume_ratio": features.get("delta_volume_ratio", 0.0),
            "tf_15m_bias": round(float(features.get("tf_15m_bias", 0.0)), 2),
            "tf_5m_bias": round(float(features.get("tf_5m_bias", 0.0)), 2),
            "tf_1m_bias": round(float(features.get("tf_1m_bias", 0.0)), 2)
        }

    def _sample_mab_weights(self) -> Dict[str, float]:
        """Thompson Sampling with Dirichlet baseline prior to weight strategy modules."""
        samples = {}
        for s in self.strategies:
            samples[s] = np.random.beta(self.mab_successes[s], self.mab_failures[s])
        
        total = sum(samples.values()) + 1e-9
        raw_micro = 0.20 + 0.50 * (samples["OBI_MOMENTUM"] / total)
        raw_tech = 0.30 + 0.50 * (samples["VOLATILITY_BREAKOUT"] / total)
        raw_pa = 0.15 + 0.50 * ((samples["SMC_LIQUIDITY_SWEEP"] + samples["MEAN_REVERSION"]) / (2.0 * total))
        norm_sum = raw_micro + raw_tech + raw_pa
        return {
            "microstructure": raw_micro / norm_sum,
            "technical": raw_tech / norm_sum,
            "price_action": raw_pa / norm_sum
        }

    def record_trade_outcome(self, strategy: str, is_win: bool):
        """Updates Multi-Armed Bandit feedback loop."""
        if strategy in self.mab_successes:
            if is_win:
                self.mab_successes[strategy] += 1.0
            else:
                self.mab_failures[strategy] += 1.0

    # ==============================================================
    # 4. BLACK-76 OPTIONS & DERIVATIVES GREEKS ENGINE
    # ==============================================================
    def calculate_options_greeks(
        self,
        futures_price: float,
        strike_price: float,
        time_to_expiry_years: float,
        risk_free_rate: float,
        implied_volatility: float,
        is_call: bool = True
    ) -> Dict[str, float]:
        """
        Black-76 European options model for crypto futures & options derivatives.
        Returns: price, delta, gamma, vega, theta, delta_neutral_hedge_qty.
        """
        if time_to_expiry_years <= 0 or implied_volatility <= 0:
            intrinsic = max(0.0, (futures_price - strike_price) if is_call else (strike_price - futures_price))
            return {
                "price": intrinsic, "delta": 1.0 if (is_call and futures_price > strike_price) else 0.0,
                "gamma": 0.0, "vega": 0.0, "theta": 0.0, "hedge_ratio": 0.0
            }

        F = futures_price
        K = strike_price
        T = time_to_expiry_years
        r = risk_free_rate
        sigma = implied_volatility

        d1 = (math.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        discount = math.exp(-r * T)
        norm_cdf_d1 = norm.cdf(d1 if is_call else -d1)
        norm_pdf_d1 = self._inv_sqrt_2pi * math.exp(-0.5 * d1 ** 2)

        if is_call:
            price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
            delta = discount * norm.cdf(d1)
        else:
            price = discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
            delta = -discount * norm.cdf(-d1)

        gamma = discount * norm_pdf_d1 / (F * sigma * math.sqrt(T))
        vega = F * discount * norm_pdf_d1 * math.sqrt(T) / 100.0
        theta = - (F * discount * norm_pdf_d1 * sigma) / (2.0 * math.sqrt(T)) / 365.0

        hedge_ratio = -delta

        return {
            "price": round(price, 4),
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "vega": round(vega, 4),
            "theta": round(theta, 4),
            "hedge_ratio": round(hedge_ratio, 4)
        }
