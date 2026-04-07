# ============================================================
# Project 1a: Crypto Market Risk Intelligence System
# trading_signals.py -- Buy/Sell/Hold trading signal generation
# ============================================================

import logging
import config
from config import TRADING_SIGNAL_ENABLED, TRADING_STRATEGIES

logger = logging.getLogger(__name__)


def generate_trading_signal(risk_score: float, strategy: str = None) -> dict:
    """
    Generate trading signal based on risk score and selected strategy.
    
    BUY:  Risk score is LOW (market is calm/stable)
    HOLD: Risk score is in MEDIUM range (uncertain)
    SELL: Risk score is HIGH (market is unstable/risky)
    
    Args:
        risk_score: Current risk score (0-100)
        strategy: Trading strategy name ("conservative", "balanced", "aggressive", "asymmetric")
                 If None, uses CURRENT_TRADING_STRATEGY
    
    Returns:
        {
            "signal": "BUY" | "SELL" | "HOLD",
            "confidence": float (0-100),
            "reasoning": str,
            "recommendation": str,
            "strategy": str,
            "emoji": str
        }
    """
    
    if not TRADING_SIGNAL_ENABLED:
        return {
            "signal": "NONE",
            "confidence": 0,
            "reasoning": "Trading signals are disabled",
            "recommendation": "Monitor the market",
            "strategy": strategy or config.CURRENT_TRADING_STRATEGY,
            "emoji": "⚪"
        }

    # Use provided strategy or fall back to current global (always reads live value)
    selected_strategy = strategy or config.CURRENT_TRADING_STRATEGY
    
    # Validate strategy exists
    if selected_strategy not in TRADING_STRATEGIES:
        selected_strategy = "conservative"
    
    # Get strategy parameters
    strat = TRADING_STRATEGIES[selected_strategy]
    buy_threshold = strat["buy_threshold"]
    sell_threshold = strat["sell_threshold"]
    hold_low = buy_threshold + 1
    hold_high = sell_threshold - 1

    if risk_score <= buy_threshold:
        confidence = min(100, (buy_threshold - risk_score) * 3)
        return {
            "signal": "BUY",
            "confidence": confidence,
            "reasoning": f"Risk score {risk_score:.1f} is at optimal buy level (≤{buy_threshold}). Market conditions favor entry.",
            "recommendation": f"✅ Strong BUY signal using {strat['name']} strategy. Market volatility is predictable and favorable for buyers.",
            "strategy": selected_strategy,
            "strategy_name": strat["name"],
            "emoji": "🟢"
        }
    
    elif risk_score >= sell_threshold:
        confidence = min(100, (risk_score - sell_threshold) * 3)
        return {
            "signal": "SELL",
            "confidence": confidence,
            "reasoning": f"Risk score {risk_score:.1f} has reached sell signal (≥{sell_threshold}). Market volatility is elevated.",
            "recommendation": f"🛑 SELL signal using {strat['name']} strategy. Consider taking profits or reducing exposure.",
            "strategy": selected_strategy,
            "strategy_name": strat["name"],
            "emoji": "🔴"
        }
    
    else:
        hold_range_size = hold_high - hold_low
        position_in_range = risk_score - hold_low
        confidence = 50 - (abs(hold_range_size // 2 - position_in_range)) * (50 / (hold_range_size / 2 + 1))
        confidence = max(30, min(100, confidence))
        
        return {
            "signal": "HOLD",
            "confidence": confidence,
            "reasoning": f"Risk score {risk_score:.1f} is in the HOLD zone ({hold_low}-{hold_high}). Mixed market signals detected.",
            "recommendation": f"⏸️ HOLD signal using {strat['name']} strategy. Wait for clearer directional signals before acting.",
            "strategy": selected_strategy,
            "strategy_name": strat["name"],
            "emoji": "🟡"
        }


def get_signal_color(signal_type: str) -> str:
    """Get color code for signal type."""
    color_map = {
        "BUY": "#4caf50",    # Green
        "SELL": "#f44336",   # Red
        "HOLD": "#ff9800",   # Orange
        "NONE": "#9e9e9e",   # Gray
    }
    return color_map.get(signal_type, "#9e9e9e")


def get_available_strategies() -> dict:
    """Get all available trading strategies."""
    return TRADING_STRATEGIES


def get_strategy_info(strategy: str) -> dict:
    """Get detailed info about a specific strategy."""
    if strategy not in TRADING_STRATEGIES:
        return None
    return TRADING_STRATEGIES[strategy]
