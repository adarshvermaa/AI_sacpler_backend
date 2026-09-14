"""
AlphaScalper - Core Configuration Module
Handles environment variables, API endpoints, risk management limits, and strategy presets.
"""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AlphaScalperSettings(BaseSettings):
    # Brand Information
    BRAND_NAME: str = "AlphaScalper"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    
    # Trading Mode: "PAPER" (simulation) or "LIVE"
    TRADING_MODE: str = Field(default="PAPER", description="PAPER or LIVE")
    
    # CoinDCX API Credentials
    COINDCX_API_KEY: str = Field(default="YOUR_API_KEY", description="CoinDCX API Key")
    COINDCX_API_SECRET: str = Field(default="YOUR_API_SECRET", description="CoinDCX API Secret")
    
    # CoinDCX Endpoints
    COINDCX_REST_BASE_URL: str = "https://api.coindcx.com"
    COINDCX_PUBLIC_REST_URL: str = "https://public.coindcx.com"
    COINDCX_SPOT_SOCKET_URL: str = "wss://stream-spot.coindcx.com"
    
    # Webhook Secret for Inbound CoinDCX Notifications
    WEBHOOK_SECRET: str = "alphascalper_webhook_secret_key"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ]
    
    # Screener & Scanner Settings
    UNIVERSE_MAX_ASSETS: int = 1500  # Total available perpetual futures assets
    FILTER_TOP_CANDIDATES: int = 10  # Strictly Top 10 high-probability profit candidates
    MAX_ACTIVE_ORDERS: int = 10
    DEFAULT_ACTIVE_ORDERS: int = 10
    
    # Scalper Execution & Risk Control
    DEFAULT_LEVERAGE: float = 15.0
    MAX_LEVERAGE: float = 50.0
    MARGIN_MODE: str = "isolated"  # "isolated" or "crossed"
    MARGIN_CURRENCY: str = "INR"   # "INR" for CoinDCX India accounts, or "USDT"
    
    # "Win-Win" Strategy Parameters
    TP1_RATIO: float = 0.0035       # +0.35% Take Profit 1 (Closes 50% & ratchets SL to Breakeven)
    TP2_RATIO: float = 0.0085       # +0.85% Take Profit 2 (Target runner)
    HARD_SL_RATIO: float = 0.0045   # -0.45% Emergency Stop Loss
    FEE_BUFFER: float = 0.0015      # +0.15% buffer over entry price to guarantee net profit after CoinDCX taker fees (0.10%) and GST (0.018%)
    MICRO_TIMEOUT_SECONDS: int = 600 # 10 minutes to allow scalp to reach TP1 and clear fees (eliminates 30s churn)
    OBI_FLIP_EXIT: bool = True      # Exit immediately if orderbook imbalance turns hostile
    
    # Capital Allocation & Risk Limits
    MAX_RISK_PER_TRADE_PERCENT: float = 1.0  # Max 1% account risk per scalp
    MAX_DAILY_DRAWDOWN_PERCENT: float = 5.0  # Kill-switch if daily PnL drops -5%
    INITIAL_SIMULATION_BALANCE: float = 10000.0  # USDT for Paper Trading
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = AlphaScalperSettings()
