"""
AlphaScalper - Dual-Storage Persistence & Analytics Engine
Primary Store: Neon Cloud PostgreSQL (Serverless Connection Pooling)
Local Mirror / Offline Fallback: SQLite WAL (Zero-Downtime, High-Frequency Writes)

Handles:
- Trade persistence across server restarts
- Granular lifecycle trade events (ENTRY, TP1_HIT, TP2_HIT, STOP_LOSS_HIT, BREAKEVEN_HIT)
- Rolling 24-Hour financial analytics (Gross Profits, Gross Losses, Net PnL in INR & USDT, Win Rate, Event Counts)
- Equity snapshots & AI signal auditing
"""

import os
import re
import time
import logging
import asyncio
import sqlite3
from typing import Dict, List, Optional, Any

try:
    import asyncpg
except ImportError:
    asyncpg = None

from config import settings

logger = logging.getLogger("AlphaScalper.DB")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.db")


# ==============================================================================
# 1. SQLITE LOCAL WAL STORAGE (Sub-millisecond local cache & offline fallback)
# ==============================================================================

def get_sqlite_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_sqlite_db(db_path: str = DB_PATH):
    """Initializes local SQLite database schemas."""
    conn = get_sqlite_conn(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                total_size_usdt REAL NOT NULL,
                entry_price REAL NOT NULL,
                tp1_price REAL NOT NULL,
                tp2_price REAL NOT NULL,
                sl_price REAL NOT NULL,
                breakeven_sl REAL NOT NULL,
                leverage REAL NOT NULL,
                current_state TEXT NOT NULL,
                filled_qty REAL NOT NULL,
                remaining_qty REAL NOT NULL,
                realized_pnl REAL DEFAULT 0.0,
                total_fees_paid REAL DEFAULT 0.0,
                net_realized_pnl REAL DEFAULT 0.0,
                is_risk_free INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                entry_time REAL,
                exit_time REAL,
                updated_at REAL NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_state ON trades(current_state);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                event_type TEXT NOT NULL,
                price REAL DEFAULT 0.0,
                qty REAL DEFAULT 0.0,
                pnl REAL DEFAULT 0.0,
                fee REAL DEFAULT 0.0,
                message TEXT,
                timestamp REAL NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_trade_id ON trade_events(trade_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON trade_events(event_type);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON trade_events(timestamp);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_inr REAL NOT NULL,
                wallet_usdt REAL NOT NULL,
                unrealized_pnl REAL DEFAULT 0.0,
                active_positions_count INTEGER DEFAULT 0,
                margin_used REAL DEFAULT 0.0,
                timestamp REAL NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(timestamp);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_signals_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                volatility_regime TEXT,
                obi_score REAL DEFAULT 0.0,
                executed INTEGER DEFAULT 0,
                timestamp REAL NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_ts ON ai_signals_log(timestamp);")

    conn.close()
    logger.info(f"Initialized AlphaScalper local SQLite WAL store at {db_path}")


# Legacy alias
init_db = init_sqlite_db


# ==============================================================================
# 2. NEON CLOUD POSTGRESQL ENGINE (asyncpg Connection Pool)
# ==============================================================================

class NeonPostgresManager:
    def __init__(self, dsn: Optional[str] = None):
        raw_dsn = dsn or getattr(settings, "NEON_DATABASE_URL", None)
        # Sanitize DSN: remove any unsupported channel_binding params for asyncpg
        if raw_dsn:
            self.dsn = re.sub(r'[\?&]channel_binding=[^&]+', '', raw_dsn)
            if 'sslmode=' not in self.dsn and '?' not in self.dsn:
                self.dsn += "?sslmode=require"
        else:
            self.dsn = None

        self.pool: Optional[Any] = None
        self.is_connected: bool = False
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Initializes connection pool to Neon Cloud PostgreSQL."""
        if not self.dsn or not asyncpg:
            logger.warning("Neon PostgreSQL DSN not configured or asyncpg missing. Operating in SQLite-only mode.")
            return False

        try:
            async with self._lock:
                if self.pool is None:
                    self.pool = await asyncpg.create_pool(
                        dsn=self.dsn,
                        min_size=2,
                        max_size=10,
                        ssl="require",
                        command_timeout=20
                    )
                    self.is_connected = True
                    logger.info("Connected to Neon Cloud PostgreSQL connection pool successfully.")
                    await self._init_neon_schema()
                    return True
                return True
        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to connect to Neon Cloud PostgreSQL: {e}")
            return False

    async def close(self):
        """Closes the Neon connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self.is_connected = False
            logger.info("Closed Neon Cloud PostgreSQL connection pool.")

    async def _init_neon_schema(self):
        """Provisions tables and institutional indexes on Neon Cloud PostgreSQL."""
        if not self.pool:
            return

        queries = [
            """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id VARCHAR(64) PRIMARY KEY,
                symbol VARCHAR(32) NOT NULL,
                side VARCHAR(8) NOT NULL,
                total_size_usdt DOUBLE PRECISION NOT NULL,
                entry_price DOUBLE PRECISION NOT NULL,
                tp1_price DOUBLE PRECISION NOT NULL,
                tp2_price DOUBLE PRECISION NOT NULL,
                sl_price DOUBLE PRECISION NOT NULL,
                breakeven_sl DOUBLE PRECISION NOT NULL,
                leverage DOUBLE PRECISION NOT NULL,
                current_state VARCHAR(64) NOT NULL,
                filled_qty DOUBLE PRECISION NOT NULL,
                remaining_qty DOUBLE PRECISION NOT NULL,
                realized_pnl DOUBLE PRECISION DEFAULT 0.0,
                total_fees_paid DOUBLE PRECISION DEFAULT 0.0,
                net_realized_pnl DOUBLE PRECISION DEFAULT 0.0,
                is_risk_free BOOLEAN DEFAULT FALSE,
                created_at DOUBLE PRECISION NOT NULL,
                entry_time DOUBLE PRECISION,
                exit_time DOUBLE PRECISION,
                updated_at DOUBLE PRECISION NOT NULL
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);",
            "CREATE INDEX IF NOT EXISTS idx_trades_state ON trades(current_state);",
            "CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);",
            "CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);",

            """
            CREATE TABLE IF NOT EXISTS trade_events (
                id SERIAL PRIMARY KEY,
                trade_id VARCHAR(64) NOT NULL,
                symbol VARCHAR(32) NOT NULL,
                event_type VARCHAR(64) NOT NULL,
                price DOUBLE PRECISION DEFAULT 0.0,
                qty DOUBLE PRECISION DEFAULT 0.0,
                pnl DOUBLE PRECISION DEFAULT 0.0,
                fee DOUBLE PRECISION DEFAULT 0.0,
                message TEXT,
                timestamp DOUBLE PRECISION NOT NULL
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_events_trade_id ON trade_events(trade_id);",
            "CREATE INDEX IF NOT EXISTS idx_events_type ON trade_events(event_type);",
            "CREATE INDEX IF NOT EXISTS idx_events_ts ON trade_events(timestamp);",

            """
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id SERIAL PRIMARY KEY,
                wallet_inr DOUBLE PRECISION NOT NULL,
                wallet_usdt DOUBLE PRECISION NOT NULL,
                unrealized_pnl DOUBLE PRECISION DEFAULT 0.0,
                active_positions_count INT DEFAULT 0,
                margin_used DOUBLE PRECISION DEFAULT 0.0,
                timestamp DOUBLE PRECISION NOT NULL
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(timestamp);",

            """
            CREATE TABLE IF NOT EXISTS ai_signals_log (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(32) NOT NULL,
                direction VARCHAR(8) NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                volatility_regime VARCHAR(32),
                obi_score DOUBLE PRECISION DEFAULT 0.0,
                executed BOOLEAN DEFAULT FALSE,
                timestamp DOUBLE PRECISION NOT NULL
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_signals_ts ON ai_signals_log(timestamp);"
        ]

        async with self.pool.acquire() as conn:
            for q in queries:
                await conn.execute(q)
        logger.info("Neon Cloud PostgreSQL tables and indexes verified/provisioned.")

    async def save_trade_neon(self, trade: Any):
        """Asynchronously upserts a trade record into Neon Cloud PostgreSQL."""
        if not self.pool:
            return
        try:
            now = time.time()
            query = """
                INSERT INTO trades (
                    trade_id, symbol, side, total_size_usdt, entry_price,
                    tp1_price, tp2_price, sl_price, breakeven_sl, leverage,
                    current_state, filled_qty, remaining_qty, realized_pnl,
                    total_fees_paid, net_realized_pnl, is_risk_free,
                    created_at, entry_time, exit_time, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
                ON CONFLICT (trade_id) DO UPDATE SET
                    current_state = EXCLUDED.current_state,
                    remaining_qty = EXCLUDED.remaining_qty,
                    sl_price = EXCLUDED.sl_price,
                    realized_pnl = EXCLUDED.realized_pnl,
                    total_fees_paid = EXCLUDED.total_fees_paid,
                    net_realized_pnl = EXCLUDED.net_realized_pnl,
                    is_risk_free = EXCLUDED.is_risk_free,
                    entry_time = EXCLUDED.entry_time,
                    exit_time = EXCLUDED.exit_time,
                    updated_at = EXCLUDED.updated_at;
            """
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    trade.trade_id,
                    trade.symbol,
                    trade.side,
                    float(trade.total_size_usdt),
                    float(trade.entry_price),
                    float(trade.tp1_price),
                    float(trade.tp2_price),
                    float(trade.sl_price),
                    float(trade.breakeven_sl),
                    float(trade.leverage),
                    str(trade.current_state),
                    float(trade.filled_qty),
                    float(trade.remaining_qty),
                    float(getattr(trade, "realized_pnl", 0.0)),
                    float(getattr(trade, "total_fees_paid", 0.0)),
                    float(getattr(trade, "net_realized_pnl", 0.0)),
                    bool(getattr(trade, "is_risk_free", False)),
                    float(trade.created_at),
                    float(trade.entry_time) if trade.entry_time else None,
                    float(trade.exit_time) if trade.exit_time else None,
                    now
                )
        except Exception as e:
            logger.error(f"[NEON] Error saving trade {trade.trade_id}: {e}")

    async def mark_trade_closed_neon(self, trade_id: str, exit_time: float, net_pnl: float, exit_state: str):
        """Marks trade closed on Neon Cloud PostgreSQL."""
        if not self.pool:
            return
        try:
            query = """
                UPDATE trades
                SET remaining_qty = 0.0,
                    exit_time = $1,
                    net_realized_pnl = $2,
                    current_state = $3,
                    updated_at = $4
                WHERE trade_id = $5;
            """
            async with self.pool.acquire() as conn:
                await conn.execute(query, exit_time, net_pnl, exit_state, time.time(), trade_id)
        except Exception as e:
            logger.error(f"[NEON] Error marking trade {trade_id} closed: {e}")

    async def log_event_neon(self, trade_id: str, symbol: str, event_type: str, price: float, qty: float, pnl: float, fee: float, message: str, timestamp: float):
        """Records granular trade lifecycle event into Neon Cloud PostgreSQL."""
        if not self.pool:
            return
        try:
            query = """
                INSERT INTO trade_events (trade_id, symbol, event_type, price, qty, pnl, fee, message, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9);
            """
            async with self.pool.acquire() as conn:
                await conn.execute(query, trade_id, symbol, event_type, price, qty, pnl, fee, message, timestamp)
        except Exception as e:
            logger.error(f"[NEON] Error logging trade event for {symbol}: {e}")

    async def save_equity_snapshot_neon(self, wallet_inr: float, wallet_usdt: float, unrealized_pnl: float, active_positions_count: int, margin_used: float, timestamp: float):
        """Records account equity snapshot into Neon Cloud PostgreSQL."""
        if not self.pool:
            return
        try:
            query = """
                INSERT INTO equity_snapshots (wallet_inr, wallet_usdt, unrealized_pnl, active_positions_count, margin_used, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6);
            """
            async with self.pool.acquire() as conn:
                await conn.execute(query, wallet_inr, wallet_usdt, unrealized_pnl, active_positions_count, margin_used, timestamp)
        except Exception as e:
            logger.error(f"[NEON] Error saving equity snapshot: {e}")

    async def log_ai_signal_neon(self, symbol: str, direction: str, confidence: float, volatility_regime: str, obi_score: float, executed: bool, timestamp: float):
        """Logs AI generated signal into Neon Cloud PostgreSQL."""
        if not self.pool:
            return
        try:
            query = """
                INSERT INTO ai_signals_log (symbol, direction, confidence, volatility_regime, obi_score, executed, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7);
            """
            async with self.pool.acquire() as conn:
                await conn.execute(query, symbol, direction, confidence, volatility_regime, obi_score, executed, timestamp)
        except Exception as e:
            logger.error(f"[NEON] Error logging AI signal for {symbol}: {e}")


# Singleton instance of Neon Manager
neon_manager = NeonPostgresManager()


# ==============================================================================
# 3. DUAL-STORAGE UNIFIED INTERFACE (Zero Downtime, Non-Blocking Execution)
# ==============================================================================

def save_trade(trade: Any, db_path: str = DB_PATH):
    """
    Synchronously persists trade to SQLite WAL (instant, no tick delay),
    and dispatches non-blocking async persistence to Neon Cloud PostgreSQL.
    """
    now = time.time()
    # 1. SQLite WAL Write
    try:
        conn = get_sqlite_conn(db_path)
        with conn:
            conn.execute("""
                INSERT INTO trades (
                    trade_id, symbol, side, total_size_usdt, entry_price,
                    tp1_price, tp2_price, sl_price, breakeven_sl, leverage,
                    current_state, filled_qty, remaining_qty, realized_pnl,
                    total_fees_paid, net_realized_pnl, is_risk_free,
                    created_at, entry_time, exit_time, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    current_state=excluded.current_state,
                    remaining_qty=excluded.remaining_qty,
                    sl_price=excluded.sl_price,
                    realized_pnl=excluded.realized_pnl,
                    total_fees_paid=excluded.total_fees_paid,
                    net_realized_pnl=excluded.net_realized_pnl,
                    is_risk_free=excluded.is_risk_free,
                    entry_time=excluded.entry_time,
                    exit_time=excluded.exit_time,
                    updated_at=excluded.updated_at;
            """, (
                trade.trade_id,
                trade.symbol,
                trade.side,
                float(trade.total_size_usdt),
                float(trade.entry_price),
                float(trade.tp1_price),
                float(trade.tp2_price),
                float(trade.sl_price),
                float(trade.breakeven_sl),
                float(trade.leverage),
                str(trade.current_state),
                float(trade.filled_qty),
                float(trade.remaining_qty),
                float(getattr(trade, "realized_pnl", 0.0)),
                float(getattr(trade, "total_fees_paid", 0.0)),
                float(getattr(trade, "net_realized_pnl", 0.0)),
                1 if getattr(trade, "is_risk_free", False) else 0,
                float(trade.created_at),
                float(trade.entry_time) if trade.entry_time else None,
                float(trade.exit_time) if trade.exit_time else None,
                now
            ))
        conn.close()
    except Exception as e:
        logger.error(f"[SQLITE] Failed to persist trade {trade.trade_id}: {e}")

    # 2. Neon Cloud Async Dispatch
    if neon_manager.is_connected:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(neon_manager.save_trade_neon(trade))
        except RuntimeError:
            # Running outside event loop (e.g. unit tests or sync scripts)
            pass


def mark_trade_closed(trade_id: str, exit_time: float, net_pnl: float, exit_state: str, db_path: str = DB_PATH):
    """Marks a trade closed in SQLite WAL and dispatches to Neon Cloud PostgreSQL."""
    try:
        conn = get_sqlite_conn(db_path)
        with conn:
            conn.execute("""
                UPDATE trades
                SET remaining_qty = 0.0,
                    exit_time = ?,
                    net_realized_pnl = ?,
                    current_state = ?,
                    updated_at = ?
                WHERE trade_id = ?;
            """, (exit_time, net_pnl, exit_state, time.time(), trade_id))
        conn.close()
    except Exception as e:
        logger.error(f"[SQLITE] Failed to mark trade {trade_id} closed: {e}")

    if neon_manager.is_connected:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(neon_manager.mark_trade_closed_neon(trade_id, exit_time, net_pnl, exit_state))
        except RuntimeError:
            pass


def log_trade_event(
    trade_id: str,
    symbol: str,
    event_type: str,
    price: float = 0.0,
    qty: float = 0.0,
    pnl: float = 0.0,
    fee: float = 0.0,
    message: str = "",
    timestamp: Optional[float] = None,
    db_path: str = DB_PATH
):
    """
    Logs specific trade events:
    - 'ENTRY'
    - 'TP1_HIT'
    - 'TP2_HIT'
    - 'STOP_LOSS_HIT'
    - 'BREAKEVEN_HIT'
    - 'SCALE_OUT'
    """
    ts = timestamp or time.time()
    try:
        conn = get_sqlite_conn(db_path)
        with conn:
            conn.execute("""
                INSERT INTO trade_events (trade_id, symbol, event_type, price, qty, pnl, fee, message, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (trade_id, symbol, event_type, price, qty, pnl, fee, message, ts))
        conn.close()
    except Exception as e:
        logger.error(f"[SQLITE] Failed to log trade event {event_type} for {trade_id}: {e}")

    if neon_manager.is_connected:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(neon_manager.log_event_neon(trade_id, symbol, event_type, price, qty, pnl, fee, message, ts))
        except RuntimeError:
            pass


def save_equity_snapshot(
    wallet_inr: float,
    wallet_usdt: float,
    unrealized_pnl: float = 0.0,
    active_positions_count: int = 0,
    margin_used: float = 0.0,
    timestamp: Optional[float] = None,
    db_path: str = DB_PATH
):
    """Saves equity telemetry snapshot for charting and analytics."""
    ts = timestamp or time.time()
    try:
        conn = get_sqlite_conn(db_path)
        with conn:
            conn.execute("""
                INSERT INTO equity_snapshots (wallet_inr, wallet_usdt, unrealized_pnl, active_positions_count, margin_used, timestamp)
                VALUES (?, ?, ?, ?, ?, ?);
            """, (wallet_inr, wallet_usdt, unrealized_pnl, active_positions_count, margin_used, ts))
        conn.close()
    except Exception as e:
        logger.error(f"[SQLITE] Failed to save equity snapshot: {e}")

    if neon_manager.is_connected:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(neon_manager.save_equity_snapshot_neon(
                wallet_inr, wallet_usdt, unrealized_pnl, active_positions_count, margin_used, ts
            ))
        except RuntimeError:
            pass


def log_ai_signal(
    symbol: str,
    direction: str,
    confidence: float,
    volatility_regime: str = "NORMAL",
    obi_score: float = 0.0,
    executed: bool = False,
    timestamp: Optional[float] = None,
    db_path: str = DB_PATH
):
    """Logs AI signals for performance auditing."""
    ts = timestamp or time.time()
    try:
        conn = get_sqlite_conn(db_path)
        with conn:
            conn.execute("""
                INSERT INTO ai_signals_log (symbol, direction, confidence, volatility_regime, obi_score, executed, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (symbol, direction, confidence, volatility_regime, obi_score, 1 if executed else 0, ts))
        conn.close()
    except Exception as e:
        logger.error(f"[SQLITE] Failed to log AI signal for {symbol}: {e}")

    if neon_manager.is_connected:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(neon_manager.log_ai_signal_neon(
                symbol, direction, confidence, volatility_regime, obi_score, executed, ts
            ))
        except RuntimeError:
            pass


def load_active_trades(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Loads all non-closed trades from the database for state recovery."""
    try:
        conn = get_sqlite_conn(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM trades
            WHERE remaining_qty > 0 
            AND current_state NOT IN ('TP2_HIT_CLOSED', 'BREAKEVEN_CLOSED', 'STOP_LOSS_CLOSED', 'MICRO_TIMEOUT_CLOSED')
            ORDER BY created_at ASC;
        """)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Failed to load active trades from DB: {e}")
        return []


def load_closed_trades(limit: int = 100, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Loads recently closed trades from the database."""
    try:
        conn = get_sqlite_conn(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM trades
            WHERE current_state IN ('TP2_HIT_CLOSED', 'BREAKEVEN_CLOSED', 'STOP_LOSS_CLOSED', 'MICRO_TIMEOUT_CLOSED')
            ORDER BY exit_time DESC
            LIMIT ?;
        """, (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Failed to load closed trades from DB: {e}")
        return []


# ==============================================================================
# 4. ROLLING 24-HOUR PERFORMANCE & EXECUTION ANALYTICS
# ==============================================================================

async def get_24h_analytics(inr_rate: float = 87.5, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Computes rigorous institutional 24-hour rolling statistics:
    - TP1 Hits Count (Winning scalps locking 50% profit and making trade risk-free)
    - TP2 Hits Count (Maximum profit target reached)
    - Stop Loss Hits Count (Trades stopped out at emergency SL)
    - Breakeven Exits Count (Trades exiting at entry + fee buffer risk-free)
    - Gross Earnings in 24h (Total $ USDT and ₹ INR earned from winning trades)
    - Gross Losses in 24h (Total $ USDT and ₹ INR lost from losing trades)
    - Net PnL in 24h (Net profit/loss including all exchange fees & GST)
    - Win Rate % & Profit Factor over rolling 24 hours
    """
    now = time.time()
    cutoff_24h = now - 86400.0  # 24 hours ago

    # Try querying Neon Cloud PostgreSQL first if pool is connected
    if neon_manager.is_connected and neon_manager.pool:
        try:
            async with neon_manager.pool.acquire() as conn:
                # 1. Closed trades within last 24 hours
                trades_rows = await conn.fetch("""
                    SELECT trade_id, symbol, side, entry_price, realized_pnl, total_fees_paid,
                           net_realized_pnl, current_state, is_risk_free, exit_time
                    FROM trades
                    WHERE exit_time >= $1
                    ORDER BY exit_time DESC;
                """, cutoff_24h)

                # 2. Granular events within last 24 hours
                events_rows = await conn.fetch("""
                    SELECT event_type, COUNT(*) as count
                    FROM trade_events
                    WHERE timestamp >= $1
                    GROUP BY event_type;
                """, cutoff_24h)

                return _compute_stats_from_records(trades_rows, events_rows, inr_rate, is_neon=True)
        except Exception as e:
            logger.warning(f"[NEON] Analytics query failed, falling back to SQLite: {e}")

    # Fallback to local SQLite WAL database
    try:
        conn = get_sqlite_conn(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT trade_id, symbol, side, entry_price, realized_pnl, total_fees_paid,
                   net_realized_pnl, current_state, is_risk_free, exit_time
            FROM trades
            WHERE exit_time >= ?
            ORDER BY exit_time DESC;
        """, (cutoff_24h,))
        trades_rows = [dict(r) for r in cursor.fetchall()]

        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM trade_events
            WHERE timestamp >= ?
            GROUP BY event_type;
        """, (cutoff_24h,))
        events_rows = [dict(r) for r in cursor.fetchall()]
        conn.close()

        return _compute_stats_from_records(trades_rows, events_rows, inr_rate, is_neon=False)
    except Exception as e:
        logger.error(f"[SQLITE] Failed to calculate 24h analytics: {e}")
        return _empty_analytics(inr_rate)


def _compute_stats_from_records(trades_rows: List[Any], events_rows: List[Any], inr_rate: float, is_neon: bool) -> Dict[str, Any]:
    """Helper to calculate financial metrics from raw rows."""
    event_counts = {}
    for row in events_rows:
        e_type = row["event_type"] if isinstance(row, dict) else row["event_type"]
        cnt = row["count"] if isinstance(row, dict) else row["count"]
        event_counts[e_type] = int(cnt)

    total_closed = len(trades_rows)
    gross_profit_usdt = 0.0
    gross_loss_usdt = 0.0
    total_fees_usdt = 0.0
    winning_trades = 0
    losing_trades = 0

    tp2_hit_count = 0
    sl_hit_count = 0
    breakeven_hit_count = 0

    for r in trades_rows:
        net_pnl = float(r["net_realized_pnl"] or 0.0)
        fees = float(r["total_fees_paid"] or 0.0)
        state = str(r["current_state"] or "")
        total_fees_usdt += fees

        if net_pnl > 0:
            gross_profit_usdt += net_pnl
            winning_trades += 1
        elif net_pnl < 0:
            gross_loss_usdt += abs(net_pnl)
            losing_trades += 1

        if "TP2_HIT" in state:
            tp2_hit_count += 1
        elif "STOP_LOSS" in state:
            sl_hit_count += 1
        elif "BREAKEVEN" in state:
            breakeven_hit_count += 1

    # TP1 hits can be determined both from trade_events table and from partial closed flags
    tp1_hit_count = event_counts.get("TP1_HIT", 0) + event_counts.get("WIN_WIN_BREAKEVEN_LOCKED", 0)
    # If no event row recorded yet, check trades that reached risk free or tp2
    if tp1_hit_count == 0:
        for r in trades_rows:
            is_rf = r.get("is_risk_free") if isinstance(r, dict) else r["is_risk_free"]
            c_state = str(r.get("current_state") if isinstance(r, dict) else r["current_state"])
            if bool(is_rf) or "TP2_HIT" in c_state or "BREAKEVEN" in c_state:
                tp1_hit_count += 1

    # Adjust event counts with direct event table values if present
    if event_counts.get("TP2_HIT", 0) > tp2_hit_count:
        tp2_hit_count = event_counts.get("TP2_HIT", 0)
    if event_counts.get("STOP_LOSS_HIT", 0) > sl_hit_count:
        sl_hit_count = event_counts.get("STOP_LOSS_HIT", 0)
    if event_counts.get("BREAKEVEN_HIT", 0) > breakeven_hit_count:
        breakeven_hit_count = event_counts.get("BREAKEVEN_HIT", 0)

    net_pnl_usdt = gross_profit_usdt - gross_loss_usdt
    gross_profit_inr = gross_profit_usdt * inr_rate
    gross_loss_inr = gross_loss_usdt * inr_rate
    net_pnl_inr = net_pnl_usdt * inr_rate
    total_fees_inr = total_fees_usdt * inr_rate

    win_rate = round((winning_trades / total_closed * 100.0), 2) if total_closed > 0 else 0.0
    profit_factor = round(gross_profit_usdt / max(0.001, gross_loss_usdt), 2) if gross_loss_usdt > 0 else (round(gross_profit_usdt, 2) if gross_profit_usdt > 0 else 0.0)

    return {
        "status": "success",
        "database": "Neon Cloud PostgreSQL (Primary)" if is_neon else "SQLite WAL (Local Offline Fallback)",
        "neon_connected": neon_manager.is_connected,
        "period": "24h_rolling",
        "tp1_hits": tp1_hit_count,
        "tp2_hits": tp2_hit_count,
        "sl_hits": sl_hit_count,
        "breakeven_exits": breakeven_hit_count,
        "total_trades_24h": total_closed,
        "winning_trades_24h": winning_trades,
        "losing_trades_24h": losing_trades,
        "win_rate_24h_pct": win_rate,
        "profit_factor_24h": profit_factor,
        "earned_24h_usdt": round(gross_profit_usdt, 4),
        "lost_24h_usdt": round(gross_loss_usdt, 4),
        "net_pnl_24h_usdt": round(net_pnl_usdt, 4),
        "earned_24h_inr": round(gross_profit_inr, 2),
        "lost_24h_inr": round(gross_loss_inr, 2),
        "net_pnl_24h_inr": round(net_pnl_inr, 2),
        "total_fees_24h_usdt": round(total_fees_usdt, 4),
        "total_fees_24h_inr": round(total_fees_inr, 2),
        "inr_rate": inr_rate,
        "updated_at": time.time()
    }


def _empty_analytics(inr_rate: float) -> Dict[str, Any]:
    """Returns zeroed analytics structure if no data available."""
    return {
        "status": "success",
        "database": "SQLite WAL (Default)",
        "neon_connected": neon_manager.is_connected,
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
        "inr_rate": inr_rate,
        "updated_at": time.time()
    }
