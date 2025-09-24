# main.py
import os
import time
import logging
import threading
import sqlite3
import base64
import requests
import pytz
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence
import operator
import pandas as pd
import pandas_ta as ta
import numpy as np
import ccxt
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langsmith import Client
from langchain_core.messages import SystemMessage
from pydantic import BaseModel
from typing_extensions import Annotated
import json

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants from .env
SYMBOL = os.getenv("SYMBOL", "SYMBOL")
TIMEFRAME = os.getenv("TIMEFRAME", "TIMEFRAME")
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 15.0))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", 5.0))
AMOUNTS = float(os.getenv("AMOUNTS", "AMOUNTS"))
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "GITHUB_REPO")
GITHUB_PATH = os.getenv("GITHUB_PATH", "rnn_bot.db")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "LANGSMITH_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN", "BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID", "CHAT_ID")
DB_PATH = os.getenv("DB_PATH", "rnn_bot.db")
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 60))  # 5 minutes 300
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 20))  # Number of trades to buffer 10

# Initialize exchange
exchange = ccxt.binance({
    'apiKey': os.getenv("BINANCE_API_KEY", "BINANCE_API_KEY"),
    'secret': os.getenv("BINANCE_API_SECRET", "BINANCE_API_SECRET"),
    'enableRateLimit': True,
    'sandbox': True,  # Paper trading
})

# GitHub API setup
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# LangSmith client
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "LANGSMITH_API_KEY")
langsmith_client = Client()

# Database connection and lock
db_lock = threading.Lock()
conn = None
trade_buffer = []  # Buffer for batch inserts
last_db_hash = None  # Track database file hash for GitHub uploads

# Strategy parameters (initial defaults)
current_strategy = {
    "rsi_buy_threshold": 34.0,
    "rsi_sell_threshold": 52.0,
    "stop_loss_percent": STOP_LOSS_PERCENT,
    "take_profit_percent": TAKE_PROFIT_PERCENT,
    "macd_hollow_buy": 0.01,
    "macd_hollow_sell": 0.01,
    "stoch_k_buy": 0.41,
    "stoch_k_sell": 99.98,
    "strategy_id": "default_1"
}

def setup_database(first_attempt=False):
    global conn
    with db_lock:
        for attempt in range(3):
            try:
                logger.info(f"Database setup attempt {attempt + 1}/3")
                if not os.path.exists(DB_PATH):
                    logger.info(f"Creating new database at {DB_PATH}")
                    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
                else:
                    try:
                        test_conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
                        c = test_conn.cursor()
                        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        logger.info(f"Existing database found at {DB_PATH}, tables: {c.fetchall()}")
                        test_conn.close()
                    except sqlite3.DatabaseError as e:
                        logger.error(f"Corrupted database at {DB_PATH}: {e}")
                        os.remove(DB_PATH)
                        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)

                if not first_attempt:
                    if download_from_github('rnn_bot.db', DB_PATH):
                        try:
                            test_conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
                            c = test_conn.cursor()
                            c.execute("SELECT name FROM sqlite_master WHERE type='table';")
                            logger.info(f"Downloaded database is valid, tables: {c.fetchall()}")
                            test_conn.close()
                        except sqlite3.DatabaseError as e:
                            logger.error(f"Downloaded database corrupted: {e}")
                            os.remove(DB_PATH)
                            conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)

                if conn is None:
                    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)

                # Optimize SQLite settings
                c = conn.cursor()
                c.execute("PRAGMA synchronous = NORMAL")
                c.execute("PRAGMA journal_mode = WAL")

                # Create trades table
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades';")
                if not c.fetchone():
                    c.execute('''
                        CREATE TABLE trades (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            time TEXT NOT NULL,
                            action TEXT NOT NULL,
                            symbol TEXT NOT NULL,
                            price REAL,
                            open_price REAL,
                            close_price REAL,
                            stop_loss REAL,
                            take_profit REAL,
                            return_profit REAL,
                            total_return_profit REAL,
                            ema1 REAL,
                            ema2 REAL,
                            rsi REAL,
                            k REAL,
                            d REAL,
                            j REAL,
                            diff REAL,
                            diff1e REAL,
                            diff2m REAL,
                            diff3k REAL,
                            macd REAL,
                            macd_signal REAL,
                            macd_hist REAL,
                            macd_hollow REAL,
                            lst_diff REAL,
                            supertrend REAL,
                            supertrend_trend TEXT,
                            stoch_rsi REAL,
                            stoch_k REAL,
                            stoch_d REAL,
                            obv REAL,
                            message TEXT,
                            timeframe TEXT,
                            order_id TEXT,
                            strategy_id TEXT
                        )
                    ''')
                    c.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(time)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_trades_action ON trades(action)")
                    conn.commit()

                # Create strategies table
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategies';")
                if not c.fetchone():
                    c.execute('''
                        CREATE TABLE strategies (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            strategy_id TEXT NOT NULL,
                            time TEXT NOT NULL,
                            parameters TEXT NOT NULL,
                            win_rate REAL,
                            total_profit REAL,
                            sharpe_ratio REAL,
                            num_trades INTEGER,
                            last_updated TEXT
                        )
                    ''')
                    c.execute("CREATE INDEX IF NOT EXISTS idx_strategies_id ON strategies(strategy_id)")
                    conn.commit()

                # Verify and add missing columns for trades
                c.execute("PRAGMA table_info(trades);")
                existing_columns = {col[1] for col in c.fetchall()}
                required_columns = {
                    'time': 'TEXT NOT NULL',
                    'action': 'TEXT NOT NULL',
                    'symbol': 'TEXT NOT NULL',
                    'price': 'REAL',
                    'open_price': 'REAL',
                    'close_price': 'REAL',
                    'stop_loss': 'REAL',
                    'take_profit': 'REAL',
                    'return_profit': 'REAL',
                    'total_return_profit': 'REAL',
                    'ema1': 'REAL',
                    'ema2': 'REAL',
                    'rsi': 'REAL',
                    'k': 'REAL',
                    'd': 'REAL',
                    'j': 'REAL',
                    'diff': 'REAL',
                    'diff1e': 'REAL',
                    'diff2m': 'REAL',
                    'diff3k': 'REAL',
                    'macd': 'REAL',
                    'macd_signal': 'REAL',
                    'macd_hist': 'REAL',
                    'macd_hollow': 'REAL',
                    'lst_diff': 'REAL',
                    'supertrend': 'REAL',
                    'supertrend_trend': 'TEXT',
                    'stoch_rsi': 'REAL',
                    'stoch_k': 'REAL',
                    'stoch_d': 'REAL',
                    'obv': 'REAL',
                    'message': 'TEXT',
                    'timeframe': 'TEXT',
                    'order_id': 'TEXT',
                    'strategy_id': 'TEXT'
                }
                for col, col_type in required_columns.items():
                    if col not in existing_columns:
                        c.execute(f"ALTER TABLE trades ADD COLUMN {col} {col_type};")
                        conn.commit()

                logger.info(f"Database initialized at {DB_PATH}, size: {os.path.getsize(DB_PATH)} bytes")
                return True
            except Exception as e:
                logger.error(f"Error during database setup: {e}")
                if conn:
                    conn.close()
                    conn = None
                time.sleep(2)
        logger.error("Failed to initialize database. Forcing new creation.")
        try:
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
            conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
            c = conn.cursor()
            c.execute("PRAGMA synchronous = NORMAL")
            c.execute("PRAGMA journal_mode = WAL")
            c.execute('''
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT NOT NULL,
                    action TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL,
                    open_price REAL,
                    close_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    return_profit REAL,
                    total_return_profit REAL,
                    ema1 REAL,
                    ema2 REAL,
                    rsi REAL,
                    k REAL,
                    d REAL,
                    j REAL,
                    diff REAL,
                    diff1e REAL,
                    diff2m REAL,
                    diff3k REAL,
                    macd REAL,
                    macd_signal REAL,
                    macd_hist REAL,
                    macd_hollow REAL,
                    lst_diff REAL,
                    supertrend REAL,
                    supertrend_trend TEXT,
                    stoch_rsi REAL,
                    stoch_k REAL,
                    stoch_d REAL,
                    obv REAL,
                    message TEXT,
                    timeframe TEXT,
                    order_id TEXT,
                    strategy_id TEXT
                )
            ''')
            c.execute('''
                CREATE TABLE strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    time TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    win_rate REAL,
                    total_profit REAL,
                    sharpe_ratio REAL,
                    num_trades INTEGER,
                    last_updated TEXT
                )
            ''')
            c.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(time)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_trades_action ON trades(action)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_strategies_id ON strategies(strategy_id)")
            conn.commit()
            logger.info(f"Forced new database at {DB_PATH}")
            return True
        except Exception as e:
            logger.error(f"Failed to force create database: {e}")
            return False

def upload_to_github(file_path, file_name):
    try:
        with open(file_path, "rb") as f:
            content = f.read()
            current_hash = hashlib.sha256(content).hexdigest()
        global last_db_hash
        if last_db_hash == current_hash:
            logger.debug(f"No changes in {file_name}. Skipping GitHub upload.")
            return
        content_b64 = base64.b64encode(content).decode("utf-8")
        response = requests.get(GITHUB_API_URL, headers=HEADERS)
        sha = response.json().get("sha") if response.status_code == 200 else None
        payload = {
            "message": f"Update {file_name} at {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')}",
            "content": content_b64
        }
        if sha:
            payload["sha"] = sha
        response = requests.put(GITHUB_API_URL, headers=HEADERS, json=payload)
        if response.status_code in [200, 201]:
            last_db_hash = current_hash
            logger.info(f"Uploaded {file_name} to GitHub")
        else:
            logger.error(f"Failed to upload {file_name}: {response.text}")
    except Exception as e:
        logger.error(f"Error uploading {file_name} to GitHub: {e}")

def download_from_github(file_name, destination_path):
    try:
        response = requests.get(GITHUB_API_URL, headers=HEADERS)
        if response.status_code == 404:
            logger.info(f"No {file_name} found in GitHub. Starting with new database.")
            return False
        elif response.status_code != 200:
            logger.error(f"Failed to fetch {file_name}: {response.text}")
            return False
        content = base64.b64decode(response.json()["content"])
        with open(destination_path, "wb") as f:
            f.write(content)
        logger.info(f"Downloaded {file_name} from GitHub to {destination_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {file_name}: {e}")
        return False

def periodic_db_backup():
    while True:
        try:
            with db_lock:
                if trade_buffer:
                    flush_trade_buffer()
                if os.path.exists(DB_PATH) and conn:
                    upload_to_github(DB_PATH, 'rnn_bot.db')
            time.sleep(UPDATE_INTERVAL)
        except Exception as e:
            logger.error(f"Error during periodic backup: {e}")
            time.sleep(60)

def flush_trade_buffer():
    global trade_buffer
    if not trade_buffer:
        return
    try:
        with db_lock:
            c = conn.cursor()
            c.executemany('''
                INSERT INTO trades (
                    time, action, symbol, price, open_price, close_price,
                    stop_loss, take_profit, return_profit, total_return_profit,
                    ema1, ema2, rsi, k, d, j, diff, diff1e, diff2m, diff3k,
                    macd, macd_signal, macd_hist, macd_hollow, lst_diff,
                    supertrend, supertrend_trend, stoch_rsi, stoch_k, stoch_d,
                    obv, message, timeframe, order_id, strategy_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', trade_buffer)
            conn.commit()
            logger.info(f"Flushed {len(trade_buffer)} trades to database")
            trade_buffer = []
    except Exception as e:
        logger.error(f"Error flushing trade buffer: {e}")

@tool
def fetch_crypto_data(symbol: str = SYMBOL, timeframe: str = TIMEFRAME, limit: int = 100) -> Dict[str, Any]:
    """Fetch OHLCV data and calculate technical indicators for a given symbol and timeframe."""
    try:
        start_time = time.time()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # ... (rest of the function remains unchanged)
        return {
            "symbol": symbol,
            "data": df.to_dict('records'),
            "indicators": {
                "close_price": latest['Close'],
                # ... (rest of the return statement unchanged)
            }
        }
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return {"error": str(e)}

@tool
def execute_trade(action: str, symbol: str, amount: float, price: float = None) -> Dict[str, Any]:
    """Execute a buy or sell trade on the specified symbol with the given amount."""
    try:
        market = exchange.load_markets()[symbol]
        quantity = exchange.amount_to_precision(symbol, amount)
        # ... (rest of the function unchanged)
        return {
            "order_id": order['id'],
            "status": order['status'],
            "filled": order['filled'],
            "price": price or order.get('average', order['price']),
            "symbol": symbol
        }
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return {"error": str(e)}

@tool
def get_portfolio_balance() -> Dict[str, Any]:
    """Retrieve the current portfolio balance for USDT and the traded asset."""
    try:
        balance = exchange.fetch_balance()
        return {
            "usdt_balance": balance['USDT']['free'],
            "asset_balance": balance[SYMBOL.split("/")[0]]['free']
        }
    except Exception as e:
        logger.error(f"Error fetching balance: {e}")
        return {"error": str(e)}

@tool
def store_trade(state: Dict[str, Any]) -> Dict[str, Any]:
    """Store trade data in the SQLite database."""
    global trade_buffer
    try:
        indicators = state.get("indicators", {})
        trade_data = (
            datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S'),
            # ... (rest of the function unchanged)
        )
        trade_buffer.append(trade_data)
        if len(trade_buffer) >= BUFFER_SIZE:
            flush_trade_buffer()
        return {"status": "success", "message": "Trade buffered for storage"}
    except Exception as e:
        logger.error(f"Error buffering trade: {e}")
        return {"status": "error", "message": str(e)}

@tool
def backtest_strategy(symbol: str, timeframe: str, strategy_params: Dict[str, Any], limit: int = 500) -> Dict[str, Any]:
    """Backtest a trading strategy on historical data."""
    try:
        data = fetch_crypto_data(symbol, timeframe, limit)
        if "error" in data:
            return {"error": data["error"]}
        # ... (rest of the function unchanged)
        return {
            "strategy_id": strategy_params["strategy_id"],
            "win_rate": win_rate,
            "total_profit": total_profit,
            "sharpe_ratio": sharpe_ratio,
            "num_trades": total_trades,
            "trades": trades
        }
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        return {"error": str(e)}

@tool
def optimize_strategy(state: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize trading strategy based on recent trade performance."""
    global current_strategy
    try:
        with db_lock:
            c = conn.cursor()
            c.execute("SELECT action, return_profit FROM trades WHERE strategy_id = ? ORDER BY time DESC LIMIT 10", 
                     (current_strategy["strategy_id"],))
            recent_trades = c.fetchall()
        # ... (rest of the function unchanged)
        return {
            "strategy_id": new_strategy["strategy_id"],
            "parameters": new_strategy,
            "win_rate": backtest_result["win_rate"],
            "total_profit": backtest_result["total_profit"],
            "sharpe_ratio": backtest_result["sharpe_ratio"]
        }
    except Exception as e:
        logger.error(f"Error in strategy optimization: {e}")
        return {"error": str(e)}

# State
class TradingState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add]
    symbol: str
    portfolio: Dict[str, Any]
    indicators: Dict[str, Any]
    signal: str
    risk_approved: bool
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_id: Optional[str] = None
    position: Optional[str] = None
    buy_price: Optional[float] = None
    last_sell_profit: float
    total_return_profit: float
    tracking_enabled: bool
    tracking_has_buy: bool
    tracking_buy_price: Optional[float] = None
    strategy_id: str
    backtest_results: Dict[str, Any]
    next: str

logger.info(f"GROQ_API_KEY: {os.getenv('GROQ_API_KEY')}")

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Agent Prompts
signal_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a crypto trading signal generator. Use these rules with provided strategy parameters:
    Buy if: (lst_diff > 0.01 and macd_hollow <= {macd_hollow_buy} and stoch_rsi <= 0.01 and stoch_k <= {stoch_k_buy} and stoch_d <= 25.00 and obv <= -1093.00 and rsi < {rsi_buy_threshold})
    OR (j < d and j < -15.00 and macd < macd_signal and ema1 < ema2 and rsi < 17.00)
    OR (lst_diff > 0.01 and macd_hollow <= {macd_hollow_buy} and stoch_k <= {stoch_k_buy} and macd < macd_signal and rsi <= {rsi_buy_threshold})
    OR (supertrend_trend == 'Down' and stoch_rsi <= 0.00 and stoch_k <= 0.00 and stoch_d <= 0.15 and obv <= -13.00 and diff1e < 0.00 and diff2m < 0.00 and diff3k < 0.00).
    Sell if in position and: (close_price <= stop_loss)
    OR (close_price >= take_profit)
    OR (lst_diff < 0.00 and macd_hollow >= {macd_hollow_sell} and stoch_rsi >= 0.99 and stoch_k >= {stoch_k_sell} and stoch_d >= 94.97 and obv >= 1009.00 and diff1e > 0.00)
    OR (j > d and j > 115.00 and macd > macd_signal and ema1 > ema2 and rsi < {rsi_sell_threshold})
    OR (lst_diff < 0.00 and macd_hollow >= {macd_hollow_sell} and stoch_k >= {stoch_k_sell} and macd > macd_signal and rsi <= {rsi_sell_threshold})
    OR (supertrend_trend == 'Up' and stoch_rsi == 1.00 and stoch_k == 100.00 and stoch_d > 90.00 and obv >= 19.00 and diff1e > 0.00 and diff2m > 0.00 and diff3k > 0.00).
    Else, hold. Return JSON: {{"signal": "buy/sell/hold", "reason": "..."}}.""".format(**current_strategy)),
    ("human", "{input}")
])

risk_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a risk manager. Validate trades:
    - Max 2% portfolio risk.
    - Set stop_loss = buy_price * (1 - {stop_loss_percent}/100).
    - Set take_profit = buy_price * (1 + {take_profit_percent}/100).
    - Prevent consecutive buys or sells without position.
    - Amount = min(usdt_balance * 0.02 / close_price, usdt_balance / close_price).
    Return JSON: {{"approved": bool, "stop_loss": float, "take_profit": float, "amount": float, "reason": "..."}}.""".format(
        stop_loss_percent=current_strategy["stop_loss_percent"], 
        take_profit_percent=current_strategy["take_profit_percent"])),
    ("human", "{input}")
])

profit_prompt = ChatPromptTemplate.from_messages([
    ("system", """Implement profit tracking:
    - If action == buy and last_sell_profit > 0: set tracking_has_buy=True, tracking_buy_price=current_price.
    - If action == sell and tracking_has_buy: compute return_profit = current_price - tracking_buy_price, update total_return_profit, set tracking_has_buy=False, tracking_enabled=True if last_sell_profit > 0 else False.
    - If action == sell and not tracking_has_buy: set tracking_enabled based on last_sell_profit.
    - Pause buys/sells if tracking_enabled=False.
    Return JSON: {{"action": str, "return_profit": float, "tracking_enabled": bool, "tracking_has_buy": bool, "total_return_profit": float, "msg": "..."}}."""),
    ("human", "{input}")
])

executor_prompt = ChatPromptTemplate.from_messages([
    ("system", """Execute approved trades via CCXT. Confirm details. Return JSON: {{"order_id": str, "status": str, "filled": float, "price": float}}."""),
    ("human", "{input}")
])

db_prompt = ChatPromptTemplate.from_messages([
    ("system", """Store trade data and signals in SQLite database. Return JSON: {{"status": str, "message": str}}."""),
    ("human", "{input}")
])

backtest_prompt = ChatPromptTemplate.from_messages([
    ("system", """Backtest a trading strategy on historical data. Return JSON: {{"strategy_id": str, "win_rate": float, "total_profit": float, "sharpe_ratio": float, "num_trades": int, "trades": list}}."""),
    ("human", "{input}")
])

optimize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Optimize trading strategy based on recent performance and market conditions. 
    - Analyze last 10 trades for win rate and losses.
    - If 3+ consecutive losses or win rate < 0.5, adjust parameters (e.g., RSI thresholds, stop-loss) based on volatility.
    - Backtest new strategy and store in database.
    Return JSON: {{"strategy_id": str, "parameters": dict, "win_rate": float, "total_profit": float, "sharpe_ratio": float}}."""),
    ("human", "{input}")
])

# Extract prompt content safely
def extract_prompt_content(prompt):
    if isinstance(prompt, ChatPromptTemplate) and prompt.messages:
        first_message = prompt.messages[0]
        if hasattr(first_message, 'prompt') and hasattr(first_message.prompt, 'template'):
            return first_message.prompt.template
        else:
            logger.error(f"Invalid ChatPromptTemplate structure: {prompt}")
            raise ValueError("ChatPromptTemplate does not contain a valid SystemMessage with template")
    elif isinstance(prompt, PromptTemplate):
        return prompt.template
    else:
        logger.error(f"Unsupported prompt type: {type(prompt)}")
        raise ValueError(f"Prompt must be ChatPromptTemplate or PromptTemplate, got {type(prompt)}")

# Create agents
try:
    signal_prompt_content = extract_prompt_content(signal_prompt)
    risk_prompt_content = extract_prompt_content(risk_prompt)
    executor_prompt_content = extract_prompt_content(executor_prompt)
    db_prompt_content = extract_prompt_content(db_prompt)
    backtest_prompt_content = extract_prompt_content(backtest_prompt)
    optimize_prompt_content = extract_prompt_content(optimize_prompt)
    profit_prompt_content = extract_prompt_content(profit_prompt)
except ValueError as e:
    logger.critical(f"Prompt extraction error: {e}")
    exit(1)

signal_agent = create_react_agent(
    llm,
    tools=[fetch_crypto_data],
    messages_modifier=SystemMessage(content=signal_prompt_content),
    checkpointer=MemorySaver()
)
risk_agent = create_react_agent(
    llm,
    tools=[get_portfolio_balance],
    messages_modifier=SystemMessage(content=risk_prompt_content),
    checkpointer=MemorySaver()
)
executor_agent = create_react_agent(
    llm,
    tools=[execute_trade],
    messages_modifier=SystemMessage(content=executor_prompt_content),
    checkpointer=MemorySaver()
)
db_agent = create_react_agent(
    llm,
    tools=[store_trade],
    messages_modifier=SystemMessage(content=db_prompt_content),
    checkpointer=MemorySaver()
)
backtest_agent = create_react_agent(
    llm,
    tools=[backtest_strategy],
    messages_modifier=SystemMessage(content=backtest_prompt_content),
    checkpointer=MemorySaver()
)
optimize_agent = create_react_agent(
    llm,
    tools=[optimize_strategy],
    messages_modifier=SystemMessage(content=optimize_prompt_content),
    checkpointer=MemorySaver()
)
profit_agent = create_react_agent(
    llm,
    tools=[],
    messages_modifier=SystemMessage(content=profit_prompt_content),
    checkpointer=MemorySaver()
)
data_agent = create_react_agent(
    llm,
    tools=[fetch_crypto_data],
    checkpointer=MemorySaver()
)

# Agent Nodes
def call_data_agent(state: TradingState) -> TradingState:
    result = data_agent.invoke({"input": f"Fetch data for {state['symbol']}"})
    return {
        "messages": result["messages"],
        "indicators": result.get("indicators", {}),
        "portfolio": get_portfolio_balance(),
        "strategy_id": current_strategy["strategy_id"]
    }

def call_signal_agent(state: TradingState) -> TradingState:
    input_data = f"Indicators: {state['indicators']}, Position: {state.get('position')}, Buy Price: {state.get('buy_price')}, Strategy: {current_strategy}"
    result = signal_agent.invoke({"input": input_data})
    try:
        signal_data = eval(result["messages"][-1].content)
    except:
        signal_data = {"signal": "hold", "reason": "Error parsing signal"}
    return {
        "messages": result["messages"],
        "signal": signal_data["signal"],
        "reason": signal_data["reason"]
    }

def call_risk_agent(state: TradingState) -> TradingState:
    input_data = f"Signal: {state['signal']}, Portfolio: {state['portfolio']}, Indicators: {state['indicators']}, Position: {state.get('position')}"
    result = risk_agent.invoke({"input": input_data})
    try:
        risk_data = eval(result["messages"][-1].content)
    except:
        risk_data = {"approved": False, "reason": "Error parsing risk"}
    return {
        "messages": result["messages"],
        "risk_approved": risk_data["approved"],
        "stop_loss": risk_data.get("stop_loss"),
        "take_profit": risk_data.get("take_profit"),
        "amount": risk_data.get("amount"),
        "reason": risk_data["reason"]
    }

def call_profit_agent(state: TradingState) -> TradingState:
    current_price = state["indicators"].get("close_price", 0)
    primary_profit = (current_price - state.get("buy_price", 0)) if state["signal"] == "sell" else 0
    input_data = f"Action: {state['signal']}, Current Price: {current_price}, Primary Profit: {primary_profit}, Tracking: {{enabled: {state.get('tracking_enabled', False)}, has_buy: {state.get('tracking_has_buy', False)}, buy_price: {state.get('tracking_buy_price', 0)}}}"
    result = profit_agent.invoke({"input": input_data})
    try:
        profit_data = eval(result["messages"][-1].content)
    except:
        profit_data = {"action": "hold", "return_profit": 0, "tracking_enabled": state.get("tracking_enabled", True), "tracking_has_buy": state.get("tracking_has_buy", False), "total_return_profit": state.get("total_return_profit", 0), "msg": "Error parsing profit"}
    return {
        "messages": result["messages"],
        "signal": profit_data["action"],
        "return_profit": profit_data["return_profit"],
        "tracking_enabled": profit_data["tracking_enabled"],
        "tracking_has_buy": profit_data["tracking_has_buy"],
        "total_return_profit": profit_data["total_return_profit"],
        "reason": profit_data["msg"]
    }

def call_executor_agent(state: TradingState) -> TradingState:
    if not state["risk_approved"] or state["signal"] == "hold" or not state.get("tracking_enabled", True):
        return state
    input_data = f"Execute {state['signal']} on {state['symbol']} for {state['amount']} at {state['indicators']['close_price']}"
    result = executor_agent.invoke({"input": input_data})
    try:
        order_data = eval(result["messages"][-1].content)
    except:
        order_data = {"order_id": None, "status": "failed", "filled": 0, "price": state["indicators"]["close_price"]}
    return {
        "messages": result["messages"],
        "order_id": order_data.get("order_id"),
        "position": "long" if state["signal"] == "buy" else None,
        "buy_price": state["indicators"]["close_price"] if state["signal"] == "buy" else None
    }

def call_db_agent(state: TradingState) -> TradingState:
    input_data = f"Store trade data: {state}"
    result = db_agent.invoke({"input": input_data})
    try:
        db_data = eval(result["messages"][-1].content)
    except:
        db_data = {"status": "error", "message": "Error storing trade"}
    return {
        "messages": result["messages"],
        "db_status": db_data["status"],
        "db_message": db_data["message"]
    }

def call_backtest_agent(state: TradingState) -> TradingState:
    input_data = f"Backtest strategy {current_strategy['strategy_id']} on {state['symbol']} with timeframe {TIMEFRAME}"
    result = backtest_agent.invoke({"input": input_data})
    try:
        backtest_data = eval(result["messages"][-1].content)
    except:
        backtest_data = {"error": "Error in backtesting"}
    return {
        "messages": result["messages"],
        "backtest_results": backtest_data
    }

def call_optimize_agent(state: TradingState) -> TradingState:
    input_data = f"Optimize strategy based on state: {state}"
    result = optimize_agent.invoke({"input": input_data})
    try:
        optimize_data = eval(result["messages"][-1].content)
    except:
        optimize_data = {"error": "Error in optimization"}
    return {
        "messages": result["messages"],
        "strategy_id": optimize_data.get("strategy_id", state.get("strategy_id", "default_1")),
        "backtest_results": optimize_data
    }

# Supervisor Routing
def supervisor(state: TradingState) -> str:
    if not state.get("indicators"):
        return "data"
    if not state.get("signal"):
        return "signal"
    if state["signal"] != "hold" and not state.get("risk_approved"):
        return "risk"
    if state["signal"] != "hold" and state["risk_approved"]:
        return "profit"
    if state.get("tracking_enabled", True):
        return "execute"
    if state.get("order_id") or state["signal"] != "hold":
        return "db"
    if not state.get("backtest_results"):
        return "backtest"
    # Trigger optimization if recent performance is poor
    with db_lock:
        c = conn.cursor()
        c.execute("SELECT action, return_profit FROM trades WHERE strategy_id = ? ORDER BY time DESC LIMIT 10", 
                 (state.get("strategy_id", "default_1"),))
        recent_trades = c.fetchall()
    losses = sum(1 for action, profit in recent_trades if action == "sell" and profit < 0)
    if losses >= 3:
        return "optimize"
    return END

# Build Graph
workflow = StateGraph(TradingState)
workflow.add_node("data", call_data_agent)
workflow.add_node("signal", call_signal_agent)
workflow.add_node("risk", call_risk_agent)
workflow.add_node("profit", call_profit_agent)
workflow.add_node("execute", call_executor_agent)
workflow.add_node("db", call_db_agent)
workflow.add_node("backtest", call_backtest_agent)
workflow.add_node("optimize", call_optimize_agent)
workflow.add_node("supervisor", supervisor)

workflow.add_edge("data", "supervisor")
workflow.add_edge("signal", "supervisor")
workflow.add_edge("risk", "supervisor")
workflow.add_edge("profit", "supervisor")
workflow.add_edge("execute", "supervisor")
workflow.add_edge("db", "supervisor")
workflow.add_edge("backtest", "supervisor")
workflow.add_edge("optimize", "supervisor")
workflow.set_entry_point("supervisor")

# Compile with memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Initialize database
if not setup_database(first_attempt=True):
    logger.critical("Failed to initialize database. Exiting.")
    exit(1)

# Start periodic backup thread
backup_thread = threading.Thread(target=periodic_db_backup, daemon=True)
backup_thread.start()

# Run Loop
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "crypto_trader_1"}}
    initial_state = {
        "symbol": SYMBOL,
        "messages": [HumanMessage(content="Start monitoring SOL/USDT for trades.")],
        "last_sell_profit": 0.0,
        "total_return_profit": 0.0,
        "tracking_enabled": True,
        "tracking_has_buy": False,
        "tracking_buy_price": None,
        "strategy_id": current_strategy["strategy_id"]
    }

    while True:
        with langsmith_client.trace("TradingCycle") as trace:
            result = app.invoke(initial_state, config)
            trace.add_metadata({"state": result})
            logger.info(f"State: {result}")
            if result.get("order_id"):
                logger.info(f"Trade executed: {result['signal']} at {result.get('price', 'unknown')}, Order ID: {result['order_id']}")

        time.sleep(300)  # Check hourly 3600





