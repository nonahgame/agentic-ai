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
from typing import Dict, Any, List, Optional
import pandas as pd
import pandas_ta as ta
import numpy as np
import ccxt
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage  # Updated import
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate  # Removed SystemMessage from here
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing_extensions import Annotated
import uvicorn
import warnings

# Suppress specific pandas_ta warning about pkg_resources
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pandas_ta",
    message="pkg_resources is deprecated as an API.*"
)

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
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_PATH = os.getenv("GITHUB_PATH", "GITHUB_PATH")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DB_PATH = os.getenv("DB_PATH", "DB_PATH")
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 60))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 20))

# Initialize exchange
exchange = ccxt.binance({
    'apiKey': os.getenv("BINANCE_API_KEY"),
    'secret': os.getenv("BINANCE_API_SECRET"),
    'enableRateLimit': True,
    'sandbox': True,
})

# GitHub API setup
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# LangSmith client
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
langsmith_client = Client()

# Database setup
db_lock = threading.Lock()
conn = None
trade_buffer = []
last_db_hash = None

# Strategy parameters
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

# FastAPI app
app = FastAPI(title="Crypto Trading Bot API")

# Pydantic models for request validation
class TradeRequest(BaseModel):
    symbol: str = SYMBOL
    amount: float = AMOUNTS
    action: str

class StrategyRequest(BaseModel):
    symbol: str = SYMBOL
    timeframe: str = TIMEFRAME
    limit: int = 500

class OptimizeRequest(BaseModel):
    strategy_id: str = current_strategy["strategy_id"]

# Database setup function (unchanged)
def setup_database(first_attempt=False):
    global conn
    with db_lock:
        try:
            if not os.path.exists(DB_PATH):
                logger.info(f"Creating new database at {DB_PATH}")
                conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
            else:
                test_conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
                c = test_conn.cursor()
                c.execute("SELECT name FROM sqlite_master WHERE type='table';")
                logger.info(f"Existing database found, tables: {c.fetchall()}")
                test_conn.close()
                conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)

            c = conn.cursor()
            c.execute("PRAGMA synchronous = NORMAL")
            c.execute("PRAGMA journal_mode = WAL")

            # Create tables (trades and strategies)
            c.execute('''
                CREATE TABLE IF NOT EXISTS trades (
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
                CREATE TABLE IF NOT EXISTS strategies (
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
            logger.info(f"Database initialized at {DB_PATH}")
            return True
        except Exception as e:
            logger.error(f"Database setup error: {e}")
            return False

# GitHub upload/download (unchanged)
def upload_to_github(file_path, file_name):
    try:
        with open(file_path, "rb") as f:
            content = f.read()
            current_hash = hashlib.sha256(content).hexdigest()
        global last_db_hash
        if last_db_hash == current_hash:
            logger.debug(f"No changes in {file_name}. Skipping upload.")
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
        logger.error(f"Error uploading {file_name}: {e}")

def download_from_github(file_name, destination_path):
    try:
        response = requests.get(GITHUB_API_URL, headers=HEADERS)
        if response.status_code == 404:
            logger.info(f"No {file_name} found in GitHub.")
            return False
        elif response.status_code != 200:
            logger.error(f"Failed to fetch {file_name}: {response.text}")
            return False
        content = base64.b64decode(response.json()["content"])
        with open(destination_path, "wb") as f:
            f.write(content)
        logger.info(f"Downloaded {file_name} to {destination_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {file_name}: {e}")
        return False

# Periodic backup as a background task
async def periodic_db_backup():
    while True:
        try:
            with db_lock:
                if trade_buffer:
                    flush_trade_buffer()
                if os.path.exists(DB_PATH) and conn:
                    upload_to_github(DB_PATH, 'rnn_bot.db')
            await asyncio.sleep(UPDATE_INTERVAL)
        except Exception as e:
            logger.error(f"Error during periodic backup: {e}")
            await asyncio.sleep(60)

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

# Tools (simplified for brevity, implement as needed)
@tool
async def fetch_crypto_data(symbol: str = SYMBOL, timeframe: str = TIMEFRAME, limit: int = 100) -> Dict[str, Any]:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Add technical indicators (EMA, RSI, MACD, etc.)
        df['ema1'] = ta.ema(df['Close'], length=12)
        df['ema2'] = ta.ema(df['Close'], length=26)
        df['rsi'] = ta.rsi(df['Close'], length=14)
        # Add more indicators as needed
        latest = df.iloc[-1]
        return {
            "symbol": symbol,
            "data": df.to_dict('records'),
            "indicators": {
                "close_price": latest['Close'],
                "ema1": latest['ema1'],
                "ema2": latest['ema2'],
                "rsi": latest['rsi']
            }
        }
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return {"error": str(e)}

@tool
async def execute_trade(action: str, symbol: str, amount: float, price: float = None) -> Dict[str, Any]:
    try:
        market = exchange.load_markets()[symbol]
        quantity = exchange.amount_to_precision(symbol, amount)
        if action == "buy":
            order = exchange.create_market_buy_order(symbol, quantity)
        elif action == "sell":
            order = exchange.create_market_sell_order(symbol, quantity)
        else:
            return {"error": "Invalid action"}
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
async def get_portfolio_balance() -> Dict[str, Any]:
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
async def store_trade(state: Dict[str, Any]) -> Dict[str, Any]:
    global trade_buffer
    try:
        indicators = state.get("indicators", {})
        trade_data = (
            datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S'),
            state.get("signal", "hold"),
            state.get("symbol", SYMBOL),
            indicators.get("close_price"),
            indicators.get("open_price"),
            indicators.get("close_price"),
            state.get("stop_loss"),
            state.get("take_profit"),
            state.get("return_profit", 0.0),
            state.get("total_return_profit", 0.0),
            indicators.get("ema1"),
            indicators.get("ema2"),
            indicators.get("rsi"),
            indicators.get("k"),
            indicators.get("d"),
            indicators.get("j"),
            indicators.get("diff"),
            indicators.get("diff1e"),
            indicators.get("diff2m"),
            indicators.get("diff3k"),
            indicators.get("macd"),
            indicators.get("macd_signal"),
            indicators.get("macd_hist"),
            indicators.get("macd_hollow"),
            indicators.get("lst_diff"),
            indicators.get("supertrend"),
            indicators.get("supertrend_trend"),
            indicators.get("stoch_rsi"),
            indicators.get("stoch_k"),
            indicators.get("stoch_d"),
            indicators.get("obv"),
            state.get("reason", ""),
            state.get("timeframe", TIMEFRAME),
            state.get("order_id"),
            state.get("strategy_id", current_strategy["strategy_id"])
        )
        trade_buffer.append(trade_data)
        if len(trade_buffer) >= BUFFER_SIZE:
            flush_trade_buffer()
        return {"status": "success", "message": "Trade buffered"}
    except Exception as e:
        logger.error(f"Error buffering trade: {e}")
        return {"status": "error", "message": str(e)}

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# FastAPI Endpoints
@app.on_event("startup")
async def startup_event():
    if not setup_database(first_attempt=True):
        logger.critical("Failed to initialize database.")
        raise Exception("Database initialization failed")
    # Start periodic backup as a background task
    import asyncio
    asyncio.create_task(periodic_db_backup())

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/fetch-data")
async def fetch_data(request: StrategyRequest):
    result = await fetch_crypto_data(request.symbol, request.timeframe, request.limit)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/execute-trade")
async def execute_trade_endpoint(request: TradeRequest):
    if request.action not in ["buy", "sell"]:
        raise HTTPException(status_code=400, detail="Invalid action")
    result = await execute_trade(request.action, request.symbol, request.amount)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.get("/portfolio")
async def get_portfolio():
    result = await get_portfolio_balance()
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/store-trade")
async def store_trade_endpoint(state: Dict[str, Any]):
    result = await store_trade(state)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/current-strategy")
async def get_current_strategy():
    return current_strategy

@app.post("/run-trading-cycle")
async def run_trading_cycle(background_tasks: BackgroundTasks):
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
    result = app.invoke(initial_state, config)
    return {"status": "success", "state": result}

# Main entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
