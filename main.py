# main.py
import os
import logging
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
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langsmith import Client
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing_extensions import Annotated
import uvicorn
import warnings
import asyncio
import operator

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pandas_ta",
    message="pkg_resources is deprecated as an API.*"
)
warnings.filterwarnings(
    "ignore",
    category=Warning,
    module="langgraph.graph",
    message="As of langchain-core 0.3.0.*"
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
DB_PATH = os.getenv("DB_PATH", "rnn_bot.db")
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
    "macd_hollow_buy": -0.00,
    "macd_hollow_sell": 0.99,
    "stoch_k_buy": 0.41,
    "stoch_k_sell": 99.98,
    "strategy_id": "default_1"
}

# FastAPI app
app = FastAPI(title="Crypto Trading Bot API")

# Pydantic models
class TradeRequest(BaseModel):
    symbol: str = SYMBOL
    amount_usdt: float = AMOUNTS
    action: str

class StrategyRequest(BaseModel):
    symbol: str = SYMBOL
    timeframe: str = TIMEFRAME
    limit: int = 500

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
    total_return_profit_usdt: float
    tracking_enabled: bool
    tracking_has_buy: bool
    tracking_buy_price: Optional[float] = None
    strategy_id: str
    backtest_results: Dict[str, Any]
    next: Optional[str] = None

# Utility function to convert asset quantity to USDT
async def convert_to_usdt(symbol: str, quantity: float, exchange: ccxt.Exchange) -> float:
    """Convert an asset quantity to USDT using the current market price."""
    try:
        base, quote = symbol.split('/')
        if quote == 'USDT':
            ticker = exchange.fetch_ticker(symbol)
            price = ticker['last']
            return quantity * price
        else:
            usdt_pair = f"{quote}/USDT"
            ticker = exchange.fetch_ticker(usdt_pair)
            quote_to_usdt_price = ticker['last']
            base_to_quote_ticker = exchange.fetch_ticker(symbol)
            base_to_quote_price = base_to_quote_ticker['last']
            return quantity * base_to_quote_price * quote_to_usdt_price
    except Exception as e:
        logger.error(f"Error converting {quantity} {symbol} to USDT: {e}")
        return 0.0

# Database setup
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
            c.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT NOT NULL,
                    action TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL,
                    amount_usdt REAL,
                    return_profit_usdt REAL,
                    total_return_profit_usdt REAL,
                    ema1 REAL,
                    ema2 REAL,
                    rsi REAL,
                    stop_loss REAL,
                    take_profit REAL,
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
                    total_profit_usdt REAL,
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
                    time, action, symbol, price, amount_usdt, return_profit_usdt,
                    total_return_profit_usdt, ema1, ema2, rsi, stop_loss, take_profit,
                    message, timeframe, order_id, strategy_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', trade_buffer)
            conn.commit()
            logger.info(f"Flushed {len(trade_buffer)} trades to database")
            trade_buffer = []
    except Exception as e:
        logger.error(f"Error flushing trade buffer: {e}")

# Tools
@tool
async def fetch_crypto_data(symbol: str = SYMBOL, timeframe: str = TIMEFRAME, limit: int = 100) -> Dict[str, Any]:
    """Fetch OHLCV data and calculate technical indicators for a given symbol and timeframe."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['ema1'] = ta.ema(df['Close'], length=12)
        df['ema2'] = ta.ema(df['Close'], length=26)
        df['rsi'] = ta.rsi(df['Close'], length=14)
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
async def execute_trade(action: str, symbol: str, amount_usdt: float, price: float = None) -> Dict[str, Any]:
    """Execute a buy or sell trade with amount specified in USDT."""
    try:
        market = exchange.load_markets()[symbol]
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last'] if price is None else price
        quantity = exchange.amount_to_precision(symbol, amount_usdt / current_price)
        if action == "buy":
            order = exchange.create_market_buy_order(symbol, quantity)
        elif action == "sell":
            order = exchange.create_market_sell_order(symbol, quantity)
        else:
            return {"error": "Invalid action"}
        amount_usdt_filled = order['filled'] * order.get('average', order['price'])
        return {
            "order_id": order['id'],
            "status": order['status'],
            "filled": order['filled'],
            "price": order.get('average', order['price']),
            "amount_usdt": amount_usdt_filled,
            "symbol": symbol
        }
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return {"error": str(e)}

@tool
async def get_portfolio_balance() -> Dict[str, Any]:
    """Retrieve portfolio balance with all assets converted to USDT."""
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        asset = SYMBOL.split("/")[0]
        asset_balance = balance.get(asset, {}).get('free', 0)
        usdt_value = await convert_to_usdt(SYMBOL, asset_balance, exchange)
        return {
            "usdt_balance": usdt_balance,
            "asset_balance": asset_balance,
            "asset_usdt_value": usdt_value,
            "total_usdt_value": usdt_balance + usdt_value
        }
    except Exception as e:
        logger.error(f"Error fetching balance: {e}")
        return {"error": str(e)}

@tool
async def store_trade(state: Dict[str, Any]) -> Dict[str, Any]:
    """Store trade data in the SQLite database with values in USDT."""
    global trade_buffer
    try:
        indicators = state.get("indicators", {})
        amount_usdt = state.get("amount_usdt", AMOUNTS)
        trade_data = (
            datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S'),
            state.get("signal", "hold"),
            state.get("symbol", SYMBOL),
            indicators.get("close_price"),
            amount_usdt,
            state.get("return_profit_usdt", 0.0),
            state.get("total_return_profit_usdt", 0.0),
            indicators.get("ema1"),
            indicators.get("ema2"),
            indicators.get("rsi"),
            state.get("stop_loss"),
            state.get("take_profit"),
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

@tool
async def backtest_strategy(symbol: str, timeframe: str, strategy_params: Dict[str, Any], limit: int = 500) -> Dict[str, Any]:
    """Backtest a trading strategy with profits in USDT."""
    try:
        data = await fetch_crypto_data(symbol, timeframe, limit)
        if "error" in data:
            return {"error": data["error"]}
        df = pd.DataFrame(data["data"])
        trades = []
        position = None
        buy_price = 0
        total_profit_usdt = 0
        wins = 0
        total_trades = 0

        for i in range(1, len(df)):
            indicators = {
                "close_price": df.iloc[i]["Close"],
                "ema1": df.iloc[i]["ema1"],
                "ema2": df.iloc[i]["ema2"],
                "rsi": df.iloc[i]["rsi"]
            }
            state = {
                "indicators": indicators,
                "position": position,
                "buy_price": buy_price,
                "strategy_id": strategy_params["strategy_id"]
            }
            signal = (await signal_agent.invoke({"input": f"Indicators: {indicators}, Position: {position}, Buy Price: {buy_price}, Strategy: {strategy_params}"})).get("signal", "hold")
            
            if signal == "buy" and not position:
                position = "long"
                buy_price = indicators["close_price"]
                trades.append({"action": "buy", "price": buy_price, "time": df.iloc[i]["timestamp"]})
            elif signal == "sell" and position == "long":
                profit_usdt = indicators["close_price"] - buy_price
                total_profit_usdt += profit_usdt
                total_trades += 1
                if profit_usdt > 0:
                    wins += 1
                position = None
                trades.append({"action": "sell", "price": indicators["close_price"], "profit_usdt": profit_usdt, "time": df.iloc[i]["timestamp"]})

        win_rate = wins / total_trades if total_trades > 0 else 0
        sharpe_ratio = total_profit_usdt / (np.std([t["profit_usdt"] for t in trades if "profit_usdt" in t]) + 1e-10) if total_trades > 0 else 0
        return {
            "strategy_id": strategy_params["strategy_id"],
            "win_rate": win_rate,
            "total_profit_usdt": total_profit_usdt,
            "sharpe_ratio": sharpe_ratio,
            "num_trades": total_trades,
            "trades": trades
        }
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        return {"error": str(e)}

@tool
async def optimize_strategy(state: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize trading strategy with profits in USDT."""
    global current_strategy
    try:
        with db_lock:
            c = conn.cursor()
            c.execute("SELECT action, return_profit_usdt FROM trades WHERE strategy_id = ? ORDER BY time DESC LIMIT 10",
                     (current_strategy["strategy_id"],))
            recent_trades = c.fetchall()
        losses = sum(1 for action, profit in recent_trades if action == "sell" and profit < 0)
        win_rate = sum(1 for action, profit in recent_trades if action == "sell" and profit > 0) / len(recent_trades) if recent_trades else 0

        new_strategy = current_strategy.copy()
        if losses >= 3 or win_rate < 0.5:
            new_strategy["rsi_buy_threshold"] += 2.0
            new_strategy["rsi_sell_threshold"] -= 2.0
            new_strategy["strategy_id"] = f"optimized_{int(time.time())}"

        backtest_result = await backtest_strategy(state["symbol"], TIMEFRAME, new_strategy)
        if "error" not in backtest_result:
            current_strategy = new_strategy
        return {
            "strategy_id": new_strategy["strategy_id"],
            "parameters": new_strategy,
            "win_rate": backtest_result.get("win_rate", 0),
            "total_profit_usdt": backtest_result.get("total_profit_usdt", 0),
            "sharpe_ratio": backtest_result.get("sharpe_ratio", 0)
        }
    except Exception as e:
        logger.error(f"Error in strategy optimization: {e}")
        return {"error": str(e)}

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Agent Prompts
signal_prompt = ChatPromptTemplate.from_messages([
    (SystemMessage(content="""You are a crypto trading signal generator. Use these rules with provided strategy parameters:
    Buy if: (rsi < {rsi_buy_threshold})
    Sell if in position and: (rsi > {rsi_sell_threshold} or close_price <= stop_loss or close_price >= take_profit)
    Else, hold. Return JSON: {{\"signal\": \"buy/sell/hold\", \"reason\": \"...\"}}.""".format(**current_strategy))),
    ("human", "{input}")
])

risk_prompt = ChatPromptTemplate.from_messages([
    (SystemMessage(content="""You are a risk manager. Validate trades:
    - Max 2% portfolio risk.
    - Set stop_loss = buy_price * (1 - {stop_loss_percent}/100).
    - Set take_profit = buy_price * (1 + {take_profit_percent}/100).
    - Prevent consecutive buys or sells without position.
    - Amount = min(total_usdt_value * 0.02 / close_price, total_usdt_value / close_price).
    Return JSON: {{\"approved\": bool, \"stop_loss\": float, \"take_profit\": float, \"amount_usdt\": float, \"reason\": \"...\"}}.""".format(
        stop_loss_percent=current_strategy["stop_loss_percent"],
        take_profit_percent=current_strategy["take_profit_percent"]))),
    ("human", "{input}")
])

profit_prompt = ChatPromptTemplate.from_messages([
    (SystemMessage(content="""Implement profit tracking in USDT:
    - If action == buy and last_sell_profit > 0: set tracking_has_buy=True, tracking_buy_price=current_price.
    - If action == sell and tracking_has_buy: compute return_profit_usdt = (current_price - tracking_buy_price) * amount, update total_return_profit_usdt, set tracking_has_buy=False, tracking_enabled=True if last_sell_profit > 0 else False.
    - If action == sell and not tracking_has_buy: set tracking_enabled based on last_sell_profit.
    - Pause buys/sells if tracking_enabled=False.
    Return JSON: {{\"action\": str, \"return_profit_usdt\": float, \"tracking_enabled\": bool, \"tracking_has_buy\": bool, \"total_return_profit_usdt\": float, \"msg\": \"...\"}}.""")),
    ("human", "{input}")
])

executor_prompt = ChatPromptTemplate.from_messages([
    (SystemMessage(content="""Execute approved trades via CCXT with amounts in USDT. Confirm details. Return JSON: {{\"order_id\": str, \"status\": str, \"filled\": float, \"price\": float, \"amount_usdt\": float}}.""")),
    ("human", "{input}")
])

db_prompt = ChatPromptTemplate.from_messages([
    (SystemMessage(content="""Store trade data and signals in SQLite database with values in USDT. Return JSON: {{\"status\": str, \"message\": str}}.""")),
    ("human", "{input}")
])

backtest_prompt = ChatPromptTemplate.from_messages([
    (SystemMessage(content="""Backtest a trading strategy on historical data with profits in USDT. Return JSON: {{\"strategy_id\": str, \"win_rate\": float, \"total_profit_usdt\": float, \"sharpe_ratio\": float, \"num_trades\": int, \"trades\": list}}.""")),
    ("human", "{input}")
])

optimize_prompt = ChatPromptTemplate.from_messages([
    (SystemMessage(content="""Optimize trading strategy based on recent performance and market conditions with profits in USDT.
    - Analyze last 10 trades for win rate and losses.
    - If 3+ consecutive losses or win rate < 0.5, adjust parameters (e.g., RSI thresholds, stop-loss).
    - Backtest new strategy and store in database.
    Return JSON: {{\"strategy_id\": str, \"parameters\": dict, \"win_rate\": float, \"total_profit_usdt\": float, \"sharpe_ratio\": float}}.""")),
    ("human", "{input}")
])

# Create agents
try:
    signal_agent = create_react_agent(
        llm,
        tools=[fetch_crypto_data],
        state_modifier=SystemMessage(content=signal_prompt.messages[0].content),
        checkpointer=MemorySaver()
    )
    risk_agent = create_react_agent(
        llm,
        tools=[get_portfolio_balance],
        state_modifier=SystemMessage(content=risk_prompt.messages[0].content),
        checkpointer=MemorySaver()
    )
    executor_agent = create_react_agent(
        llm,
        tools=[execute_trade],
        state_modifier=SystemMessage(content=executor_prompt.messages[0].content),
        checkpointer=MemorySaver()
    )
    db_agent = create_react_agent(
        llm,
        tools=[store_trade],
        state_modifier=SystemMessage(content=db_prompt.messages[0].content),
        checkpointer=MemorySaver()
    )
    backtest_agent = create_react_agent(
        llm,
        tools=[backtest_strategy],
        state_modifier=SystemMessage(content=backtest_prompt.messages[0].content),
        checkpointer=MemorySaver()
    )
    optimize_agent = create_react_agent(
        llm,
        tools=[optimize_strategy],
        state_modifier=SystemMessage(content=optimize_prompt.messages[0].content),
        checkpointer=MemorySaver()
    )
    profit_agent = create_react_agent(
        llm,
        tools=[],
        state_modifier=SystemMessage(content=profit_prompt.messages[0].content),
        checkpointer=MemorySaver()
    )
    data_agent = create_react_agent(
        llm,
        tools=[fetch_crypto_data],
        checkpointer=MemorySaver()
    )
except Exception as e:
    logger.critical(f"Agent creation error: {e}")
    exit(1)

# Agent Nodes
def call_data_agent(state: TradingState) -> TradingState:
    result = data_agent.invoke({"input": f"Fetch data for {state['symbol']}"})
    return {
        "messages": result["messages"],
        "indicators": result.get("indicators", {}),
        "portfolio": get_portfolio_balance(),
        "strategy_id": current_strategy["strategy_id"],
        "next": "signal_node"
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
        "reason": signal_data["reason"],
        "next": "risk" if signal_data["signal"] != "hold" else "backtest"
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
        "amount_usdt": risk_data.get("amount_usdt"),
        "reason": risk_data["reason"],
        "next": "profit" if risk_data["approved"] else "db"
    }

def call_profit_agent(state: TradingState) -> TradingState:
    current_price = state["indicators"].get("close_price", 0)
    amount = state.get("amount_usdt", AMOUNTS) / current_price if current_price > 0 else 1
    primary_profit_usdt = ((current_price - state.get("buy_price", 0)) * amount) if state["signal"] == "sell" and state.get("tracking_has_buy") else 0
    input_data = f"Action: {state['signal']}, Current Price: {current_price}, Amount: {amount}, Primary Profit USDT: {primary_profit_usdt}, Tracking: {{enabled: {state.get('tracking_enabled', False)}, has_buy: {state.get('tracking_has_buy', False)}, buy_price: {state.get('tracking_buy_price', 0)}}}"
    result = profit_agent.invoke({"input": input_data})
    try:
        profit_data = eval(result["messages"][-1].content)
    except:
        profit_data = {"action": "hold", "return_profit_usdt": 0, "tracking_enabled": state.get("tracking_enabled", True), "tracking_has_buy": state.get("tracking_has_buy", False), "total_return_profit_usdt": state.get("total_return_profit_usdt", 0), "msg": "Error parsing profit"}
    return {
        "messages": result["messages"],
        "signal": profit_data["action"],
        "return_profit_usdt": profit_data["return_profit_usdt"],
        "tracking_enabled": profit_data["tracking_enabled"],
        "tracking_has_buy": profit_data["tracking_has_buy"],
        "total_return_profit_usdt": profit_data["total_return_profit_usdt"],
        "reason": profit_data["msg"],
        "next": "execute" if profit_data["action"] != "hold" and profit_data["tracking_enabled"] else "db"
    }

def call_executor_agent(state: TradingState) -> TradingState:
    if not state["risk_approved"] or state["signal"] == "hold" or not state.get("tracking_enabled", True):
        return {
            "messages": state["messages"],
            "next": "db"
        }
    input_data = f"Execute {state['signal']} on {state['symbol']} for {state['amount_usdt']} USDT at {state['indicators']['close_price']}"
    result = executor_agent.invoke({"input": input_data})
    try:
        order_data = eval(result["messages"][-1].content)
    except:
        order_data = {"order_id": None, "status": "failed", "filled": 0, "price": state["indicators"]["close_price"], "amount_usdt": 0}
    return {
        "messages": result["messages"],
        "order_id": order_data.get("order_id"),
        "position": "long" if state["signal"] == "buy" else None,
        "buy_price": state["indicators"]["close_price"] if state["signal"] == "buy" else None,
        "amount_usdt": order_data.get("amount_usdt", 0),
        "next": "db"
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
        "db_message": db_data["message"],
        "next": "backtest"
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
        "backtest_results": backtest_data,
        "next": "optimize"
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
        "backtest_results": optimize_data,
        "next": END
    }

# Build Graph
workflow = StateGraph(TradingState)
workflow.add_node("data", call_data_agent)
workflow.add_node("signal_node", call_signal_agent)
workflow.add_node("risk", call_risk_agent)
workflow.add_node("profit", call_profit_agent)
workflow.add_node("execute", call_executor_agent)
workflow.add_node("db", call_db_agent)
workflow.add_node("backtest", call_backtest_agent)
workflow.add_node("optimize", call_optimize_agent)

# Define edges to ensure all nodes are reachable
workflow.add_edge("data", "signal_node")
workflow.add_edge("signal_node", "risk")
workflow.add_conditional_edges(
    "risk",
    lambda state: state["next"],
    {
        "profit": "profit",
        "db": "db"
    }
)
workflow.add_conditional_edges(
    "profit",
    lambda state: state["next"],
    {
        "execute": "execute",
        "db": "db"
    }
)
workflow.add_edge("execute", "db")
workflow.add_edge("db", "backtest")
workflow.add_edge("backtest", "optimize")
workflow.add_edge("optimize", END)

# Set the entry point
workflow.set_entry_point("data")

# Compile with memory
memory = MemorySaver()
graph_app = workflow.compile(checkpointer=memory)

# FastAPI Endpoints
@app.on_event("startup")
async def startup_event():
    if not setup_database(first_attempt=True):
        logger.critical("Failed to initialize database.")
        raise Exception("Database initialization failed")
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
    result = await execute_trade(request.action, request.symbol, request.amount_usdt)
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
        "total_return_profit_usdt": 0.0,
        "tracking_enabled": True,
        "tracking_has_buy": False,
        "tracking_buy_price": None,
        "strategy_id": current_strategy["strategy_id"],
        "portfolio": {},
        "indicators": {},
        "signal": "",
        "risk_approved": False,
        "backtest_results": {},
        "next": None
    }
    try:
        result = graph_app.invoke(initial_state, config)
        return {"status": "success", "state": result}
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
