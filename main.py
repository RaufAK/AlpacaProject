import math
import os
import re
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from urllib.parse import urlparse

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    AssetClass,
    AssetStatus,
    OrderSide,
    OrderType,
    TimeInForce,
)
from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest
from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, render_template, request  # type: ignore[import-not-found]

app = Flask(__name__)

SESSION_COOKIE = "alpaca_session"
SIGNED_OUT_COOKIE = "alpaca_signed_out"
_session_lock = threading.RLock()
_account_sessions: dict[str, tuple[str, str]] = {}
_signed_out_sessions: set[str] = set()

CHART_PERIODS = {
    "1d": (timedelta(days=1), TimeFrame.Minute),
    "1m": (timedelta(days=30), TimeFrame.Day),
    "1y": (timedelta(days=365), TimeFrame.Day),
}


class LoginRequiredError(Exception):
    pass


def get_browser_session_id() -> str | None:
    session_id = request.cookies.get(SESSION_COOKIE, "")
    if re.fullmatch(r"[A-Za-z0-9_-]{20,128}", session_id):
        return session_id
    return None


def get_account_credentials() -> tuple[str, str] | None:
    load_dotenv()
    session_id = get_browser_session_id()

    if request.cookies.get(SIGNED_OUT_COOKIE) == "1":
        return None
    if session_id:
        with _session_lock:
            if session_id in _signed_out_sessions:
                return None
            credentials = _account_sessions.get(session_id)
        if credentials:
            return credentials

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if api_key and secret_key:
        return api_key, secret_key
    return None


def get_alpaca_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    credentials = get_account_credentials()
    if not credentials:
        raise LoginRequiredError("Connect an Alpaca paper account to continue.")

    api_key, secret_key = credentials
    return (
        TradingClient(api_key, secret_key, paper=True),
        StockHistoricalDataClient(api_key, secret_key),
    )


@app.get("/")
def index():
    response = make_response(render_template("index.html"))
    if not get_browser_session_id():
        response.set_cookie(
            SESSION_COOKIE,
            secrets.token_urlsafe(32),
            httponly=True,
            samesite="Strict",
            path="/",
        )
    return response


@app.before_request
def validate_mutation_origin():
    if request.method != "POST" or request.path not in {
        "/api/auth/login",
        "/api/auth/logout",
        "/api/orders",
    }:
        return None

    origin = request.headers.get("Origin")
    if origin and urlparse(origin).netloc != request.host:
        return jsonify({"error": "Requests must come from this dashboard."}), 403
    return None


@app.get("/api/auth/status")
def auth_status():
    return jsonify({"authenticated": get_account_credentials() is not None})


@app.post("/api/auth/login")
def login():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Enter your Alpaca paper API key and secret."}), 400

    api_key = data.get("api_key")
    secret_key = data.get("secret_key")
    if not isinstance(api_key, str) or not isinstance(secret_key, str):
        return jsonify({"error": "Enter both your Alpaca API key and secret."}), 400

    api_key = api_key.strip()
    secret_key = secret_key.strip()
    if not api_key or not secret_key:
        return jsonify({"error": "Enter both your Alpaca API key and secret."}), 400

    try:
        account = TradingClient(api_key, secret_key, paper=True).get_account()
    except APIError as error:
        status = error.status_code
        client_error = status is not None and 400 <= status < 500
        return (
            jsonify(
                {
                    "error": (
                        f"Alpaca rejected the credentials (HTTP {status}): "
                        f"{error.message}. Use a paper-trading API key and secret."
                    )
                }
            ),
            401 if client_error else 502,
        )

    session_id = secrets.token_urlsafe(32)
    with _session_lock:
        previous_session_id = get_browser_session_id()
        if previous_session_id:
            _account_sessions.pop(previous_session_id, None)
            _signed_out_sessions.discard(previous_session_id)
        _account_sessions[session_id] = (api_key, secret_key)

    response = make_response(
        jsonify(
            {
                "authenticated": True,
                "account_status": str(getattr(account.status, "value", account.status)),
            }
        )
    )
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        httponly=True,
        samesite="Strict",
        path="/",
    )
    response.delete_cookie(SIGNED_OUT_COOKIE, path="/", samesite="Strict")
    return response


@app.post("/api/auth/logout")
def logout():
    session_id = get_browser_session_id()
    if session_id:
        with _session_lock:
            _account_sessions.pop(session_id, None)
            _signed_out_sessions.add(session_id)

    response = make_response(jsonify({"authenticated": False}))
    response.set_cookie(
        SIGNED_OUT_COOKIE,
        "1",
        httponly=True,
        samesite="Strict",
        path="/",
    )
    return response


def is_tradable_equity(asset) -> bool:
    asset_class = asset.get("class") if isinstance(asset, dict) else asset.asset_class
    status = asset.get("status") if isinstance(asset, dict) else asset.status
    tradable = asset.get("tradable") if isinstance(asset, dict) else asset.tradable
    return (
        getattr(asset_class, "value", asset_class) == AssetClass.US_EQUITY.value
        and getattr(status, "value", status) == AssetStatus.ACTIVE.value
        and tradable
    )


def get_asset_field(asset, field: str):
    return asset.get(field) if isinstance(asset, dict) else getattr(asset, field)


@lru_cache(maxsize=1)
def get_asset_catalog(cache_window: int):
    trading_client, _ = get_alpaca_clients()
    asset_filter = GetAssetsRequest(
        status=AssetStatus.ACTIVE,
        asset_class=AssetClass.US_EQUITY,
    )
    # Read raw asset JSON so newer exchange codes don't fail SDK model validation.
    return trading_client.get("/assets", asset_filter.to_request_fields())


@app.get("/api/assets/search")
def search_assets():
    query = request.args.get("q", "").strip()
    if len(query) < 1:
        return jsonify({"assets": []})

    try:
        assets = get_asset_catalog(int(time.time() // 300))
        normalized_query = query.casefold()
        matches = [
            asset
            for asset in assets
            if is_tradable_equity(asset)
            and (
                normalized_query in get_asset_field(asset, "symbol").casefold()
                or normalized_query in (get_asset_field(asset, "name") or "").casefold()
            )
        ]
        matches.sort(
            key=lambda asset: (
                not get_asset_field(asset, "symbol")
                .casefold()
                .startswith(normalized_query),
                get_asset_field(asset, "symbol"),
            )
        )
        return jsonify(
            {
                "assets": [
                    {
                        "symbol": get_asset_field(asset, "symbol"),
                        "name": get_asset_field(asset, "name")
                        or get_asset_field(asset, "symbol"),
                    }
                    for asset in matches[:15]
                ]
            }
        )
    except LoginRequiredError as error:
        return jsonify({"error": str(error)}), 401
    except ValueError as error:
        return jsonify({"error": str(error)}), 503
    except APIError as error:
        app.logger.exception("Alpaca asset search failed")
        return (
            jsonify(
                {
                    "error": f"Alpaca API error (HTTP {error.status_code}): {error.message}"
                }
            ),
            502,
        )


@app.post("/api/orders")
def submit_order():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Provide an order as JSON."}), 400

    symbol = str(data.get("symbol", "")).strip().upper()
    side_value = str(data.get("side", "")).strip().lower()
    if not re.fullmatch(r"[A-Z0-9.-]{1,15}", symbol):
        return jsonify({"error": "Enter a valid stock or ETF symbol."}), 400
    if side_value not in {"buy", "sell"}:
        return jsonify({"error": "Order side must be buy or sell."}), 400

    try:
        quantity = Decimal(str(data.get("qty", "")))
    except (InvalidOperation, ValueError):
        return jsonify({"error": "Enter a valid share quantity."}), 400
    if not quantity.is_finite() or quantity <= 0:
        return jsonify({"error": "Share quantity must be greater than zero."}), 400
    quantity_as_float = float(quantity)
    if not math.isfinite(quantity_as_float):
        return jsonify({"error": "Share quantity is too large."}), 400

    try:
        trading_client, _ = get_alpaca_clients()
    except LoginRequiredError as error:
        return jsonify({"error": str(error)}), 401
    except ValueError as error:
        return jsonify({"error": str(error)}), 503

    try:
        asset = trading_client.get(f"/assets/{symbol}")
        if not is_tradable_equity(asset):
            return (
                jsonify(
                    {"error": f"{symbol} is not an active, tradable stock or ETF."}
                ),
                400,
            )

        increment_value = get_asset_field(asset, "min_trade_increment")
        if (
            not get_asset_field(asset, "fractionable")
            and quantity != quantity.to_integral_value()
        ):
            return (
                jsonify({"error": f"{symbol} does not support fractional shares."}),
                400,
            )
        if increment_value:
            increment = Decimal(str(increment_value))
            if quantity % increment != 0:
                return (
                    jsonify(
                        {
                            "error": f"{symbol} share quantity must be in increments of {increment}."
                        }
                    ),
                    400,
                )

        side = OrderSide(side_value)
        if side == OrderSide.SELL:
            positions = trading_client.get_all_positions()
            position = next((item for item in positions if item.symbol == symbol), None)
            available_quantity = (
                Decimal(str(position.qty_available))
                if position and position.qty_available is not None
                else Decimal(str(position.qty))
                if position
                else Decimal("0")
            )
            if available_quantity <= 0:
                return (
                    jsonify(
                        {
                            "error": f"You do not own any available {symbol} shares to sell."
                        }
                    ),
                    400,
                )
            if quantity > available_quantity:
                return (
                    jsonify(
                        {
                            "error": (
                                f"You can sell at most {available_quantity} available "
                                f"{symbol} shares."
                            )
                        }
                    ),
                    400,
                )

        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=quantity_as_float,
                side=side,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
            )
        )
        return (
            jsonify(
                {
                    "id": str(order.id),
                    "symbol": order.symbol or symbol,
                    "side": str(order.side),
                    "qty": str(order.qty or quantity),
                    "status": str(order.status),
                }
            ),
            201,
        )
    except APIError as error:
        app.logger.exception("Alpaca order request failed")
        status = error.status_code
        client_error = status is not None and 400 <= status < 500
        return (
            jsonify({"error": f"Alpaca API error (HTTP {status}): {error.message}"}),
            400 if client_error else 502,
        )


@app.get("/api/dashboard")
def dashboard_data():
    period = request.args.get("period", "1d")
    if period not in CHART_PERIODS:
        return jsonify({"error": "Choose a chart period of 1d, 1m, or 1y."}), 400

    try:
        trading_client, market_data_client = get_alpaca_clients()
    except LoginRequiredError as error:
        return jsonify({"error": str(error)}), 401
    except ValueError as error:
        return jsonify({"error": str(error)}), 503

    try:
        account = trading_client.get_account()
        positions = trading_client.get_all_positions()
        requested_symbol = request.args.get("symbol")
        selected_position = next(
            (position for position in positions if position.symbol == requested_symbol),
            None,
        )
        selected_asset = None
        if requested_symbol and selected_position is None:
            selected_asset = trading_client.get(f"/assets/{requested_symbol}")
            if not is_tradable_equity(selected_asset):
                return (
                    jsonify(
                        {"error": "That asset is not an active, tradable stock or ETF."}
                    ),
                    404,
                )
        if selected_position is None and positions:
            if not requested_symbol:
                selected_position = positions[0]

        payload = {
            "account": {
                "status": str(account.status),
                "cash": str(account.cash),
                "buying_power": str(account.buying_power),
            },
            "positions": [
                {
                    "symbol": position.symbol,
                    "qty": float(position.qty),
                    "current_price": float(position.current_price),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                }
                for position in positions
            ],
            "selected": None,
            "period": period,
            "bars": [],
        }

        if selected_position is not None or selected_asset is not None:
            symbol = (
                selected_position.symbol
                if selected_position is not None
                else get_asset_field(selected_asset, "symbol")
            )
            current_price = (
                float(selected_position.current_price)
                if selected_position is not None
                else None
            )
            latest_trades = market_data_client.get_stock_latest_trade(
                StockLatestTradeRequest(
                    symbol_or_symbols=symbol,
                    feed=DataFeed.IEX,
                )
            )
            latest_trade = latest_trades.get(symbol)
            if latest_trade is not None:
                current_price = float(latest_trade.price)
            payload["selected"] = {
                "symbol": symbol,
                "qty": float(selected_position.qty) if selected_position else 0,
                "current_price": current_price,
                "unrealized_pl": float(selected_position.unrealized_pl)
                if selected_position
                else 0,
                "unrealized_plpc": float(selected_position.unrealized_plpc)
                if selected_position
                else 0,
            }

            duration, timeframe = CHART_PERIODS[period]
            end_time = datetime.now(timezone.utc)
            historical_bars = market_data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=timeframe,
                    start=end_time - duration,
                    end=end_time,
                    feed=DataFeed.IEX,
                )
            )
            payload["bars"] = [
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "close": float(bar.close),
                }
                for bar in historical_bars[symbol]
            ]
            if current_price is None and payload["bars"]:
                current_price = payload["bars"][-1]["close"]
                payload["selected"]["current_price"] = current_price

        return jsonify(payload)
    except APIError as error:
        app.logger.exception("Alpaca API request failed")
        status = error.status_code
        if status == 404:
            return jsonify({"error": "That stock or ETF symbol was not found."}), 404
        return (
            jsonify(
                {
                    "error": (
                        f"Alpaca API error (HTTP {error.status_code}): "
                        f"{error.message}"
                    )
                }
            ),
            502,
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
