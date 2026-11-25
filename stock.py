import os
from datetime import datetime, timedelta, timezone

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("Missing Alpaca API credentials.")

    client = StockHistoricalDataClient(api_key, secret_key)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=7)

    request = StockBarsRequest(
        symbol_or_symbols=["QQQ"],
        timeframe=TimeFrame.Day,
        start=start_time,
        end=end_time,
    )

    bars = client.get_stock_bars(request)

    for bar in bars["QQQ"]:
        print(
            f"{bar.timestamp.date()} | "
            f"Open: ${bar.open:.2f} | "
            f"Close: ${bar.close:.2f} | "
            f"Volume: {bar.volume:,}"
        )


if __name__ == "__main__":
    main()
