import os

from alpaca.data.live import StockDataStream
from dotenv import load_dotenv


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Set it in your .env file or environment."
        )
    return value


load_dotenv()

API_KEY = require_env("ALPACA_API_KEY")
API_SECRET = require_env("ALPACA_API_SECRET")


async def handle_trade(trade) -> None:
    print(trade)


def main() -> None:
    stream = StockDataStream(API_KEY, API_SECRET)
    stream.subscribe_trades(handle_trade, "AAPL")
    stream.run()


if __name__ == "__main__":
    main()
