# PricePredictor

Lightweight trading research project for crypto backtesting with Alpaca market data and VectorBT strategy simulation.

It currently includes:
- Historical data loading from Alpaca crypto bars
- SMA crossover backtesting
- Hyperparameter search across SMA windows
- Optional execution fidelity toggle for next-bar execution
- Optional friction model (commission, fixed fee, slippage, and volume caps)
- Kelly-based position sizing comparison in backtest runs
- A simple live stream example for trade prints

## Project Layout

```text
.
|- alpaca_stream.py
|- run_backtest.py
|- run_hyperparameter_search.py
|- requirements.txt
`- src/
	 |- config.py
	 |- data.py
	 |- models/
	 |	|- broker.py
	 |	`- risk.py
	 `- strategies/
			`- sma_crossover.py
```

## Requirements

- Python 3.10+
- Alpaca API credentials

## Installation

1. Clone the repo.
2. Create and activate a virtual environment.
3. Install dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4. Enable repository git hooks.

```powershell
git config core.hooksPath .githooks
```

The repository includes a `pre-push` hook that runs the full test suite before push.

## Development Validation

After making code changes, always run lint and formatting checks before finishing work:

```powershell
python -m ruff check .
python -m ruff format --check .
```

## Environment Variables

Create a `.env` file in the repository root:

```env
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
```

Notes:
- Backtests use Alpaca historical crypto data.
- The live stream example uses Alpaca live stock streaming.

## Quick Start

### 1) Run a Backtest

```powershell
python run_backtest.py
```

What it does:
- Loads BTC/USD bars from Alpaca
- Runs SMA crossover strategy with `fast=10`, `slow=30`
- Prints baseline and Kelly-sized portfolio stats
- Optionally opens an interactive chart with price, moving averages, and positions if enabled in `src/config.py` (`BACKTEST_RENDER_CHART` is `False` by default)

### 2) Run Hyperparameter Search

```powershell
python run_hyperparameter_search.py
```

What it does:
- Runs the same strategy across configured fast/slow SMA windows
- Optimizes for objective in `src/config.py` (`sharpe_ratio` or `total_return`)
- Prints best parameters and top combinations

### 3) Run Live Stream Demo

```powershell
python alpaca_stream.py
```

What it does:
- Connects to Alpaca live stream
- Subscribes to AAPL trade updates
- Prints trades as they arrive

## Configuration

Default settings live in `src/config.py`:
- Date range (`DEFAULT_START`, `DEFAULT_END`)
- Timeframe (`DEFAULT_TIMEFRAME`)
- Initial cash (`DEFAULT_INIT_CASH`)
- Hyperparameter ranges (`FAST_WINDOWS`, `SLOW_WINDOWS`)
- Scan objective (`SCAN_OBJECTIVE`)
- Execution timing (`ENABLE_NEXT_BAR_EXECUTION`)
- Friction model toggle (`ENABLE_FRICTION_MODEL`)
- Broker friction params (`BROKER_COMMISSION_PCT`, `BROKER_FIXED_FEE`, `BROKER_SLIPPAGE_PCT`)
- Volume cap (`MAX_VOLUME_PARTICIPATION`)
- Kelly scaling (`KELLY_FACTOR`)

Adjust these values to tune your experiments.

## Execution and Risk Notes

- Next-bar execution and friction are configured independently.
- Broker friction and max-volume participation are centralized in `src/models/broker.py`.
- Backtest script computes a conservative Kelly estimate from baseline signals and runs a comparison portfolio.

## Strategy Notes

`src/strategies/sma_crossover.py` implements:
- Entry: fast MA crosses above slow MA
- Exit: fast MA crosses below slow MA

Backtests are built with `vectorbt.Portfolio.from_signals`.

## Troubleshooting

- Missing credentials error:
	- Ensure `.env` exists and contains valid `ALPACA_API_KEY` and `ALPACA_API_SECRET`.
- No data returned for symbol:
	- Confirm symbol format (for example `BTC/USD`) and date range.
- Plot window does not appear:
	- Run in an environment that supports interactive plotting, or use notebook/GUI backend.

## Disclaimer

This project is for research and educational use only. It is not financial advice.
