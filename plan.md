# PLAN.md - High-Fidelity Research Platform Roadmap

## Objective
Transform the repository into a **Strategic Sandbox**. The goal is to ensure that any strategy passing these tests is mathematically, operationally, and technically ready for IRL trading by accounting for friction, avoiding overfitting, and enforcing rigorous risk management.

---

## Milestone 1: Infrastructure & Automation (The "Safety & Speed" Rail)
**Goal:** Automate quality control and optimize data handling for rapid iteration.

- [ ] **Linting & Formatting:** Set up `.github/workflows/ci.yml` using `ruff` for linting and formatting.
- [ ] **Automated Testing:** Configure GitHub Actions to run `pytest` on every Push and PR.
- [ ] **Security Scanning:** Integrate Secret Scanning and SAST (CodeQL) to protect API keys.
- [ ] **Data Persistence & Caching:** Implement a local **Parquet-based cache** in a `data/` directory to prevent redundant Alpaca API calls and significantly speed up backtests.

## Milestone 2: Data Integrity (The "Anti-Bias" Layer)
**Goal:** Ensure the "ground truth" data is clean and the strategy isn't "cheating."

- [ ] **Look-Ahead Bias Guard:** Implement a `pytest` suite ensuring strategy logic never accesses future indices (e.g., `df.iloc[t+1]`).
- [ ] **Timezone Standardization:** Enforce UTC across all modules to prevent "time-travel" bugs.
- [ ] **Survivorship Bias Audit:** Add logic to handle or flag delisted tickers to prevent upward-biased results.

## Milestone 3: Execution Fidelity & Risk (The "Physics" Layer)
**Goal:** Simulate market costs and implement survival-focused position sizing.

- [ ] **Broker Model:** Create `src/models/broker.py` for slippage, commissions, and spread logic.
- [ ] **Enforce Next-Bar Fills:** Ensure signals at Candle $T$ are executed at Candle $T+1$ Open.
- [ ] **Risk & Position Sizing:** Create `src/models/risk.py` to implement **Volatility Targeting** or **Kelly Criterion** sizing instead of fixed-lot trading.
- [ ] **Volume Constraints:** Implement `MAX_VOLUME_PARTICIPATION` limits for liquidity realism.

## Milestone 4: Validation & Sensitivity (The "Scientific" Layer)
**Goal:** Distinguish "luck" from "edge" and identify brittle strategies.

- [ ] **Walk-Forward Optimization (WFO):** Update `run_hyperparameter_search.py` for rolling In-Sample (IS) and Out-of-Sample (OOS) windows.
- [ ] **Parameter Sensitivity Testing:** Generate "Sensitivity Heatmaps" to ensure the strategy remains profitable across a range of parameters (avoiding "brittle" peaks).
- [ ] **Regime Tagging:** Classify market environments (Bull/Bear/Sideways) for granular performance analysis.

## Milestone 5: Systematic Search & Metrics (The "Laboratory")
**Goal:** Use high-performance tools to find and measure risk-adjusted success.

- [ ] **Optuna Integration:** Implement Bayesian search in `run_hyperparameter_search.py`.
- [ ] **Advanced Metrics Suite:** Include Sharpe, Sortino, **Calmar Ratio** (CAGR/Max Drawdown), and Max Drawdown Duration.

## Milestone 6: Strategy Library Expansion (The "Experiments")
**Goal:** Build out diverse trading archetypes.

- [ ] **Mean Reversion:** RSI/Bollinger Band logic with volatility filters.
- [ ] **Trend Following:** EMA Cross strategies with ATR-based trailing stops.
- [ ] **Volatility Breakout:** Donchian Channel or ATR-based breakout logic.
- [ ] **ORB:** Opening Range Breakout logic for early-session volatility.

## Milestone 7: Full-Matrix Hyperparameter Automation (The "Batch Research Engine")
**Goal:** Replace manual strategy-by-strategy optimization with a single orchestrated run over every configured strategy, symbol, and timeframe.

- [ ] **Config-Driven Universe:** Add explicit lists in `src/config/common.py` for search symbols, search timeframes, and enabled strategies (for example `HYPERPARAM_SYMBOLS`, `HYPERPARAM_TIMEFRAMES`, `HYPERPARAM_STRATEGIES`).
- [ ] **Batch Orchestrator:** Extend `run_hyperparameter_search.py` to loop over the full matrix `(strategy, timeframe, symbol)` and execute Optuna per combination.
- [ ] **Failure Isolation:** Ensure one failing combination is logged and skipped without stopping the full batch run.
- [ ] **Structured Outputs:** Save one per-run artifact plus a single aggregated leaderboard/table in `results/`, including strategy, symbol, timeframe, objective score, and key params.
- [ ] **Progress Observability:** Print concise progress updates (current combination, completed/total, ETA) to keep long runs transparent.
- [ ] **Reproducibility Guardrails:** Enforce fixed seeds and persist the effective config snapshot with each run.
- [ ] **Execution Modes:** Support `quick` and `full` matrix modes via config-only switches (no required CLI args).
- [ ] **Validation Tests:** Add integration/scenario tests to verify matrix coverage, output aggregation, and failure-isolation behavior.

## Milestone 8: Stress Testing (The "Trial by Fire")
**Goal:** Final validation before considering live deployment.

- [ ] **Monte Carlo Simulation:** Run 1,000 iterations with shuffled returns and random trade "drops."
- [ ] **Black Swan Simulation:** Stress test the strategy against extreme historical events (e.g., 2008 crash, 2020 COVID flash, overnight gaps).
- [ ] **Strategy Invariant Tests:** Verify logic on artificial "perfect" data (e.g., perfect sine waves) to confirm mathematical correctness.

---

## Agent Operational Rules

1. **Config-First:** Every new variable (friction, search ranges, risk parameters) MUST be added to `src/config.py`. No CLI flags.
2. **Stateless Logic:** Strategies in `src/strategies/` must remain stateless; they receive data and return a signal.
3. **Traceability:** Every backtest run must log its results to a `/results/` directory with a timestamp and the configuration used.
4. **CI-Ready Code:** All new code must pass `ruff` and `pytest` locally before being considered "Done."