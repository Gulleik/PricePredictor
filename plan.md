# PLAN.md - High-Fidelity Research Platform Roadmap

## Objective
Transform the current repository into a **Strategic Sandbox**. The goal is to ensure that any strategy passing these tests is mathematically and operationally ready for IRL (In Real Life) trading by accounting for friction, avoiding overfitting, and verifying robustness.

---

## Milestone 1: Execution Fidelity (The "Friction" Layer)
**Goal:** Eliminate "perfect world" hallucinations by simulating real market costs and timing.

- [ ] **Implement a Broker Model:** Create `src/models/broker.py` to encapsulate execution logic.
- [ ] **Update `src/config.py`:** Add mandatory friction parameters:
    - `SLIPPAGE_BPS`: Default 2.0 (0.02%).
    - `COMMISSION_PER_SHARE`: Standard broker fee logic.
    - `MIN_SPREAD`: Minimum dollar distance between Bid and Ask.
- [ ] **Enforce Next-Bar Fills:** Modify the backtest loop so a signal at Candle $T$ (Close) is filled at Candle $T+1$ (Open).
- [ ] **Volume Constraint:** Add a `MAX_VOLUME_PARTICIPATION` limit (e.g., 10%) to ensure trades don't exceed realistic liquidity.

---

## Milestone 2: Validation Framework (The "Scientific" Layer)
**Goal:** Prevent curve-fitting (memorizing the past) through rigorous data separation.

- [ ] **Walk-Forward Optimization (WFO):** Update `run_hyperparameter_search.py` to support rolling windows.
    - **In-Sample (IS):** Training period to find optimal parameters.
    - **Out-of-Sample (OOS):** "Blind" testing period to verify the parameters hold up.
- [ ] **Regime Analysis:** Tag data by market type (Bull, Bear, Sideways) to analyze performance in different environments.

---

## Milestone 3: Data Integrity (The "Anti-Bias" Layer)
**Goal:** Ensure the strategy isn't "cheating" by seeing the future or ignoring failures.

- [ ] **Look-Ahead Bias Guard:** Create a `pytest` suite that confirms the strategy logic never accesses `df.iloc[t+1]` or beyond during a simulation.
- [ ] **Survivorship Bias Audit:** Ensure `src/data.py` handles (or at least logs) if tickers in the universe were delisted during the test period.
- [ ] **Timezone Standardization:** Enforce UTC across all data loading to prevent "time-travel" bugs.

---

## Milestone 4: Systematic Search & Metrics
**Goal:** Use Bayesian search to find the "sweet spot" and measure risk-adjusted returns.

- [ ] **Optuna Integration:** Implement `optuna` in `run_hyperparameter_search.py` for smarter parameter discovery.
- [ ] **Advanced Metrics Suite:** Expand backtest output to include:
    - **Sharpe/Sortino Ratio:** Risk-adjusted returns.
    - **Max Drawdown Duration:** Time spent "underwater."
    - **Profit Factor:** Gross Profit vs. Gross Loss.

---

## Milestone 5: Stress Testing (The "Resilience" Layer)
**Goal:** Verify if a strategy is "Good" or just "Lucky."

- [ ] **Monte Carlo Simulation:** Create a script to run 1,000 iterations of the backtest while:
    - Randomly shuffling the sequence of daily returns.
    - Randomly "dropping" 10% of winning trades.
- [ ] **Robustness Score:** Calculate the probability of ruin based on these variations.

---

## Milestone 6: Strategy Library Expansion
**Goal:** Implement diverse algorithmic archetypes to find a robust "Edge" across different market conditions.

- [ ] **Mean Reversion (RSI/Bollinger):** Exploit "overextended" prices returning to the average. Best for ranging markets.
- [ ] **Trend Following (MACD/EMA Cross):** Ride momentum in trending markets. Focus on "cutting losers short and letting winners run."
- [ ] **Volatility Breakout (Donchian/ATR):** Enter trades when price breaks out of a defined volatility range, signaling the start of a new trend.
- [ ] **Opening Range Breakout (ORB):** Capture the high-volume volatility seen in the first 30–60 minutes of the market session.

---

## Strategy Recommendations for Discovery

1. **Mean Reversion + Volatility Filter:** Use RSI but only enter when **Bollinger Band Width** is high (high volatility) or low (squeeze), adding a statistical "reason" for the reversal.
2. **EMA Trend + Chandelier Exit:** Use a fast/slow EMA cross for entry, but use an **ATR-based trailing stop** (Chandelier Exit) to protect capital during the trend.
3. **Statistical Arbitrage (Pairs):** Trade the spread between two highly correlated assets (e.g., SPY vs IVV). This requires updating `src/data.py` to load multiple tickers simultaneously.

---

## Agent Operational Rules

1. **Config-First:** Every new variable (slippage, search ranges, strategy thresholds) MUST be added to `src/config.py`. Do not use CLI flags.
2. **Stateless Logic:** Strategies in `src/strategies/` must remain stateless; they receive a window of data and return a signal.
3. **Traceability:** Every execution of `run_backtest.py` should log its results to a `/results/` directory with a timestamp and the configuration used.