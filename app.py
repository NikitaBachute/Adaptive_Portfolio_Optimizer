# ===================================================================
# 1. IMPORTS
# ===================================================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, expected_returns, risk_models, plotting
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# ===================================================================
# PAGE CONFIGURATION
# ===================================================================
st.set_page_config(
    page_title="Adaptive Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

# ===================================================================
# CACHED BACKTESTING FUNCTION (Your existing, successful backtest)
# ===================================================================
@st.cache_data
def run_backtest():
    # This function contains the full backtest logic you've already created.
    # It's unchanged from the previous version.
    tickers = ["SPY", "AGG", "GLD", "VNQ", "MSFT", "JPM"]
    benchmark_ticker = "SPY"
    rebalance_frequency = 3
    rolling_window = 36

    all_tickers = list(set(tickers + [benchmark_ticker]))
    all_data = yf.download(all_tickers, start="2012-01-01", end="2024-12-31")['Close']
    monthly_returns = all_data.resample('M').last().pct_change().dropna()
    asset_returns = monthly_returns[tickers]
    benchmark_returns = monthly_returns[benchmark_ticker]
    spy_ma = monthly_returns['SPY'].rolling(window=12).mean()

    portfolio_returns = []
    dates = []
    all_weights = []

    for i in range(rolling_window, len(monthly_returns) - rebalance_frequency, rebalance_frequency):
        current_window_returns = monthly_returns.iloc[i - rolling_window : i]
        mu = expected_returns.ema_historical_return(current_window_returns[tickers], frequency=12)
        S = CovarianceShrinkage(current_window_returns[tickers], frequency=12).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.40))

        last_spy_price = monthly_returns['SPY'].iloc[i-1]
        current_spy_ma = spy_ma.iloc[i-1]

        try:
            if last_spy_price > current_spy_ma:
                weights = ef.max_sharpe()
            else:
                weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            all_weights.append(cleaned_weights)
        except Exception:
            if all_weights: cleaned_weights = all_weights[-1]
            else: continue

        hold_start, hold_end = i, i + rebalance_frequency
        holding_period_returns = asset_returns.iloc[hold_start:hold_end]
        weights_series = pd.Series(cleaned_weights)
        period_portfolio_return = (holding_period_returns * weights_series).sum(axis=1)
        portfolio_returns.extend(period_portfolio_return)
        dates.extend(period_portfolio_return.index)

    backtest_results = pd.Series(portfolio_returns, index=dates)
    aligned_benchmark = benchmark_returns.loc[backtest_results.index]
    cumulative_strategy_returns = (1 + backtest_results).cumprod()
    cumulative_benchmark_returns = (1 + aligned_benchmark).cumprod()

    final_results = pd.DataFrame({
        'Adaptive MPT Strategy': cumulative_strategy_returns,
        'S&P 500 Benchmark': cumulative_benchmark_returns
    })
    weights_df = pd.DataFrame(all_weights, index=dates[::rebalance_frequency][:len(all_weights)])
    weights_df.index.name = 'Date'
    return final_results, weights_df

def run_monte_carlo(expected_return, volatility, weights, portfolio_value, years=10, simulations=5000):
    """Runs a Monte Carlo simulation for the portfolio."""
    daily_return = expected_return / 252
    daily_vol = volatility / np.sqrt(252)
    
    simulation_results = np.zeros((simulations, years * 252))
    
    for i in range(simulations):
        prices = [portfolio_value]
        for _ in range(years * 252):
            daily_sim_return = np.random.normal(daily_return, daily_vol)
            prices.append(prices[-1] * (1 + daily_sim_return))
        simulation_results[i, :] = prices[1:]

    return pd.DataFrame(simulation_results).T

def calculate_performance_metrics(returns):
    """Calculates key performance metrics from a series of returns."""
    total_return = (1 + returns).prod()
    num_years = len(returns) / 12
    cagr = total_return ** (1 / num_years) - 1
    
    annualized_volatility = returns.std() * np.sqrt(12)
    
    # Assume a risk-free rate of 1% for Sharpe Ratio calculation
    risk_free_rate = 0.01
    sharpe_ratio = (cagr - risk_free_rate) / annualized_volatility
    
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    return cagr, annualized_volatility, sharpe_ratio, max_drawdown

# ===================================================================
# STREAMLIT APP UI
# ===================================================================
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Strategy Backtest", "Live Portfolio Optimizer"])

# --- PAGE 1: STRATEGY BACKTEST ---
# --- PAGE 1: STRATEGY BACKTEST (with KPIs) ---
if page == "Strategy Backtest":
    st.title("🚀 Adaptive Portfolio Strategy Backtest")
    st.markdown("This page shows the backtest results of the pre-built adaptive strategy...")
    
    # --- NEW: "Explain the Strategy" Section ---
    with st.expander("Click here to learn how this strategy works"):
        st.markdown("""
            This portfolio isn't static; it's **adaptive**. It changes its investment objective based on the overall health of the market, making it a powerful tool for navigating different economic cycles.

            ### The Regime Filter 🚦
            The "brain" of the strategy is a **regime filter**, which we determine using a simple yet effective technical indicator:
            - We calculate the **12-month (approx. 200-day) simple moving average (SMA)** for the S&P 500 (SPY).
            - This SMA acts as a dynamic trendline, representing the long-term market trend.

            ### Two Modes of Operation:
            At each quarterly rebalancing, the strategy checks the market's current state:

            1.  **"Risk-On" Mode (Bull Market):** 📈
                - **Condition:** If the current price of SPY is **above** its 12-month SMA.
                - **Action:** The portfolio optimizes for the **Maximum Sharpe Ratio**. The goal is to achieve the best possible risk-adjusted return, capturing upside during healthy market uptrends.

            2.  **"Risk-Off" Mode (Bear Market):** 📉
                - **Condition:** If the current price of SPY is **below** its 12-month SMA.
                - **Action:** The portfolio's objective switches to **Minimum Volatility**. The goal is to build the safest possible portfolio to preserve capital and minimize losses during market downturns.

            By dynamically switching between these two modes, the strategy aims to provide a smoother ride and better long-term performance than a static portfolio.
        """)
    
    # This helper function goes inside this page's logic or at the top of the file
    def calculate_performance_metrics(returns):
        total_return = (1 + returns).prod()
        num_years = len(returns) / 12
        cagr = total_return ** (1 / num_years) - 1
        annualized_volatility = returns.std() * np.sqrt(12)
        risk_free_rate = 0.01
        sharpe_ratio = (cagr - risk_free_rate) / annualized_volatility
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        return cagr, annualized_volatility, sharpe_ratio, max_drawdown
    
    # --- NEW: Helper function for the drawdown plot ---
    def calculate_drawdown(cumulative_returns):
        """Calculates the drawdown series from a cumulative returns series."""
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        return drawdown

    # Run backtest to get the results
    results_df, weights_df = run_backtest()
    
    # --- NEW: Calculate and Display KPIs ---
    st.header("Key Performance Metrics")
    
    # Calculate daily returns for metrics calculation
    strategy_returns = results_df['Adaptive MPT Strategy'].pct_change().dropna()
    benchmark_returns_aligned = results_df['S&P 500 Benchmark'].pct_change().dropna()

    strat_cagr, strat_vol, strat_sharpe, strat_mdd = calculate_performance_metrics(strategy_returns)
    bench_cagr, bench_vol, bench_sharpe, bench_mdd = calculate_performance_metrics(benchmark_returns_aligned)

    # Display metrics in columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Adaptive MPT Strategy")
        st.metric("CAGR", f"{strat_cagr:.2%}")
        st.metric("Annualized Volatility", f"{strat_vol:.2%}")
        st.metric("Sharpe Ratio", f"{strat_sharpe:.2f}")
        st.metric("Max Drawdown", f"{strat_mdd:.2%}")
        
    with col2:
        st.subheader("S&P 500 Benchmark")
        st.metric("CAGR", f"{bench_cagr:.2%}")
        st.metric("Annualized Volatility", f"{bench_vol:.2%}")
        st.metric("Sharpe Ratio", f"{bench_sharpe:.2f}")
        st.metric("Max Drawdown", f"{bench_mdd:.2%}")

    # --- Performance Chart (Existing Code) ---
    st.header("Strategy Performance vs. Benchmark")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df.index, results_df.iloc[:, 0], label=results_df.columns[0])
    ax.plot(results_df.index, results_df.iloc[:, 1], label=results_df.columns[1])
    ax.set_title('Portfolio Strategy vs. Benchmark - Cumulative Returns', fontsize=16)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # --- NEW: Drawdown Plot Section ---
    st.header("Drawdown Over Time")
    st.markdown("This chart shows the percentage loss from the last peak. It's a key indicator of risk and resilience. Note how the adaptive strategy experienced smaller drawdowns during major market downturns like in 2022.")

    # Calculate drawdowns for both series
    strategy_drawdown = calculate_drawdown(results_df['Adaptive MPT Strategy'])
    benchmark_drawdown = calculate_drawdown(results_df['S&P 500 Benchmark'])

    # Plot the drawdowns
    fig_dd, ax_dd = plt.subplots(figsize=(12, 6))
    ax_dd.plot(strategy_drawdown.index, strategy_drawdown, label='Adaptive MPT Strategy')
    ax_dd.plot(benchmark_drawdown.index, benchmark_drawdown, label='S&P 500 Benchmark')
    ax_dd.set_title('Drawdown Comparison', fontsize=16)
    ax_dd.set_xlabel('Date')
    ax_dd.set_ylabel('Drawdown')
    ax_dd.legend()
    ax_dd.grid(True)
    # Format Y-axis as percentage
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    st.pyplot(fig_dd)
    
    # --- Allocation Chart (Existing Code) ---
    st.header("Portfolio Allocation Over Time")
    weights_df = weights_df.clip(lower=0)
    fig_weights, ax_weights = plt.subplots(figsize=(12, 6))
    weights_df.plot(kind='bar', stacked=True, ax=ax_weights)
    ax_weights.set_title('Asset Allocation Over Time', fontsize=16)
    fig_weights.autofmt_xdate()
    st.pyplot(fig_weights)
    
    # --- Display Raw Data (Optional) ---
    with st.expander("View Raw Performance Data"):
        st.dataframe(results_df)

    with st.expander("View Historical Allocations"):
        st.dataframe(weights_df.fillna(0).applymap(lambda x: f"{x:.2%}"))
    
    

# --- PAGE 2: LIVE PORTFOLIO OPTIMIZER ---
elif page == "Live Portfolio Optimizer":
    st.title("💸 Live Portfolio Optimizer")
    st.markdown("Enter your desired stocks, portfolio value, and constraints to get an optimized allocation.")
    
    tickers_str = st.text_area("Enter stock tickers (separated by commas or new lines)", "AAPL, GOOG, MSFT, TSLA, JPM, V, NVDA")
    portfolio_value = st.number_input("Enter your total portfolio value", min_value=1000, value=100000, step=1000)
    
    # --- FEATURE 1: Customizable Goal & Constraints ---
    col1, col2 = st.columns(2)
    with col1:
        optimization_goal = st.selectbox("Select Optimization Goal",["Maximize Sharpe Ratio", "Minimize Volatility"])
    with col2:
        max_weight = st.slider("Set Max Allocation per Stock (%)", min_value=5, max_value=100, value=35, step=5)

    if st.button("Optimize Portfolio"):
        tickers = [t.strip().upper() for t in tickers_str.replace(",", "\n").split() if t.strip()]
        if not tickers:
            st.warning("Please enter at least one ticker.")
        else:
            with st.spinner("Fetching data and running optimization..."):
                try:
                    df = yf.download(tickers, period="5y")['Close']
                    if df.empty or df.shape[1] != len(tickers):
                        st.error("Could not fetch data for one or more tickers. Please check them.")
                    else:
                        mu = expected_returns.ema_historical_return(df)
                        S = CovarianceShrinkage(df).ledoit_wolf()
                        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight / 100)) # Use the slider value

                        if optimization_goal == "Maximize Sharpe Ratio":
                            weights = ef.max_sharpe()
                        else:
                            weights = ef.min_volatility()
                            
                        cleaned_weights = ef.clean_weights()
                        perf = ef.portfolio_performance(verbose=False)
                        
                        st.success("Optimization Complete!")
                        
                        # Display Results in columns
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Optimal Asset Weights")
                            weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])
                            st.dataframe(weights_df.style.format("{:.2%}"))
                            
                            st.subheader("Discrete Allocation")
                            latest_prices = get_latest_prices(df)
                            da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_value)
                            allocation, leftover = da.lp_portfolio()
                            st.write(f"Number of shares to buy for a **${portfolio_value:,.2f}** portfolio:")
                            st.write({k: v for k, v in allocation.items() if v > 0})
                            st.write(f"**Funds remaining:** ${leftover:.2f}")

                        with col2:
                            st.subheader("Expected Portfolio Performance")
                            st.metric("Annual Return", f"{perf[0]:.2%}")
                            st.metric("Annual Volatility", f"{perf[1]:.2%}")
                            st.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                            
                            # --- FEATURE 5: Download Button ---
                            csv = weights_df.to_csv().encode('utf-8')
                            st.download_button(
                                label="Download Weights as CSV",
                                data=csv,
                                file_name='optimal_weights.csv',
                                mime='text/csv',
                            )
                        
                        st.divider()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # --- FEATURE 2: Interactive Efficient Frontier Plot (Robust Version) ---
                            st.subheader("Efficient Frontier")
                            fig, ax = plt.subplots()

                            # Create a 'base' instance that will NOT be used for optimization, only for copying.
                            ef_base = EfficientFrontier(mu, S)

                            # --- 1. Create a copy for the main plot ---
                            ef_for_plotting = ef_base.deepcopy()
                            plotting.plot_efficient_frontier(ef_for_plotting, ax=ax, show_assets=False)

                            # --- 2. Create a fresh copy for the Max Sharpe calculation ---
                            ef_sharpe = ef_base.deepcopy() 
                            ef_sharpe.max_sharpe()
                            ret_s, risk_s, _ = ef_sharpe.portfolio_performance()
                            ax.scatter(risk_s, ret_s, marker="*", color="r", s=150, label="Max Sharpe")

                            # --- 3. Create another fresh copy for the Min Volatility calculation ---
                            ef_vol = ef_base.deepcopy()
                            ef_vol.min_volatility()
                            ret_v, risk_v, _ = ef_vol.portfolio_performance()
                            ax.scatter(risk_v, ret_v, marker="*", color="g", s=150, label="Min Volatility")

                            ax.legend()
                            st.pyplot(fig)
                        
                        with col2:
                            # --- FEATURE 3: Asset Correlation Heatmap ---
                            st.subheader("Asset Correlation Heatmap")
                            fig_corr, ax_corr = plt.subplots()
                            sns.heatmap(df.pct_change().corr(), ax=ax_corr, cmap='viridis', annot=True, fmt=".2f")
                            st.pyplot(fig_corr)
                            
                        st.divider()

                        # --- FEATURE 4: Monte Carlo Simulation ---
                        st.subheader("10-Year Monte Carlo Simulation")
                        sim_results = run_monte_carlo(perf[0], perf[1], cleaned_weights, portfolio_value)
                        fig_mc, ax_mc = plt.subplots(figsize=(10, 6))
                        ax_mc.plot(sim_results)
                        ax_mc.set_title("Range of Potential Portfolio Outcomes")
                        ax_mc.set_xlabel("Trading Days")
                        ax_mc.set_ylabel("Portfolio Value ($)")
                        st.pyplot(fig_mc)

                except Exception as e:
                    st.error(f"An error occurred: {e}")