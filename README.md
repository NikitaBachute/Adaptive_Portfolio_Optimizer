# 📈 Adaptive Portfolio Optimizer

This is an interactive web application built with Streamlit that provides tools for advanced portfolio analysis and optimization. It features two main sections:
1.  **Strategy Backtest**: Demonstrates and analyzes a pre-built adaptive investment strategy that switches its objective based on market conditions.
2.  **Live Portfolio Optimizer**: An interactive tool for users to build, optimize, and forecast their own custom stock portfolios based on Modern Portfolio Theory (MPT).

## 🚀 Features

### 1. Strategy Backtest Page
This page provides a complete analysis of a dynamic trading strategy from 2012 to 2024.

* **Adaptive Logic**: The strategy is not static. It adapts to the market regime:
    * **📈 Risk-On (Bull Market)**: When the S&P 500 (SPY) is above its 12-month moving average, the portfolio optimizes for the **Maximum Sharpe Ratio** to capture upside.
    * **📉 Risk-Off (Bear Market)**: When SPY is below its 12-month moving average, the portfolio switches to a **Minimum Volatility** objective to preserve capital.
* **Performance vs. Benchmark**: Compares the cumulative returns of the adaptive strategy against a standard S&P 500 benchmark.
* **Key Performance Indicators (KPIs)**: Displays critical metrics for both the strategy and the benchmark, including:
    * Compound Annual Growth Rate (CAGR)
    * Annualized Volatility
    * Sharpe Ratio
    * Maximum Drawdown
* **Visualizations**:
    * **Drawdown Chart**: A visual comparison of portfolio losses from peak values, highlighting the strategy's risk management effectiveness.
    * **Allocation Over Time**: A stacked bar chart showing how the portfolio's asset allocation has changed at each rebalancing period.

### 2. Live Portfolio Optimizer Page
An interactive dashboard for creating and analyzing your own portfolios.

* **Custom Inputs**: Users can input their own list of stock tickers and define the total portfolio value.
* **Customizable Optimization**:
    * Choose your investment goal: **Maximize Sharpe Ratio** or **Minimize Volatility**.
    * Set constraints, such as the maximum allowable weight for any single stock.
* **Detailed Results**:
    * **Optimal Weights**: A clear breakdown of the ideal percentage allocation for each asset.
    * **Discrete Allocation**: A practical list of the exact number of shares to purchase for each stock based on the portfolio value.
    * **Expected Performance**: Forward-looking estimates for Annual Return, Volatility, and Sharpe Ratio.
* **Advanced Analytics & Visuals**:
    * **Efficient Frontier Plot**: An interactive chart showing the set of optimal portfolios, with the Max Sharpe and Min Volatility points highlighted.
    * **Asset Correlation Heatmap**: A matrix that visualizes how the selected assets move in relation to one another.
    * **Monte Carlo Simulation**: A 10-year simulation that projects thousands of potential future outcomes for the optimized portfolio, providing a probabilistic range of future values.
* **Export Functionality**: Download the calculated optimal weights as a CSV file for your records.


## 🛠️ Tech Stack & Libraries

This application is built entirely in Python and leverages the following powerful libraries:

* **Streamlit**: For creating the interactive web application UI.
* **PyPortfolioOpt**: The core engine for all Modern Portfolio Theory calculations, including efficient frontier optimization and performance metrics.
* **yfinance**: For downloading historical stock price data from Yahoo Finance.
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical operations, especially in the Monte Carlo simulation.
* **Matplotlib & Seaborn**: For creating the charts and visualizations.

## ⚙️ Installation & Setup

To run this application on your local machine, follow these steps:

1.  **Prerequisites**:
    * Ensure you have Python 3.8 or newer installed.

2.  **Clone the Repository (or save the file)**:
    * Save the Python script as `app.py`.

3.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install Required Libraries**:
    * Create a file named `requirements.txt` and add the following lines:
        ```
        streamlit
        yfinance
        pandas
        numpy
        matplotlib
        seaborn
        PyPortfolioOpt
        ```
    * Install all dependencies with pip:
        ```bash
        pip install -r requirements.txt
        ```

5.  **Run the Application**:
    * Open your terminal, navigate to the directory where you saved `app.py`, and run the following command:
        ```bash
        streamlit run app.py
        ```
    * The application will open in a new tab in your default web browser.
