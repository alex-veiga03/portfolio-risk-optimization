from shiny import App, ui, render, reactive
from yahooquery import Ticker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the UI
app_ui = ui.page_fluid(
    ui.panel_title("Enhanced Portfolio Risk and Return Analyzer"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_text("tickers", "Enter Asset Tickers (comma-separated):", value="AAPL, MSFT, NVDA, TSLA"),
            ui.input_date_range("date_range", "Select Date Range:", start="2015-01-01", end="2023-01-01"),
            ui.input_radio_buttons(
                "optimization_mode",
                "Optimization Mode:",
                choices={"optimize": "Optimize Weights", "import": "Import Weights"},
                selected="optimize",
            ),
            ui.input_text(
                "imported_weights",
                "Enter Imported Weights (comma-separated, in %):",
                value="",
                placeholder="e.g., 40, 40, 20",
            ),
            ui.input_select("optimize_for", "Optimize for:", choices=["maximize sharpe", "maximize return", "minimize risk"], selected="maximize sharpe"),
            ui.input_action_button("analyze", "Analyze Portfolio"),
        ),
        ui.panel_main(
            ui.h3("Portfolio Analysis Results"),
            ui.output_text("portfolio_metrics"),
            ui.navset_tab(
                ui.nav("Risk-Return Chart", ui.output_plot("risk_return_chart")),
                ui.nav("Historical Performance", ui.output_plot("historical_performance")),
                ui.nav("Allocation Pie Chart", ui.output_plot("allocation_pie_chart")),
                ui.nav("Correlation Heatmap", ui.output_plot("correlation_heatmap")),
            ),
        ),
    ),
)

# Portfolio optimization logic
def optimize_portfolio_nonlinear(mean_returns, cov_matrix, risk_free_rate=0.02, optimize_for="maximize sharpe"):
    num_assets = len(mean_returns)

    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights

    def portfolio_return(weights):
        return np.sum(mean_returns * weights)

    def sharpe_ratio(weights):
        port_return = portfolio_return(weights)
        port_variance = portfolio_variance(weights)
        return -(port_return - risk_free_rate) / np.sqrt(port_variance)

    if optimize_for == "maximize sharpe":
        objective = sharpe_ratio
    elif optimize_for == "maximize return":
        objective = lambda w: -portfolio_return(w)
    elif optimize_for == "minimize risk":
        objective = portfolio_variance
    else:
        raise ValueError("Invalid optimization metric. Choose 'sharpe', 'return', or 'risk'.")

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0) for _ in range(num_assets)]
    init_guess = np.array([1.0 / num_assets] * num_assets)

    result = minimize(objective, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError("Optimization failed:", result.message)

    return result.x

# Define Server Logic
def server(input, output, session):
    @reactive.Calc
    def analyze_portfolio():
        tickers = [ticker.strip() for ticker in input.tickers().split(",") if ticker.strip() != ""]
        ticker_data = Ticker(tickers)
        prices = pd.DataFrame()

        for ticker in tickers:
            try:
                ticker_history = ticker_data.history(start=input.date_range()[0], end=input.date_range()[1])
                if "adjclose" in ticker_history.loc[ticker].columns:
                    price_series = ticker_history.loc[ticker]["adjclose"].ffill().bfill()
                else:
                    price_series = ticker_history.loc[ticker]["close"].ffill().bfill()
                prices[ticker] = price_series
            except Exception as e:
                raise ValueError(f"Error fetching data for {ticker}: {e}")

        if prices.empty:
            raise ValueError("No valid data retrieved for the selected tickers.")

        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        if input.optimization_mode() == "import":
            if not input.imported_weights().strip():
                raise ValueError("Imported weights cannot be empty when 'Import Weights' is selected.")
            
            weights = np.array([float(w.strip()) / 100 for w in input.imported_weights().split(",")])
            if len(weights) != len(tickers):
                raise ValueError("Imported weights do not match the number of tickers.")
            if not np.isclose(np.sum(weights), 1, atol=0.001):
                raise ValueError("Imported weights must sum to approximately 100% (with minor tolerance).")
        else:
            weights = optimize_portfolio_nonlinear(mean_returns.values, cov_matrix.values, optimize_for=input.optimize_for())

        portfolio_return = np.dot(mean_returns.values, weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility

        return {
            "returns": returns,
            "tickers": tickers,
            "weights": weights,
            "portfolio_return": portfolio_return,
            "portfolio_volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "cov_matrix": cov_matrix,
            "mean_returns": mean_returns,
            "prices": prices,
        }

    @output
    @render.text
    def portfolio_metrics():
        result = analyze_portfolio()
        weights = result["weights"]
        tickers = result["tickers"]
    
    # Format the weights section
        weights_str = "\n".join(f"{tickers[i]}: {weights[i] * 100:.2f}%" for i in range(len(tickers)))
    
    # Create output with explicit newlines for separation
        return (
            f"Portfolio Weights:\n{weights_str}\n\n"
            f"Expected Annual Return:\n{result['portfolio_return']:.2%}\n"
            f"Portfolio Volatility:\n{result['portfolio_volatility']:.2%}\n"
            f"Sharpe Ratio:\n{result['sharpe_ratio']:.2f}"
        )

    @output
    @render.plot
    def risk_return_chart():
        result = analyze_portfolio()
        tickers = result["tickers"]
        risk = np.sqrt(np.diag(result["cov_matrix"]))
        returns = result["mean_returns"]

        fig, ax = plt.subplots()
        ax.scatter(risk, returns, c="blue", label="Assets")
        for i, ticker in enumerate(tickers):
            ax.text(risk[i], returns.iloc[i], ticker, fontsize=10)
        ax.scatter(result["portfolio_volatility"], result["portfolio_return"], c="red", label="Portfolio", s=50)
        ax.text(result["portfolio_volatility"], result["portfolio_return"], "Portfolio", fontsize=10, color="red")
        ax.set_xlabel("Risk (Volatility)")
        ax.set_ylabel("Return")
        ax.set_title("Risk-Return Tradeoff")
        ax.legend()
        return fig

    @output
    @render.plot
    def historical_performance():
        result = analyze_portfolio()
        prices = result["prices"]
        weights = result["weights"]

    # Calculate cumulative returns for each asset
        returns = prices.pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()

    # Calculate portfolio cumulative return
        portfolio_return = (returns * weights).sum(axis=1)
        cumulative_portfolio_return = (1 + portfolio_return).cumprod()

    # Plot the cumulative returns
        fig, ax = plt.subplots()
        cumulative_returns.plot(ax=ax, legend=True)
        ax.plot(cumulative_portfolio_return.index, cumulative_portfolio_return, label="Portfolio", color="red", linewidth=2)
        ax.set_title("Cumulative Returns Over Time")
        ax.set_ylabel("Cumulative Return")
        ax.set_xlabel("Date")
        ax.legend()
        return fig


    @output
    @render.plot
    def allocation_pie_chart():
        result = analyze_portfolio()
        weights = result["weights"]
        tickers = result["tickers"]

        fig, ax = plt.subplots()
        ax.pie(weights, labels=tickers, autopct="%1.1f%%", startangle=140)
        ax.set_title("Portfolio Allocation")
        return fig

    @output
    @render.plot
    def correlation_heatmap():
        result = analyze_portfolio()
        returns = result["returns"]
        correlation_matrix = returns.corr()

        fig, ax = plt.subplots()
        cax = ax.matshow(correlation_matrix, cmap="coolwarm")
        fig.colorbar(cax)
        ax.set_xticks(np.arange(len(correlation_matrix.columns)))
        ax.set_yticks(np.arange(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=90)
        ax.set_yticklabels(correlation_matrix.columns)
        ax.set_title("Correlation Heatmap")
        return fig

# Run the app
app = App(app_ui, server)






