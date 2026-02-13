# Portfolio Risk & Return Optimization Platform

## Overview

This project is a Python-based portfolio optimization platform that applies Modern Portfolio Theory (MPT) to analyze risk-return tradeoffs and compute optimal asset allocations under different investment objectives.

The model retrieves historical market data, computes annualized returns and covariance matrices, and performs constrained nonlinear optimization to determine optimal portfolio weights.

## Key Features

- Historical price retrieval via Yahoo Finance API
- Annualized return and covariance matrix computation (252 trading days)
- Sharpe ratio maximization
- Portfolio variance minimization
- Expected return maximization
- Full-investment weight constraints (weights sum to 1, no short-selling)
- Correlation matrix analysis
- Interactive Shiny-based analytics dashboard
- Risk-return tradeoff visualization
- Cumulative portfolio performance tracking

## Optimization Methodology

The model implements constrained nonlinear optimization using Sequential Least Squares Programming (SLSQP).

### Objective Functions

- Maximize Sharpe Ratio  
- Minimize Portfolio Variance  
- Maximize Expected Return  

### Constraints

- Sum of weights = 1  
- Bounds: 0 ≤ weight ≤ 1  

Risk-free rate assumption: 2%  
Return scaling: 252 trading days (annualized)

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- Shiny for Python
- Yahooquery API

## How to Run Locally

1. Clone the repository:
   git clone https://github.com/alex-veiga03/portfolio-risk-optimization.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   python app.py

## Purpose

This project was developed to demonstrate quantitative portfolio construction, risk modeling, and applied optimization techniques relevant to investment analytics and asset management roles.
