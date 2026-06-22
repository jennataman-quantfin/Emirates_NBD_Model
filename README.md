# Emirates NBD Quantitative Investment Dashboard

## Overview

The Emirates NBD Quantitative Investment Dashboard is a comprehensive financial analytics platform developed to evaluate the investment potential of Emirates NBD (EMIRATESNBD.AE) through the integration of quantitative finance, machine learning, valuation methodologies, and macroeconomic analysis.

This project replicates a professional equity research workflow commonly employed by investment analysts and portfolio managers. By combining fundamental analysis, technical indicators, predictive modeling, and risk assessment techniques, the dashboard transforms financial and market data into actionable investment insights.

---

## Live Application

The dashboard is deployed as a web-based application and can be accessed directly through a browser without local installation.

**Live Dashboard:**
https://emiratesnbdmodel-3zz77ojwtlxgaztdvuc99k.streamlit.app/
---

## Project Objectives

The primary objective of this project is to develop a data-driven investment decision support system capable of:

* Evaluating the financial strength and valuation of Emirates NBD.
* Identifying market trends through technical analysis.
* Forecasting potential price movements using machine learning models.
* Quantifying investment risk through probabilistic simulations.
* Assessing the impact of macroeconomic and geopolitical factors on stock performance.
* Supporting informed investment decisions through an integrated analytical framework.

---

## Core Analytical Components

### Market and Technical Analysis

The dashboard provides comprehensive market analytics through:

* Real-time stock market data integration via Finnhub API.
* Interactive candlestick charts and trading volume visualization.
* Relative Strength Index (RSI) analysis.
* Moving Average Convergence Divergence (MACD) indicators.
* Bollinger Bands for volatility assessment.
* Moving average trend identification and signal generation.

### Fundamental and Valuation Analysis

To evaluate the intrinsic value and financial health of the company, the platform incorporates:

* More than sixteen key financial ratios, including:

  * Price-to-Earnings (P/E)
  * Price-to-Book (P/B)
  * Return on Equity (ROE)
  * Return on Assets (ROA)
  * Dividend Yield
* Enterprise Value (EV) analysis.
* EV/EBITDA valuation metrics.
* Discounted Cash Flow (DCF) valuation models.
* Intrinsic value estimation and margin of safety calculations.

### Quantitative and Predictive Modeling

The forecasting framework utilizes an ensemble machine learning architecture consisting of:

* Random Forest Regression (30%)
* Gradient Boosting Regression (30%)
* Linear Regression (40%)

Additional features include:

* Walk-forward validation and backtesting.
* Multi-factor investment signal generation.
* Buy/Sell recommendation engine based on a six-factor scoring methodology.

### Risk Analysis and Simulation

Investment risk is evaluated through:

* Monte Carlo simulation using 10,000 stochastic scenarios.
* Future price distribution modeling.
* Downside risk assessment.
* Risk-adjusted return evaluation.
* Probability-based forecasting.

### Macroeconomic and Scenario Analysis

The dashboard incorporates external economic variables to assess broader market influences, including:

* Correlation analysis between Emirates NBD stock performance and Brent crude oil prices.
* Inflation-adjusted return calculations.
* Comparative analysis of nominal and real returns.
* Scenario-based forecasting under multiple economic and geopolitical conditions.

### News and Sentiment Analysis

To capture market sentiment, the system integrates:

* Real-time financial news aggregation.
* Sentiment analysis of news content.
* Qualitative insights supporting quantitative findings.

### Investment Planning Tools

Practical investment evaluation tools include:

* Historical profit and loss analysis.
* Investment growth projections.
* Portfolio return estimation.

---

## Dashboard Architecture

The platform consists of thirteen integrated analytical modules:

1. Price and Technical Analysis
2. Financial Ratio Analysis
3. Enterprise Value Assessment
4. Investment Signal Engine
5. Machine Learning Forecasting
6. Monte Carlo Simulation
7. Investment Calculator
8. Discounted Cash Flow Valuation
9. Backtesting Framework
10. Scenario Analysis
11. Inflation Adjustment Analysis
12. Oil Price Correlation Analysis
13. News and Sentiment Monitoring

---

## Technical Skills Demonstrated

This project demonstrates proficiency across multiple domains within finance and data science, including:

* Equity Research and Security Analysis
* Corporate Valuation Techniques
* Quantitative Finance and Risk Modeling
* Machine Learning Applications in Finance
* Financial Data Analysis using Python
* Statistical Modeling and Forecasting
* Interactive Data Visualization
* API Integration and Real-Time Data Processing
* Scenario and Sensitivity Analysis

---

## Technology Stack

### Frontend

* Streamlit

### Data Sources

* yFinance
* Finnhub API
* NewsAPI

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Visualization

* Plotly

---

## Installation and Deployment

### Prerequisites

* Python 3.8 or higher
* pip package manager

### Repository Setup

```bash
git clone https://github.com/jennataman-quantfin/Emirates_NBD_Model.git
cd Emirates_NBD_Model
```

### Dependency Installation

```bash
pip install -r requirements.txt
```

### API Configuration

The application requires API credentials from:

* Finnhub
* NewsAPI

Insert the credentials into the application configuration:

```python
NEWS_API_KEY = "your_newsapi_key"
FINNHUB_API_KEY = "your_finnhub_key"
```

### Running the Application

```bash
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

## Illustrative Insights

The investment recommendation framework integrates valuation metrics, technical indicators, predictive analytics, and risk factors into a structured scoring methodology.

Monte Carlo simulations further enhance decision-making by estimating:

* Expected return distributions.
* Downside risk probabilities.
* Confidence intervals for future price outcomes.
* Potential investment scenarios under varying market conditions.

---

## Disclaimer

This project has been developed solely for educational and research purposes.

The analyses, forecasts, and investment signals generated by the platform should not be interpreted as financial advice. The project is not affiliated with Emirates NBD, and historical performance does not guarantee future investment outcomes.

---

## Author

**Jennataman Urmi**

LinkedIn: https://linkedin.com/in/jennataman-urmi

Email: [jennataman.urmi77@gmail.com](mailto:jennataman.urmi77@gmail.com)

---

## License

This project is distributed under the MIT License.
