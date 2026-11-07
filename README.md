# Stock Research Assistant

A small Streamlit app that helps you research a stock: price chart, a few transparent indicators, available fundamentals, recent headlines and a plain-language analysis.

> **Disclaimer.** This is an educational research tool, not financial advice. Market data comes from Yahoo Finance through the unofficial `yfinance` library and may be delayed, incomplete or incorrect. Indicators describe the past and do not predict the future. Verify information with authoritative sources before making any decision.

## What it shows

- Company name, ticker, sector, latest close, the date of that close and the currency
- A price chart with 20-day and 50-day averages (periods: 1 month, 6 months, 1 year, 5 years)
- Period return, 20-day and 50-day averages, 14-day RSI, annualized volatility and maximum drawdown, each with a plain-language explanation and its limits
- Market cap, P/E, revenue, revenue growth, profit margin, dividend yield and 52-week range. Anything Yahoo does not provide is shown as "unavailable"
- Recent headlines with source and date, when the data source returns them
- A rule-based analysis: what the data shows, positive factors, risks, what could change the view, and an evidence summary of "Positive", "Mixed" or "Negative" with its reasoning
- Downloads: price data with indicators (CSV) and the full report (text)

The app never places trades, connects to a broker, needs an API key or stores what you type.

## Run it on your computer

You need Python 3.9 or newer.

```bash
cd stock-research-agent
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local address (usually http://localhost:8501) and opens it in your browser.

## Get a link you can share (free)

Streamlit Community Cloud hosts Streamlit apps from a GitHub repository and gives each app a public web address.

1. Create a free account at [github.com](https://github.com) and make a new **public** repository, for example `stock-research-agent`.
2. In the repository, choose **Add file > Upload files** and upload `app.py`, `requirements.txt` and `README.md`. Commit them to the `main` branch.
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub. Allow access to your repository when asked.
4. Click **Create app** and choose your repository. Set the branch to `main` and the main file path to `app.py`. You can pick your own app address if the form offers it.
5. Click **Deploy**. The first start takes a few minutes while the libraries install.
6. Copy the address of the running app (it ends in `.streamlit.app`) and send it to anyone.

The button names can change slightly over time. Streamlit's own guide is at [docs.streamlit.io](https://docs.streamlit.io/deploy/streamlit-community-cloud). Free apps may go to sleep when nobody has used them for a while and wake up when someone opens the link, so the first visit can be slow.

To change the app later, edit `app.py` on GitHub and commit. The deployed app updates itself.

## How to use it

1. Type a ticker in the sidebar and press Enter, or click an example. Use an exchange suffix for stocks outside the US, such as `RELIANCE.NS` for NSE India.
2. Pick a time period.
3. Read the chart, then the tabs: Analysis, Indicators, Fundamentals and Headlines.
4. Download the data or the report if you want to keep it.

## Limits and troubleshooting

- **Data source.** `yfinance` is not an official Yahoo product. Fields can be missing, change names or stop working without notice.
- **Rate limiting.** Yahoo may block requests from shared cloud servers, which can make a hosted copy fail for a few minutes. The app caches results for 15 minutes to reduce requests. If you see a data error, wait a few minutes and press Enter in the ticker box again. If it keeps failing, update the library by changing the `yfinance` line in `requirements.txt`.
- **Short periods.** The 50-day average needs at least 50 trading days, so it is unavailable for the 1 month period. RSI needs at least 15 prices.
- **Prices.** Closing prices are adjusted for stock splits but not for dividends, so returns exclude dividends.
- **Analysis.** The evidence summary only counts price-trend signals in the chosen window. It does not judge the business or its valuation, and it is not a recommendation.
- **Headlines.** Shown as the data source provides them. They are not checked or analysed here.
- **Privacy.** Ticker lookups go to Yahoo Finance. The app does not store them, but a hosting service may keep standard access logs.
