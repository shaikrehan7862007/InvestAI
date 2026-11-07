"""Stock Research Assistant: an educational tool, not financial advice.

Run locally:  streamlit run app.py
All market data comes from Yahoo Finance through the unofficial yfinance library.
"""

import datetime as dt
import re

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

PERIODS = {"1 month": "1mo", "6 months": "6mo", "1 year": "1y", "5 years": "5y"}
EXAMPLES = ["AAPL", "MSFT", "RELIANCE.NS"]
TRADING_DAYS = 252  # trading days per year, used to annualize volatility

DISCLAIMER = (
    "**Educational research tool, not financial advice.** Market data comes from an "
    "unofficial source (Yahoo Finance via yfinance) and may be delayed, incomplete or "
    "incorrect. Technical indicators describe the past and do not predict the future. "
    "Verify everything with authoritative sources, such as the company's filings and "
    "your exchange, before making any decision."
)

# Plain-language help for each indicator, shown in the app.
INDICATOR_HELP = {
    "Period return": (
        "How much the price changed from the first to the last day in the selected "
        "period. It uses the closing price and ignores dividends.",
        "A single start and end date can flatter or hurt a stock. Pick another period "
        "and the answer can change a lot.",
    ),
    "20-day average": (
        "The average closing price of the last 20 trading days, a rough line for the "
        "recent trend.",
        "It lags the price, so it reacts after a move has already happened.",
    ),
    "50-day average": (
        "The average closing price of the last 50 trading days, a slower trend line. "
        "Needs at least 50 trading days in the selected period.",
        "It lags even more than the 20-day average and often gives false signals when "
        "prices move sideways.",
    ),
    "RSI (14-day)": (
        "Relative Strength Index, scaled 0 to 100. It compares recent gains with recent "
        "losses. Readings above 70 are often called 'overbought' and below 30 "
        "'oversold'.",
        "A strong stock can stay above 70 for weeks. The labels are conventions, not "
        "forecasts.",
    ),
    "Annualized volatility": (
        "How much daily returns vary, scaled to a one-year figure. Higher means larger "
        "typical swings up and down.",
        "It is based on past moves only and says nothing about direction.",
    ),
    "Maximum drawdown": (
        "The biggest fall from a peak to a later low inside the selected period.",
        "It depends on the window you pick and does not show how long recovery took.",
    ),
}


# ----------------------------------------------------------------------------
# Small formatting helpers
# ----------------------------------------------------------------------------

def is_missing(x):
    """True for None and NaN."""
    return x is None or (isinstance(x, float) and np.isnan(x))


def fmt_pct(x, signed=False):
    if is_missing(x):
        return "unavailable"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def fmt_num(x, digits=2):
    if is_missing(x):
        return "unavailable"
    return f"{x:,.{digits}f}"


def fmt_big(x, currency=""):
    """Format large money amounts such as 2.95T or 410.2B."""
    if is_missing(x):
        return "unavailable"
    prefix = f"{currency} " if currency else ""
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(x) >= limit:
            return f"{prefix}{x / limit:,.2f}{suffix}"
    return f"{prefix}{x:,.0f}"


def clean_ticker(raw):
    """Return an upper-case ticker, or None if it looks invalid."""
    ticker = (raw or "").strip().upper()
    if re.fullmatch(r"[A-Z0-9.\-^=]{1,15}", ticker):
        return ticker
    return None


# ----------------------------------------------------------------------------
# Indicators (plain pandas and numpy, so each one is easy to follow)
# ----------------------------------------------------------------------------

def rsi_series(close, window=14):
    """Relative Strength Index using Wilder's smoothing.

    The first value needs `window` price changes (so window + 1 prices).
    It starts from a simple average of the first `window` gains and losses,
    then smooths each new day into that average.
    """
    values = np.full(len(close), np.nan)
    if len(close) <= window:
        return pd.Series(values, index=close.index)

    change = np.diff(close.to_numpy(dtype=float), prepend=np.nan)
    gains = np.where(change > 0, change, 0.0)
    losses = np.where(change < 0, -change, 0.0)

    avg_gain = gains[1 : window + 1].mean()
    avg_loss = losses[1 : window + 1].mean()
    for i in range(window, len(close)):
        if i > window:
            avg_gain = (avg_gain * (window - 1) + gains[i]) / window
            avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        if avg_loss > 0:
            values[i] = 100 - 100 / (1 + avg_gain / avg_loss)
        elif avg_gain > 0:
            values[i] = 100.0  # no losses at all in the window
        # if there were no gains and no losses the RSI is undefined, so it stays NaN
    return pd.Series(values, index=close.index)


def last_valid(series):
    """Last non-NaN value of a series, or None."""
    series = series.dropna()
    return float(series.iloc[-1]) if len(series) else None


def compute_indicators(close):
    """Return a dict of indicators. A value is None when there is not enough data."""
    close = close.dropna()
    daily_returns = close.pct_change().dropna()
    n = len(close)

    out = {
        "period_return": None,
        "ma20": None,
        "ma50": None,
        "rsi14": None,
        "volatility": None,
        "max_drawdown": None,
    }
    if n >= 2:
        out["period_return"] = float(close.iloc[-1] / close.iloc[0] - 1)
        out["max_drawdown"] = float((close / close.cummax() - 1).min())
    if n >= 20:
        out["ma20"] = float(close.rolling(20).mean().iloc[-1])
    if n >= 50:
        out["ma50"] = float(close.rolling(50).mean().iloc[-1])
    if n >= 15:
        out["rsi14"] = last_valid(rsi_series(close))
    if len(daily_returns) >= 2:
        out["volatility"] = float(daily_returns.std() * np.sqrt(TRADING_DAYS))
    return out


# ----------------------------------------------------------------------------
# Data loading (cached so many visitors do not hammer Yahoo Finance)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker, period):
    """Daily price history. Raises on network problems; returns an empty frame for bad tickers."""
    return yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker):
    # Errors are not cached by Streamlit, so a temporary failure is retried next time.
    return dict(yf.Ticker(ticker).info or {})


def load_info(ticker):
    """Company details. Returns {} if Yahoo does not provide them."""
    try:
        return fetch_info(ticker)
    except Exception:
        return {}


def load_news(ticker):
    """Recent headlines as a list of dicts. Returns [] if none can be fetched."""
    try:
        return fetch_news(ticker)
    except Exception:
        return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(ticker):
    items = yf.Ticker(ticker).news or []
    headlines = []
    for item in items[:8]:
        # Newer yfinance versions nest the details under "content".
        body = item.get("content") or item
        title = body.get("title")
        url = (
            (body.get("canonicalUrl") or {}).get("url")
            or (body.get("clickThroughUrl") or {}).get("url")
            or body.get("link")
        )
        published = body.get("pubDate") or body.get("displayTime")
        if not published and item.get("providerPublishTime"):
            published = dt.datetime.fromtimestamp(
                item["providerPublishTime"], tz=dt.timezone.utc
            ).isoformat()
        source = (body.get("provider") or {}).get("displayName") or body.get("publisher")
        if title and url:
            headlines.append(
                {
                    "title": title,
                    "url": url,
                    "date": str(published)[:10] if published else "date not provided",
                    "source": source or "source not provided",
                }
            )
    return headlines


def get_currency(ticker, info):
    """Currency code from company info, falling back to fast_info."""
    code = info.get("currency")
    if not code:
        try:
            code = yf.Ticker(ticker).fast_info.get("currency")
        except Exception:
            code = None
    return code or "unknown"


# ----------------------------------------------------------------------------
# Fundamentals
# ----------------------------------------------------------------------------

def build_fundamentals(info, price, currency):
    """Rows of (label, text). Anything Yahoo does not provide is marked unavailable."""
    yield_value = info.get("trailingAnnualDividendYield")
    if is_missing(yield_value) and info.get("dividendRate") and price:
        yield_value = info["dividendRate"] / price

    return [
        ("Market capitalization", fmt_big(info.get("marketCap"), currency)),
        ("P/E (trailing)", fmt_num(info.get("trailingPE"))),
        ("P/E (forward)", fmt_num(info.get("forwardPE"))),
        ("Revenue (trailing 12 months)", fmt_big(info.get("totalRevenue"), currency)),
        ("Revenue growth (latest quarter, year on year)", fmt_pct(info.get("revenueGrowth"), True)),
        ("Profit margin", fmt_pct(info.get("profitMargins"))),
        ("Dividend yield (trailing)", fmt_pct(yield_value)),
        ("52-week high", fmt_num(info.get("fiftyTwoWeekHigh"))),
        ("52-week low", fmt_num(info.get("fiftyTwoWeekLow"))),
    ]


# ----------------------------------------------------------------------------
# Analysis (rule based and fully visible: every sentence comes from fetched numbers)
# ----------------------------------------------------------------------------

def build_analysis(ind, price, info, start, end, has_news):
    facts, positives, risks, changes = [], [], [], []
    signals = []  # (description, +1 / 0 / -1)

    # --- Trend signals -------------------------------------------------------
    r = ind["period_return"]
    if r is not None:
        facts.append(
            f"The price moved {fmt_pct(r, True)} between {start} and {end} (closing prices, "
            "dividends excluded)."
        )
        if r > 0.01:
            signals.append(("Positive return over the period", 1))
            positives.append(f"The price rose {fmt_pct(r)} over the selected period.")
        elif r < -0.01:
            signals.append(("Negative return over the period", -1))
            risks.append(f"The price fell {fmt_pct(abs(r))} over the selected period.")
        else:
            signals.append(("Roughly flat return over the period", 0))

    ma20, ma50 = ind["ma20"], ind["ma50"]
    reference, label = (ma50, "50-day") if ma50 is not None else (ma20, "20-day")
    if reference is not None:
        above = price > reference
        facts.append(
            f"The latest close ({fmt_num(price)}) is {'above' if above else 'below'} its "
            f"{label} average ({fmt_num(reference)})."
        )
        signals.append((f"Price vs {label} average", 1 if above else -1))
        if above:
            positives.append(f"The price is trading above its {label} average.")
        else:
            risks.append(f"The price is trading below its {label} average.")
    if ma20 is not None and ma50 is not None:
        signals.append(("20-day vs 50-day average", 1 if ma20 > ma50 else -1))
        facts.append(
            f"The 20-day average ({fmt_num(ma20)}) is "
            f"{'above' if ma20 > ma50 else 'below'} the 50-day average ({fmt_num(ma50)})."
        )

    rsi = ind["rsi14"]
    if rsi is not None:
        facts.append(f"The 14-day RSI is {rsi:.0f}.")
        if rsi >= 70:
            risks.append(
                f"RSI of {rsi:.0f} is in the zone often called overbought, so recent gains may "
                "be stretched."
            )
        elif rsi <= 30:
            risks.append(
                f"RSI of {rsi:.0f} is in the zone often called oversold, which can mean heavy "
                "selling pressure."
            )

    vol = ind["volatility"]
    if vol is not None:
        facts.append(f"Annualized volatility is {fmt_pct(vol)}.")
        if vol > 0.40:
            risks.append(
                f"Annualized volatility of {fmt_pct(vol)} means large price swings have been common."
            )

    dd = ind["max_drawdown"]
    if dd is not None:
        facts.append(f"The largest peak-to-low fall in the period was {fmt_pct(dd)}.")
        if dd <= -0.20:
            risks.append(f"The price fell as much as {fmt_pct(abs(dd))} from a peak inside this period.")

    # --- Fundamentals (only when Yahoo provides them) --------------------------
    eps = info.get("trailingEps")
    if not is_missing(eps) and eps < 0:
        risks.append("Trailing earnings per share is negative, so the company was not profitable over the last 12 months.")
    margin = info.get("profitMargins")
    if not is_missing(margin):
        if margin > 0:
            positives.append(f"The company reports a positive profit margin ({fmt_pct(margin)}).")
        else:
            risks.append(f"The reported profit margin is negative ({fmt_pct(margin)}).")
    growth = info.get("revenueGrowth")
    if not is_missing(growth):
        if growth > 0:
            positives.append(f"Revenue grew {fmt_pct(growth)} year on year in the latest reported quarter.")
        else:
            risks.append(f"Revenue changed {fmt_pct(growth, True)} year on year in the latest reported quarter.")
    div = info.get("trailingAnnualDividendYield")
    if not is_missing(div) and div > 0:
        positives.append(
            f"The company paid dividends (trailing yield {fmt_pct(div)}). Dividends can be cut or stopped."
        )
    high = info.get("fiftyTwoWeekHigh")
    if not is_missing(high) and high > 0:
        facts.append(f"The latest close is {fmt_pct(price / high - 1, True)} versus the 52-week high.")

    if not positives:
        positives.append("The available data shows no clear positive factor.")
    if not risks:
        risks.append("The available data shows no standout risk, but absence of data is not absence of risk.")
    risks.append(
        "Indicators look backward. News, earnings, policy and market-wide moves can change the picture quickly."
    )
    risks.append("Yahoo Finance data is unofficial and can be delayed or wrong.")

    # --- What could change the assessment -------------------------------------
    if reference is not None:
        changes.append(
            f"A sustained move to the other side of the {label} average ({fmt_num(reference)}) "
            "would flip the trend signals."
        )
    changes.append("New quarterly results, guidance changes or other company announcements.")
    changes.append("Sector or market-wide moves, interest-rate news and, for foreign listings, currency changes.")
    changes.append("A different time window. Check at least two periods before drawing a conclusion.")
    if has_news:
        changes.append("Anything in the recent headlines tab that you find credible after reading the source.")

    # --- Evidence summary ----------------------------------------------------
    total = sum(v for _, v in signals)
    if len(signals) < 2:
        label_text = "Mixed"
        reasoning = "Too few indicators could be calculated from this data to lean either way."
    else:
        positive_count = sum(1 for _, v in signals if v > 0)
        negative_count = sum(1 for _, v in signals if v < 0)
        label_text = "Positive" if total >= 2 else "Negative" if total <= -2 else "Mixed"
        reasoning = (
            f"{positive_count} of {len(signals)} trend signals were positive and "
            f"{negative_count} were negative. This summary covers price-trend evidence in the "
            "selected window only. It does not judge business quality or valuation."
        )

    questions = [
        "What does the company earn money from, and how steady is that income?",
        "How does the valuation compare with similar companies and with its own history?",
        "What do the latest earnings report and official filings say about the outlook?",
        "What would make you change your mind about this stock?",
    ]

    return {
        "facts": facts,
        "positives": positives,
        "risks": risks,
        "changes": changes,
        "questions": questions,
        "signals": signals,
        "summary_label": label_text,
        "summary_reasoning": reasoning,
    }


# ----------------------------------------------------------------------------
# Text report for download
# ----------------------------------------------------------------------------

def build_report(ctx, ind, fundamentals, analysis, headlines):
    lines = [
        f"STOCK RESEARCH REPORT: {ctx['name']} ({ctx['ticker']})",
        f"Generated: {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
        "DISCLAIMER: " + DISCLAIMER.replace("**", ""),
        "",
        "DATA",
        f"Sector: {ctx['sector']}",
        f"Latest close: {fmt_num(ctx['price'])} {ctx['currency']} (as of {ctx['end']})",
        f"Period shown: {ctx['start']} to {ctx['end']} ({ctx['period_label']}, {ctx['days']} trading days)",
        "Source: Yahoo Finance via yfinance (unofficial; may be delayed or incomplete)",
        "",
        "INDICATORS",
        f"Period return: {fmt_pct(ind['period_return'], True)}",
        f"20-day average: {fmt_num(ind['ma20'])}",
        f"50-day average: {fmt_num(ind['ma50'])}",
        f"RSI (14-day): {fmt_num(ind['rsi14'], 0)}",
        f"Annualized volatility: {fmt_pct(ind['volatility'])}",
        f"Maximum drawdown: {fmt_pct(ind['max_drawdown'])}",
        "",
        "FUNDAMENTALS",
    ]
    lines += [f"{label}: {value}" for label, value in fundamentals]
    lines += ["", f"EVIDENCE SUMMARY: {analysis['summary_label']}", analysis["summary_reasoning"]]
    for title, key in (
        ("WHAT THE DATA SHOWS", "facts"),
        ("POTENTIAL POSITIVE FACTORS", "positives"),
        ("RISKS AND UNCERTAINTIES", "risks"),
        ("WHAT COULD CHANGE THE ASSESSMENT", "changes"),
        ("QUESTIONS TO INVESTIGATE", "questions"),
    ):
        lines += ["", title] + [f"- {text}" for text in analysis[key]]
    lines += ["", "RECENT HEADLINES"]
    if headlines:
        lines += [f"- {h['date']} | {h['source']} | {h['title']} | {h['url']}" for h in headlines]
    else:
        lines.append("- Could not be fetched from this data source.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# The app
# ----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Stock Research Assistant", page_icon="📈", layout="wide")

    st.title("Stock Research Assistant")
    st.warning(DISCLAIMER)

    # --- Sidebar inputs -----------------------------------------------------
    if "ticker" not in st.session_state:
        st.session_state.ticker = "AAPL"

    def pick_example(symbol):
        st.session_state.ticker = symbol

    with st.sidebar:
        st.header("Choose a stock")
        st.text_input(
            "Ticker symbol",
            key="ticker",
            help="Press Enter to load. Add an exchange suffix for non-US stocks, such as RELIANCE.NS.",
        )
        st.caption("Examples")
        for symbol in EXAMPLES:
            st.button(symbol, on_click=pick_example, args=(symbol,), key=f"ex_{symbol}")
        period_label = st.selectbox("Time period", list(PERIODS), index=2)
        with st.expander("Exchange suffixes"):
            st.markdown(
                "- US stocks: no suffix (`AAPL`)\n"
                "- NSE India: `.NS` (`RELIANCE.NS`)\n"
                "- BSE India: `.BO`\n"
                "- London: `.L`\n"
                "- Toronto: `.TO`\n"
                "- Hong Kong: `.HK`\n"
                "- Tokyo: `.T`\n"
                "- Australia: `.AX`\n"
                "- Germany (Xetra): `.DE`"
            )
        st.caption(
            "Ticker lookups are sent to Yahoo Finance through yfinance. This app does not "
            "store them, though the hosting service may keep standard access logs."
        )

    ticker = clean_ticker(st.session_state.ticker)
    if ticker is None:
        st.error(
            "That ticker does not look valid. Use letters and numbers, plus an optional "
            "suffix such as `.NS` (for example `AAPL` or `RELIANCE.NS`)."
        )
        st.stop()

    # --- Load data ----------------------------------------------------------
    try:
        with st.spinner(f"Loading {ticker}..."):
            hist = load_history(ticker, PERIODS[period_label])
    except Exception as exc:  # network errors, rate limits, library errors
        st.error(
            f"Could not reach the data source ({type(exc).__name__}). Yahoo Finance may be "
            "rate-limiting this app or temporarily down. Wait a minute and press Enter in the "
            "ticker box to try again."
        )
        st.stop()

    if hist is None or hist.empty or "Close" not in hist.columns or hist["Close"].dropna().empty:
        st.error(
            f"No price data came back for **{ticker}**. Check the spelling and the exchange "
            "suffix (for example `RELIANCE.NS`, not `RELIANCE`), or try another symbol. If "
            "symbols that normally work also fail, Yahoo Finance may be rate-limiting this "
            "app, so try again in a few minutes."
        )
        st.stop()

    hist = hist.dropna(subset=["Close"])
    close = hist["Close"]
    info = load_info(ticker)
    currency = get_currency(ticker, info)
    headlines = load_news(ticker)

    price = float(close.iloc[-1])
    start = hist.index[0].date().isoformat()
    end = hist.index[-1].date().isoformat()
    name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector") or "unavailable"

    ind = compute_indicators(close)
    fundamentals = build_fundamentals(info, price, currency)
    analysis = build_analysis(ind, price, info, start, end, bool(headlines))

    # --- Header: company, price, caveats -----------------------------------------
    st.subheader(f"{name} ({ticker})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest close", f"{price:,.2f} {currency}")
    c2.metric("As of", end)
    c3.metric(f"Return ({period_label})", fmt_pct(ind["period_return"], True))
    c4.metric("Sector", sector)
    st.caption(
        f"Showing {start} to {end} ({len(close)} trading days), prices in {currency}. "
        "Source: Yahoo Finance via yfinance, which is unofficial and may be delayed."
    )
    if (dt.date.today() - hist.index[-1].date()).days > 5:
        st.warning(
            f"The latest price is from {end}. The data may be stale, or trading may be halted "
            "or the stock delisted."
        )
    if currency == "unknown":
        st.info("Yahoo did not report a currency for this symbol, so prices are shown without one.")

    # --- Chart --------------------------------------------------------------
    chart = pd.DataFrame({"Close": close})
    chart["20-day average"] = close.rolling(20).mean()
    chart["50-day average"] = close.rolling(50).mean()
    chart.index = chart.index.tz_localize(None) if chart.index.tz is not None else chart.index
    st.line_chart(chart.dropna(axis=1, how="all"))

    # --- Tabs ---------------------------------------------------------------
    tab_analysis, tab_ind, tab_fund, tab_news = st.tabs(
        ["Analysis", "Indicators", "Fundamentals", "Headlines"]
    )

    with tab_analysis:
        st.markdown(f"### Evidence summary: {analysis['summary_label']}")
        st.write(analysis["summary_reasoning"])
        if analysis["signals"]:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Signal": text, "Reading": {1: "Positive", 0: "Neutral", -1: "Negative"}[v]}
                        for text, v in analysis["signals"]
                    ]
                ),
                hide_index=True,
            )
        for title, key in (
            ("What the data shows", "facts"),
            ("Potential positive factors", "positives"),
            ("Risks and uncertainties", "risks"),
            ("What could change the assessment", "changes"),
            ("Questions to investigate yourself", "questions"),
        ):
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- {text}" for text in analysis[key]))
        st.caption("This is a conditional research view built from the numbers above. It is not a recommendation to buy, sell or hold.")

    with tab_ind:
        values = {
            "Period return": fmt_pct(ind["period_return"], True),
            "20-day average": fmt_num(ind["ma20"]),
            "50-day average": fmt_num(ind["ma50"]),
            "RSI (14-day)": fmt_num(ind["rsi14"], 0),
            "Annualized volatility": fmt_pct(ind["volatility"]),
            "Maximum drawdown": fmt_pct(ind["max_drawdown"]),
        }
        for label, value in values.items():
            meaning, limit = INDICATOR_HELP[label]
            st.markdown(f"**{label}: {value}**")
            st.markdown(f"{meaning}  \n*Limitation:* {limit}")

    with tab_fund:
        st.dataframe(pd.DataFrame(fundamentals, columns=["Measure", "Value"]), hide_index=True)
        st.caption("Values come from Yahoo Finance. 'unavailable' means Yahoo did not provide the figure; nothing is estimated.")

    with tab_news:
        if headlines:
            for h in headlines:
                st.markdown(f"- [{h['title']}]({h['url']})  \n  {h['source']}, {h['date']}")
            st.caption("Headlines are shown as provided by the data source and are not analysed or verified here.")
        else:
            st.info("Headlines could not be fetched reliably for this symbol, so none are shown. The rest of the analysis does not depend on them.")

    # --- Downloads ----------------------------------------------------------
    st.divider()
    csv_frame = hist[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in hist.columns]].copy()
    csv_frame["MA20"] = close.rolling(20).mean()
    csv_frame["MA50"] = close.rolling(50).mean()
    csv_frame["RSI14"] = rsi_series(close)
    csv_frame.index = csv_frame.index.strftime("%Y-%m-%d")
    csv_frame.index.name = "Date"

    report = build_report(
        {
            "name": name, "ticker": ticker, "sector": sector, "price": price,
            "currency": currency, "start": start, "end": end,
            "period_label": period_label, "days": len(close),
        },
        ind, fundamentals, analysis, headlines,
    )
    d1, d2 = st.columns(2)
    d1.download_button("Download price data (CSV)", csv_frame.to_csv().encode("utf-8"),
                       file_name=f"{ticker}_{period_label.replace(' ', '')}.csv", mime="text/csv")
    d2.download_button("Download report (text)", report.encode("utf-8"),
                       file_name=f"{ticker}_report.txt", mime="text/plain")

    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
