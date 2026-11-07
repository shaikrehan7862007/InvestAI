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
SAMPLE_TEN = "AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, XOM, JNJ, RELIANCE.NS, TCS.NS"
MAX_TICKERS = 20  # most stocks analysed in one go, to keep requests to Yahoo reasonable

BUY, HOLD, AVOID = "Potential buy candidate", "Hold/watch", "Avoid for now"

# Ranking criteria: (key, label, maximum points, how points are earned).
# A criterion with no data earns 0 points, the same as the weakest reading,
# so missing information can never lift a stock's score or rank.
CRITERIA = [
    ("return", "Return over the period", 1,
     "1 point if the closing price rose over the selected period."),
    ("trend", "Price vs trend average", 1,
     "1 point if the latest close is above its 50-day average (its 20-day average when "
     "there are fewer than 50 trading days)."),
    ("cross", "20-day vs 50-day average", 1,
     "1 point if the 20-day average is above the 50-day average. Needs 50 trading days."),
    ("volatility", "Annualized volatility", 2,
     "2 points below 25%, 1 point from 25% to 40%, 0 points at 40% or more."),
    ("drawdown", "Maximum drawdown", 2,
     "2 points if the deepest fall from a peak was less than 15%, 1 point if less than 30%, "
     "otherwise 0."),
    ("margin", "Profit margin", 2,
     "2 points above 10%, 1 point above 0%, 0 points at or below 0%."),
    ("growth", "Revenue growth (year on year)", 2,
     "2 points above 10%, 1 point above 0%, 0 points at or below 0%."),
]
TOTAL_POINTS = sum(c[2] for c in CRITERIA)

RESEARCH_NOTE = (
    "These are research suggestions based on the data fetched for the chosen period. They are "
    "not guarantees or predictions and not personalised financial advice. The app does not "
    "place trades and does not tell you what you should buy."
)
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


NUMERIC_INFO_KEYS = (
    "marketCap", "trailingPE", "forwardPE", "totalRevenue", "revenueGrowth", "profitMargins",
    "trailingAnnualDividendYield", "dividendRate", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "trailingEps",
)


def to_number(x):
    """A finite float, or None. Yahoo sometimes sends text such as 'Infinity' instead of a number."""
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker):
    # Errors are not cached by Streamlit, so a temporary failure is retried next time.
    info = dict(yf.Ticker(ticker).info or {})
    for key in NUMERIC_INFO_KEYS:
        if key in info:
            info[key] = to_number(info[key])
    return info


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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_currency(ticker):
    return yf.Ticker(ticker).fast_info.get("currency")


def get_currency(ticker, info):
    """Currency code from company info, falling back to fast_info."""
    code = info.get("currency")
    if not code:
        try:
            code = fetch_currency(ticker)
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
    specific_risks = list(risks)  # the data-driven risks, before the general ones below
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
        "risks_specific": specific_risks,
        "changes": changes,
        "questions": questions,
        "signals": signals,
        "summary_label": label_text,
        "summary_reasoning": reasoning,
    }


# ----------------------------------------------------------------------------
# Research suggestion, score and ranking (rule based, uses only fetched data)
# ----------------------------------------------------------------------------

def score_stock(ind, price, info, days, stale):
    """Score one stock against CRITERIA and turn the score into a research suggestion.

    A criterion with no data earns 0 points, so missing information never helps.
    """
    ma20, ma50 = ind["ma20"], ind["ma50"]
    ref, ref_name = (ma50, "50-day") if ma50 is not None else (ma20, "20-day")
    r, vol, dd = ind["period_return"], ind["volatility"], ind["max_drawdown"]
    margin, growth = info.get("profitMargins"), info.get("revenueGrowth")

    # Points per criterion. None means the data is unavailable.
    earned = {
        "return": None if r is None else int(r > 0),
        "trend": None if ref is None else int(price > ref),
        "cross": None if ma20 is None or ma50 is None else int(ma20 > ma50),
        "volatility": None if vol is None else (2 if vol < 0.25 else 1 if vol < 0.40 else 0),
        "drawdown": None if dd is None else (2 if dd > -0.15 else 1 if dd > -0.30 else 0),
        "margin": None if is_missing(margin) else (2 if margin > 0.10 else 1 if margin > 0 else 0),
        "growth": None if is_missing(growth) else (2 if growth > 0.10 else 1 if growth > 0 else 0),
    }
    readings = {
        "return": fmt_pct(r, True),
        "trend": f"{fmt_num(price)} vs {ref_name} average {fmt_num(ref)}",
        "cross": f"{fmt_num(ma20)} vs {fmt_num(ma50)}",
        "volatility": fmt_pct(vol),
        "drawdown": fmt_pct(dd),
        "margin": fmt_pct(margin),
        "growth": fmt_pct(growth, True),
    }
    phrases = {  # what to say when a criterion earns full points
        "return": f"the price rose {fmt_pct(r)} over the period",
        "trend": f"it trades above its {ref_name} average",
        "cross": "its 20-day average is above its 50-day average",
        "volatility": f"volatility is relatively low ({fmt_pct(vol)} a year)",
        "drawdown": f"the deepest fall from a peak was shallow ({fmt_pct(dd)})",
        "margin": f"the profit margin is {fmt_pct(margin)}",
        "growth": f"revenue grew {fmt_pct(growth)} year on year",
    }

    rows, missing, strengths = [], [], []
    for key, label, max_points, _rule in CRITERIA:
        got = earned[key]
        if got is None:
            rows.append({"Criterion": label, "Reading": "unavailable", "Points": f"0 of {max_points}, data missing"})
            missing.append(f"{label} is unavailable, so it earns 0 of {max_points} points.")
        else:
            rows.append({"Criterion": label, "Reading": readings[key], "Points": f"{got} of {max_points}"})
            if got == max_points:
                strengths.append(phrases[key])
    if not info:
        missing.append("Yahoo returned no company details, so fundamentals are unavailable.")

    points = sum(v for v in earned.values() if v is not None)
    coverage = sum(1 for v in earned.values() if v is not None)
    available_max = sum(c[2] for c in CRITERIA if earned[c[0]] is not None)
    share = points / available_max if available_max else None

    # --- Suggestion ------------------------------------------------------------
    if coverage < 4:
        label = HOLD
        why = "Too little data is available to lean either way, so this is a watch rather than a call."
    elif points >= 8 and coverage >= 5 and earned["trend"] == 1:
        label = BUY
        why = (
            f"It scored {points} of {TOTAL_POINTS} points, trades above its {ref_name} average, "
            f"and {coverage} of {len(CRITERIA)} criteria had data."
        )
    elif share is not None and share <= 0.35:
        label = AVOID
        why = (
            f"It earned only {points} of the {available_max} points available from the data we "
            f"have ({share * 100:.0f}%)."
        )
    else:
        label = HOLD
        if points >= 8:
            gaps = []
            if earned["trend"] != 1:
                gaps.append("the price is not confirmed above its trend average")
            if coverage < 5:
                gaps.append("fewer than 5 criteria had data")
            why = "The score is high, but " + " and ".join(gaps) + ", so it does not qualify as a buy candidate."
        elif share >= 0.7:
            why = (
                f"The data we have looks strong ({points} of {available_max} available points), "
                f"but {len(CRITERIA) - coverage} criteria had no data and earn nothing, so it "
                f"cannot reach the {BUY} bar of 8 points."
            )
        else:
            why = (
                f"The evidence is mixed: it earned {points} of the {available_max} points "
                f"available from the data we have ({share * 100:.0f}%)."
            )

    # --- Confidence ------------------------------------------------------------
    clear = share is not None and (share >= 0.7 or share <= 0.3)
    reasons = [
        f"{coverage} of {len(CRITERIA)} criteria had data.",
        f"{days} trading days of price history were used.",
    ]
    if stale:
        reasons.append("The latest price is more than 5 days old.")
    if share is not None:
        reasons.append("The signals mostly agree." if clear else "The signals are mixed, which limits confidence.")
    if days < 60:
        reasons.append("Windows shorter than about 60 trading days (such as 1 month) give noisy signals.")
    if stale or coverage < 4 or days < 60:
        confidence = "Low"
    elif coverage >= 6 and days >= 120 and clear:
        confidence = "High"
    else:
        confidence = "Medium"

    return {
        "label": label, "why": why, "points": points, "available_max": available_max,
        "coverage": coverage, "share": share, "confidence": confidence,
        "confidence_reasons": reasons, "rows": rows, "missing": missing,
        "strengths": strengths,
    }


def rank_results(results):
    """Highest score first. Ties go to the stock with more data, then alphabetical order."""
    ranked = sorted(
        results, key=lambda x: (-x["suggestion"]["points"], -x["suggestion"]["coverage"], x["ticker"])
    )
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position
    return ranked


def build_comparison_frame(ranked):
    rows = []
    for r in ranked:
        s, ind, info = r["suggestion"], r["ind"], r["info"]
        rows.append({
            "Rank": r["rank"],
            "Ticker": r["ticker"],
            "Name": r["name"],
            "Research suggestion": s["label"],
            f"Score (max {TOTAL_POINTS})": s["points"],
            "Confidence": s["confidence"],
            "Data coverage": f"{s['coverage']} of {len(CRITERIA)}",
            "Latest close": f"{fmt_num(r['price'])} {r['currency']}",
            "Period return": fmt_pct(ind["period_return"], True),
            "Volatility": fmt_pct(ind["volatility"]),
            "Max drawdown": fmt_pct(ind["max_drawdown"]),
            "P/E (trailing)": fmt_num(info.get("trailingPE")),
            "Profit margin": fmt_pct(info.get("profitMargins")),
            "Revenue growth": fmt_pct(info.get("revenueGrowth"), True),
        })
    return pd.DataFrame(rows)


def build_overall_summary(ranked, failed_count=0):
    """Markdown lines describing the strongest research candidates and why."""
    candidates = [r for r in ranked if r["suggestion"]["label"] == BUY]
    counts = {label: sum(1 for r in ranked if r["suggestion"]["label"] == label) for label in (BUY, HOLD, AVOID)}

    def describe(r):
        s = r["suggestion"]
        shown = s["strengths"][:4]
        strengths = "; ".join(shown) if shown else "no criterion earned full points"
        if len(s["strengths"]) > len(shown):
            strengths += f"; plus {len(s['strengths']) - len(shown)} more"
        line = (
            f"- **#{r['rank']} {r['ticker']}** ({r['name']}): {s['points']} of {TOTAL_POINTS} "
            f"points, {s['confidence'].lower()} confidence. Strengths: {strengths}. "
            f"Watch-out: {r['analysis']['risks_specific'][0]}"
        )
        if s["missing"]:
            line += f" Data gaps: {len(s['missing'])}."
        return line

    if candidates:
        lines = [
            f"**{len(candidates)} of {len(ranked)} stocks met the criteria for \"{BUY}\".** "
            "The strongest, in rank order:"
        ]
        lines += [describe(r) for r in candidates[:3]]
    else:
        lines = [
            f"**No stock met the criteria for \"{BUY}\".** The highest-scoring stocks are "
            "listed for further research, but none has strong enough evidence in this data:"
        ]
        lines += [describe(r) for r in ranked[:3]]
    tally = f"Overall: {counts[BUY]} {BUY}, {counts[HOLD]} {HOLD}, {counts[AVOID]} {AVOID}"
    if failed_count:
        tally += f", {failed_count} could not be analysed"
    lines.append(tally + ". Ranks compare only the stocks you entered.")
    return lines


def suggestion_lines(r):
    """Plain-text version of the suggestion block, used in downloadable reports."""
    s = r["suggestion"]
    lines = [
        f"RESEARCH SUGGESTION: {s['label']} (score {s['points']} of {TOTAL_POINTS}, confidence {s['confidence']})",
        s["why"],
        "",
        "Evidence by criterion:",
    ]
    lines += [f"- {row['Criterion']}: {row['Reading']} ({row['Points']})" for row in s["rows"]]
    lines += ["", "Risks:"] + [f"- {t}" for t in r["analysis"]["risks_specific"]]
    lines += ["", "Missing or limited data:"]
    lines += [f"- {t}" for t in s["missing"]] or ["- Every scored criterion had data."]
    lines += ["", f"Confidence: {s['confidence']}"] + [f"- {t}" for t in s["confidence_reasons"]]
    lines += ["", RESEARCH_NOTE]
    return lines


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
    if "suggestion" in ctx:
        lines += [""] + suggestion_lines(ctx)
    lines += ["", f"TREND EVIDENCE SUMMARY: {analysis['summary_label']}", analysis["summary_reasoning"]]
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
    elif headlines is None:
        lines.append("- Not loaded in the multi-stock report. Open the stock on its own to see them.")
    else:
        lines.append("- Could not be fetched from this data source.")
    return "\n".join(lines)


def build_multi_report(period_label, ranked, failed, summary_lines):
    lines = [
        "MULTI-STOCK RESEARCH REPORT",
        f"Generated: {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC",
        f"Period: {period_label}",
        "",
        "DISCLAIMER: " + DISCLAIMER.replace("**", ""),
        RESEARCH_NOTE,
        "",
        "SUMMARY",
    ]
    lines += [line.replace("**", "") for line in summary_lines]
    lines += ["", f"RANKING (score out of {TOTAL_POINTS}; a criterion with missing data earns 0 points)"]
    for r in ranked:
        s = r["suggestion"]
        lines.append(
            f"{r['rank']}. {r['ticker']} | {s['label']} | {s['points']} of {TOTAL_POINTS} | "
            f"confidence {s['confidence']} | data for {s['coverage']} of {len(CRITERIA)} criteria"
        )
    if failed:
        lines += ["", "COULD NOT BE ANALYSED"] + [f"- {f['ticker']}: {f['short_error']}" for f in failed]
    for r in ranked:
        lines += ["", "=" * 70, build_report(r, r["ind"], r["fundamentals"], r["analysis"], None)]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# The app
# ----------------------------------------------------------------------------

def parse_tickers(raw):
    """Split the input box into (valid tickers, invalid entries, valid tickers over the limit)."""
    valid, invalid, extra = [], [], []
    for entry in re.split(r"[,\s;]+", raw or ""):
        if not entry:
            continue
        ticker = clean_ticker(entry)
        if ticker is None:
            shown = entry[:20].replace("`", "")
            if shown not in invalid:
                invalid.append(shown)
        elif ticker not in valid and ticker not in extra:
            (valid if len(valid) < MAX_TICKERS else extra).append(ticker)
    return valid, invalid, extra


def analyze_stock(ticker, period_label):
    """Load and analyse one stock. Never raises: problems come back as ok=False with a message."""
    try:
        hist = load_history(ticker, PERIODS[period_label])
    except Exception as exc:  # network errors, rate limits, library errors
        return {
            "ok": False, "ticker": ticker,
            "short_error": f"data source error ({type(exc).__name__})",
            "error": (
                f"Could not reach the data source for {ticker} ({type(exc).__name__}). Yahoo "
                "Finance may be rate-limiting this app or temporarily down. Wait a minute and "
                "try again."
            ),
        }

    if hist is None or hist.empty or "Close" not in hist.columns or hist["Close"].dropna().empty:
        return {
            "ok": False, "ticker": ticker,
            "short_error": "no price data came back (check the spelling and exchange suffix)",
            "error": (
                f"No price data came back for **{ticker}**. Check the spelling and the exchange "
                "suffix (for example `RELIANCE.NS`, not `RELIANCE`), or try another symbol. If "
                "symbols that normally work also fail, Yahoo Finance may be rate-limiting this "
                "app, so try again in a few minutes."
            ),
        }

    try:
        hist = hist.dropna(subset=["Close"])
        close = hist["Close"]
        info = load_info(ticker)
        currency = get_currency(ticker, info)
        price = float(close.iloc[-1])
        start = hist.index[0].date().isoformat()
        end = hist.index[-1].date().isoformat()
        stale = (dt.date.today() - hist.index[-1].date()).days > 5

        ind = compute_indicators(close)
        suggestion = score_stock(ind, price, info, len(close), stale)
        if currency == "unknown":
            suggestion["missing"].append("Yahoo did not report a currency, so prices are shown without one.")
        if stale:
            suggestion["missing"].append(f"The latest price is from {end}, more than 5 days old.")
        return {
            "ok": True, "ticker": ticker, "period_label": period_label,
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector") or "unavailable",
            "currency": currency, "price": price, "start": start, "end": end,
            "days": len(close), "stale": stale, "hist": hist, "close": close, "info": info,
            "ind": ind,
            "fundamentals": build_fundamentals(info, price, currency),
            "analysis": build_analysis(ind, price, info, start, end, False),
            "suggestion": suggestion,
        }
    except Exception as exc:  # unexpected data shapes for this one symbol
        return {
            "ok": False, "ticker": ticker,
            "short_error": f"unexpected data problem ({type(exc).__name__})",
            "error": (
                f"The data for {ticker} could not be analysed ({type(exc).__name__}). Try another "
                "period or symbol."
            ),
        }


def render_suggestion(r):
    """Research suggestion with its evidence, risks, missing data and confidence."""
    s = r["suggestion"]
    colour = {BUY: "green", HOLD: "orange", AVOID: "red"}[s["label"]]
    st.markdown(
        f"**Research suggestion: :{colour}[{s['label']}]** · score {s['points']} of "
        f"{TOTAL_POINTS} · confidence {s['confidence']}"
    )
    st.write(s["why"])
    st.markdown("**Evidence**")
    st.dataframe(pd.DataFrame(s["rows"]), hide_index=True)
    st.markdown("**Risks**")
    st.markdown("\n".join(f"- {t}" for t in r["analysis"]["risks_specific"]))
    st.markdown("**Missing or limited data**")
    st.markdown("\n".join(f"- {t}" for t in s["missing"]) or "- Every scored criterion had data.")
    st.markdown(f"**Confidence: {s['confidence']}**")
    st.markdown("\n".join(f"- {t}" for t in s["confidence_reasons"]))
    st.caption("Not scored: valuation (P/E), dividends, headlines and sector context. " + RESEARCH_NOTE)


def render_method():
    st.markdown(
        f"**Score.** Each stock is scored out of {TOTAL_POINTS} points on {len(CRITERIA)} criteria, "
        "using only the data fetched for the chosen period. **A criterion with missing data earns "
        "0 points**, the same as the weakest reading, so missing information never helps a stock."
    )
    st.dataframe(
        pd.DataFrame(
            [{"Criterion": label, "Max points": m, "How points are earned": rule} for _, label, m, rule in CRITERIA]
        ),
        hide_index=True,
    )
    st.markdown(
        "**Ranking.** Highest score first. Ties go to the stock with data for more criteria, then "
        "to alphabetical ticker order, which has no analytical meaning.\n\n"
        "**Suggestion**\n"
        f"- *{BUY}*: at least 8 of {TOTAL_POINTS} points, the price above its trend average, and "
        "data for at least 5 criteria.\n"
        f"- *{AVOID}*: data for at least 4 criteria, and 35% or less of the points available "
        "from that data. Missing data alone never produces this label.\n"
        f"- *{HOLD}*: everything else, including stocks with data for fewer than 4 criteria.\n\n"
        "**Confidence.** *Low* if the latest price is more than 5 days old, fewer than 4 criteria "
        "have data, or there are fewer than 60 trading days. *High* if at least 6 criteria have "
        "data, there are at least 120 trading days, and the signals clearly agree (70% or more, or "
        "30% or less, of the available points). *Medium* otherwise.\n\n"
        "**Not scored:** valuation (P/E), dividends, headlines and sector or industry context. The "
        "thresholds are rules of thumb, not industry standards, and ranks compare only the stocks "
        "you entered."
    )


def render_detail(r, include_suggestion=True):
    """Chart, indicators, fundamentals, headlines and downloads for one stock."""
    ticker, name, currency = r["ticker"], r["name"], r["currency"]
    price, start, end = r["price"], r["start"], r["end"]
    hist, close, info, ind = r["hist"], r["close"], r["info"], r["ind"]
    period_label, fundamentals = r["period_label"], r["fundamentals"]

    headlines = load_news(ticker)
    analysis = build_analysis(ind, price, info, start, end, bool(headlines))

    # --- Header: company, price, caveats -----------------------------------------
    st.subheader(f"{name} ({ticker})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest close", f"{price:,.2f} {currency}")
    c2.metric("As of", end)
    c3.metric(f"Return ({period_label})", fmt_pct(ind["period_return"], True))
    c4.metric("Sector", r["sector"])
    st.caption(
        f"Showing {start} to {end} ({len(close)} trading days), prices in {currency}. "
        "Source: Yahoo Finance via yfinance, which is unofficial and may be delayed."
    )
    if r["stale"]:
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
        if include_suggestion:
            render_suggestion(r)
            st.divider()
        st.markdown(f"### Trend evidence summary: {analysis['summary_label']}")
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
        st.caption(RESEARCH_NOTE)

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

    report = build_report(r, ind, fundamentals, analysis, headlines)
    d1, d2 = st.columns(2)
    d1.download_button("Download price data (CSV)", csv_frame.to_csv().encode("utf-8"),
                       file_name=f"{ticker}_{period_label.replace(' ', '')}.csv", mime="text/csv",
                       key=f"csv_{ticker}")
    d2.download_button("Download report (text)", report.encode("utf-8"),
                       file_name=f"{ticker}_report.txt", mime="text/plain", key=f"txt_{ticker}")


def render_multi(ok, failed, period_label):
    """Summary, ranking table and per-stock explanations for several stocks."""
    if failed:
        st.warning(
            "These tickers could not be analysed and are left out of the ranking. The others "
            "were analysed normally:\n\n"
            + "\n".join(f"- **{f['ticker']}**: {f['short_error']}" for f in failed)
            + "\n\nCheck the spelling and exchange suffix. If valid symbols fail, Yahoo may be "
            "rate-limiting this app, so try again in a few minutes."
        )
    if not ok:
        st.error("None of the tickers could be analysed, so there is nothing to rank.")
        st.stop()

    ranked = rank_results(ok)
    summary = build_overall_summary(ranked, len(failed))

    st.subheader("Strongest research candidates")
    for line in summary:
        st.markdown(line)
    st.caption(RESEARCH_NOTE)

    st.subheader("Comparison and ranking")
    frame = build_comparison_frame(ranked)
    st.dataframe(frame, hide_index=True)
    st.caption(
        f"Ranked by score out of {TOTAL_POINTS}. Missing data earns 0 points. Prices are in each "
        "stock's own currency, so compare the scores and percentages, not the prices."
    )
    with st.expander("How the score, suggestion and confidence are worked out"):
        render_method()

    d1, d2 = st.columns(2)
    d1.download_button("Download comparison (CSV)", frame.to_csv(index=False).encode("utf-8"),
                       file_name=f"comparison_{period_label.replace(' ', '')}.csv", mime="text/csv",
                       key="cmp_csv")
    d2.download_button("Download all reports (text)",
                       build_multi_report(period_label, ranked, failed, summary).encode("utf-8"),
                       file_name="multi_stock_report.txt", mime="text/plain", key="cmp_txt")

    st.subheader("Suggestion for each stock")
    for r in ranked:
        s = r["suggestion"]
        title = f"#{r['rank']} {r['ticker']} · {r['name']} · {s['label']} · {s['confidence']} confidence"
        with st.expander(title):
            render_suggestion(r)

    st.subheader("Chart, indicators, fundamentals and headlines")
    names = {r["ticker"]: r["name"] for r in ranked}
    choice = st.selectbox(
        "Show full detail for", [r["ticker"] for r in ranked],
        format_func=lambda t: f"{t} ({names[t]})",
    )
    render_detail(next(r for r in ranked if r["ticker"] == choice), include_suggestion=False)


def main():
    st.set_page_config(page_title="Stock Research Assistant", page_icon="📈", layout="wide")

    st.title("Stock Research Assistant")
    st.warning(DISCLAIMER)

    # --- Sidebar inputs -----------------------------------------------------
    if "tickers" not in st.session_state:
        st.session_state.tickers = "AAPL"

    def pick_example(text):
        st.session_state.tickers = text

    with st.sidebar:
        st.header("Choose stocks")
        st.text_area(
            f"Ticker symbols (1 to {MAX_TICKERS})",
            key="tickers",
            height=110,
            help="Separate symbols with commas, spaces or new lines. Press Ctrl+Enter or click "
                 "outside the box to load. Add an exchange suffix for non-US stocks, such as "
                 "RELIANCE.NS. Several symbols give a ranked comparison.",
        )
        st.caption("Examples")
        for symbol in EXAMPLES:
            st.button(symbol, on_click=pick_example, args=(symbol,), key=f"ex_{symbol}")
        st.button("10-stock sample", on_click=pick_example, args=(SAMPLE_TEN,), key="ex_ten")
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

    valid, invalid, extra = parse_tickers(st.session_state.tickers)
    if invalid:
        st.warning(
            "Skipped entries that do not look like ticker symbols: "
            + ", ".join(f"`{e}`" for e in invalid[:10])
            + ". Use letters and numbers, plus an optional suffix such as `.NS`."
        )
    if extra:
        st.warning(f"Only the first {MAX_TICKERS} tickers are analysed. Skipped: {', '.join(extra)}.")
    if not valid:
        st.error("Enter at least one valid ticker, for example `AAPL` or `RELIANCE.NS`.")
        st.stop()

    # --- Load and analyse each stock separately, so one failure never stops the rest ----
    results = []
    progress = st.progress(0.0, text="Loading data...")
    for i, ticker in enumerate(valid, start=1):
        progress.progress((i - 1) / len(valid), text=f"Loading {ticker} ({i} of {len(valid)})...")
        results.append(analyze_stock(ticker, period_label))
    progress.empty()

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    if len(valid) == 1:
        if failed:
            st.error(failed[0]["error"])
            st.stop()
        render_detail(ok[0], include_suggestion=True)
    else:
        render_multi(ok, failed, period_label)

    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
