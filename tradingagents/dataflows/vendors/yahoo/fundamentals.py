from typing import Annotated

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.date_window import (
    get_current_date,
    is_historical,
    withhold_live_profile,
    withhold_undated_statements,
    withhold_undisclosed_trades,
)
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.net import vendor_reachable
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import (
    YAHOO_HOST,
    raise_for_empty,
    yf_retry,
)

# Quote currencies Yahoo gives in a minor unit, and the main unit of each.
_MAIN_UNIT = {"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS"}


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    as_of_date: Annotated[str, "analysis date in YYYY-MM-DD format"] = None
):
    """Get company fundamentals overview from yfinance.

    ``Ticker.info`` is a present-day snapshot with no historical vintage, so a
    past ``as_of_date`` withholds it through the shared point-in-time guard
    (``date_window.withhold_live_profile``, #1300).
    """
    canonical = normalize_symbol(ticker)

    # Guard before the request: the response would only be discarded, and the
    # answer does not depend on it.
    withheld = withhold_live_profile(as_of_date, canonical)
    if withheld:
        return withheld

    info = yf_retry(lambda: yf.Ticker(canonical).info)
    if not info:
        raise_for_empty(ticker, canonical, "fundamentals")

    # Yahoo gives these two in percent (dividendYield 0.41 is 0.41%; debtToEquity
    # 78.4 is 78.4%, a ratio of 0.78) but the margins and returns as fractions,
    # so each carries its unit.
    dividend_yield, debt_to_equity = info.get("dividendYield"), info.get("debtToEquity")
    # Money figures come in three currencies (#1456): price levels in the quote
    # currency, which can be a minor unit (London in pence, GBp); market cap and
    # per-share figures in that currency's main unit (GBP), as Yahoo's own P/E
    # and P/B are computed against them; income figures in the currency the
    # company reports in (an ADR quotes in USD, reports in TWD), or bare when
    # Yahoo does not say which.
    quote = info.get("currency")
    main = _MAIN_UNIT.get(quote, quote)
    reported = info.get("financialCurrency")

    def money(key, currency):
        value = info.get(key)
        return None if value is None else f"{value} {currency}" if currency else value

    fields = [
        ("Name", info.get("longName")),
        ("Sector", info.get("sector")),
        ("Industry", info.get("industry")),
        ("Market Cap", money("marketCap", main)),
        ("PE Ratio (TTM)", info.get("trailingPE")),
        ("Forward PE", info.get("forwardPE")),
        ("PEG Ratio", info.get("pegRatio")),
        ("Price to Book", info.get("priceToBook")),
        ("EPS (TTM)", money("trailingEps", main)),
        ("Forward EPS", money("forwardEps", main)),
        ("Dividend Yield", None if dividend_yield is None else f"{dividend_yield}%"),
        ("Beta", info.get("beta")),
        ("52 Week High", money("fiftyTwoWeekHigh", quote)),
        ("52 Week Low", money("fiftyTwoWeekLow", quote)),
        ("50 Day Average", money("fiftyDayAverage", quote)),
        ("200 Day Average", money("twoHundredDayAverage", quote)),
        ("Revenue (TTM)", money("totalRevenue", reported)),
        ("Gross Profit", money("grossProfits", reported)),
        ("EBITDA", money("ebitda", reported)),
        ("Net Income", money("netIncomeToCommon", reported)),
        ("Profit Margin", info.get("profitMargins")),
        ("Operating Margin", info.get("operatingMargins")),
        ("Return on Equity", info.get("returnOnEquity")),
        ("Return on Assets", info.get("returnOnAssets")),
        ("Debt to Equity", None if debt_to_equity is None
         else f"{debt_to_equity}% ({debt_to_equity / 100:.2f}x)"),
        ("Current Ratio", info.get("currentRatio")),
        ("Book Value", money("bookValue", main)),
        ("Free Cash Flow", money("freeCashflow", reported)),
    ]

    lines = [f"{label}: {v}" for label, v in fields if v is not None]

    # yfinance returns a stub dict (e.g. {"trailingPegRatio": None}) for
    # unknown symbols, so `info` is truthy but every field is empty. Treat
    # "no usable fields" as no data rather than emitting a bare header the
    # agent might fabricate around.
    if not lines:
        raise_for_empty(ticker, canonical, "fundamental fields")

    return f"# Company Fundamentals for {canonical}\n\n" + "\n".join(lines)


def _statement(ticker, freq, as_of_date, title, quarterly_attr, annual_attr) -> str:
    """One financial statement as CSV, for a run dated today."""
    canonical = normalize_symbol(ticker)
    withheld = withhold_undated_statements(as_of_date, canonical, title)
    if withheld:
        return withheld
    what = title.lower()
    attr = quarterly_attr if freq.lower() == "quarterly" else annual_attr
    data = yf_retry(lambda: getattr(yf.Ticker(canonical), attr))
    if data is None or data.empty:
        raise_for_empty(ticker, canonical, f"{what} data")
    return f"# {title} data for {canonical} ({freq})\n" + data.to_csv()


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    as_of_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get balance sheet data from yfinance."""
    return _statement(ticker, freq, as_of_date, "Balance Sheet", "quarterly_balance_sheet", "balance_sheet")


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    as_of_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get cash flow data from yfinance."""
    return _statement(ticker, freq, as_of_date, "Cash Flow", "quarterly_cashflow", "cashflow")


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    as_of_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get income statement data from yfinance."""
    return _statement(ticker, freq, as_of_date, "Income Statement", "quarterly_income_stmt", "income_stmt")


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"],
    as_of_date: Annotated[str | None, "analysis date, yyyy-mm-dd"] = None,
):
    """Get insider transactions data from yfinance, for a run dated today."""
    canonical = normalize_symbol(ticker)
    withheld = withhold_undisclosed_trades(as_of_date, canonical)
    if withheld:
        return withheld
    data = yf_retry(lambda: yf.Ticker(canonical).insider_transactions)

    # Empty is normal here (many valid symbols have no insider filings),
    # so report it plainly rather than treating the symbol as invalid.
    if data is None or data.empty:
        if not vendor_reachable(YAHOO_HOST):
            raise VendorUnavailableError("Yahoo Finance is unreachable; insider filings were not retrieved")
        return f"No insider transactions reported for symbol '{canonical}'"

    return f"# Insider Transactions data for {canonical}\n" + data.to_csv()


def get_company_profile(ticker: str) -> dict:
    """Yahoo's current profile for ``ticker``: name, sector, industry and the like."""
    return yf_retry(lambda: yf.Ticker(normalize_symbol(ticker)).info) or {}


def _earnings_day(value) -> str | None:
    """Calendar day of an earnings timestamp, in the timezone it was labeled."""
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return f"{ts.year:04d}-{ts.month:02d}-{ts.day:02d}"


def _row_get(row, *names):
    """First non-empty column among ``names``, matching case-insensitively."""
    lowered = {str(col).lower(): col for col in row.index}
    for name in names:
        key = lowered.get(name.lower())
        if key is None:
            continue
        val = row[key]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if isinstance(val, str) and val.strip() in ("", "-", "—"):
            continue
        return val
    return None


def get_earnings_context(
    ticker: Annotated[str, "ticker symbol of the company"],
    as_of_date: Annotated[str, "analysis date in YYYY-MM-DD format"] = None,
) -> str:
    """Next earnings date and recent EPS surprises, as known on ``as_of_date``.

    Surprise rows are those whose earnings date is on or before the run date.
    The next date is the earliest later date in the same series, and only when
    that series already has a row on or before the run date — otherwise a
    historical run would read today's still-upcoming print as if it were next.
    """
    canonical = normalize_symbol(ticker)
    as_of_date = as_of_date or get_current_date()
    frame = yf_retry(lambda: yf.Ticker(canonical).get_earnings_dates(limit=20))
    if frame is None or getattr(frame, "empty", True):
        raise_for_empty(ticker, canonical, "earnings dates")

    rows: list[tuple[str, object]] = []
    for idx, row in frame.iterrows():
        day = _earnings_day(idx)
        if day:
            rows.append((day, row))
    if not rows:
        raise_for_empty(ticker, canonical, "earnings dates")

    history = [(day, row) for day, row in rows if day <= as_of_date]
    history.sort(key=lambda item: item[0], reverse=True)
    history = history[:4]
    upcoming = sorted(day for day, _row in rows if day > as_of_date)
    next_date = upcoming[0] if upcoming and (history or not is_historical(as_of_date)) else None

    lines = [
        f"# Earnings context for {canonical}",
        f"# Point-in-time as of: {as_of_date}",
        "",
        f"Next earnings date: {next_date}" if next_date
        else f"Next earnings date: not available as of {as_of_date}",
    ]
    if history:
        lines.append("")
        lines.append("Recent earnings surprises (most recent first):")
        for day, row in history:
            reported = _row_get(row, "Reported EPS", "epsactual", "reported eps")
            estimate = _row_get(row, "EPS Estimate", "epsestimate")
            surprise = _row_get(row, "Surprise(%)", "epssurprisepct", "surprise(%)")
            bits = []
            if reported is not None:
                bits.append(f"reported EPS {reported}")
            if estimate is not None:
                bits.append(f"estimate {estimate}")
            if surprise is not None:
                try:
                    bits.append(f"surprise {float(surprise):+.1f}%")
                except (TypeError, ValueError):
                    bits.append(f"surprise {surprise}")
            lines.append(f"{day}: {'; '.join(bits)}" if bits else day)
    return "\n".join(lines) + "\n"
