"""Company statements as they were filed, from SEC EDGAR.

Every other fundamentals vendor serves a period's current value and cuts the
statement at the fiscal period end. That is two claims a run should not make: a
period that has ended is not public until the company files, weeks later, and a
figure that was later restated is not what investors saw at the time.

EDGAR reports every fact with the date it was filed, so a run dated ``as_of_date``
serves exactly what was on file by then, restatements included at the vintage
that was current: Apple's 2008 total assets read 39.6B until the 2010 amendment
restated them to 36.2B.

Access needs no key or account, only a User-Agent identifying the caller, which
SEC requires and refuses requests without. US filers only: anything absent from
EDGAR's ticker map falls through to the next configured vendor.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from tradingagents import __version__
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.files import replace_file

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# A filing history only changes when something new is filed, so one fetch per
# company per day serves every date a run asks about.
_CACHE_TTL_SECONDS = 24 * 60 * 60

# Line items, each with the tags filers use for it, best first. First match wins
# and values are never summed across tags: a company reporting revenue under two
# tags would otherwise be counted twice.
_STATEMENTS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "balance_sheet": [
        ("Total Assets", ("Assets",)),
        ("Current Assets", ("AssetsCurrent",)),
        ("Cash and Equivalents", ("CashAndCashEquivalentsAtCarryingValue",)),
        ("Short-Term Investments", ("ShortTermInvestments",
                                    "MarketableSecuritiesCurrent",
                                    "AvailableForSaleSecuritiesCurrent",
                                    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
                                    "DebtSecuritiesCurrent",
                                    "OtherShortTermInvestments")),
        ("Total Liabilities", ("Liabilities",)),
        ("Current Liabilities", ("LiabilitiesCurrent",)),
        ("Stockholders Equity", ("StockholdersEquity",
                                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    ],
    "income_statement": [
        ("Revenue", ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                     "SalesRevenueNet")),
        ("Cost of Revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
        ("Gross Profit", ("GrossProfit",)),
        ("Operating Income", ("OperatingIncomeLoss",)),
        ("Net Income", ("NetIncomeLoss",)),
        ("Diluted EPS", ("EarningsPerShareDiluted",)),
    ],
    "cashflow": [
        ("Operating Cash Flow", ("NetCashProvidedByUsedInOperatingActivities",
                                 "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
        ("Investing Cash Flow", ("NetCashProvidedByUsedInInvestingActivities",)),
        ("Financing Cash Flow", ("NetCashProvidedByUsedInFinancingActivities",)),
        ("Capital Expenditure", ("PaymentsToAcquirePropertyPlantAndEquipment",
                                 "PaymentsToAcquireProductiveAssets")),
    ],
}

# A statement's figures cover a span: a quarter is about 90 days, a year about
# 365. One filing reports both the quarter and the year to date under the same
# end date, so a match on the end date alone can report half a year as a quarter.
_SPANS = {"quarterly": (60, 115), "annual": (300, 400)}

# A 10-Q's cash flows are often filed only year to date. A quarterly table takes
# the quarter where filed, else the span to date, named in the column; it never
# subtracts one filing from another, which would give a figure no filing states.
_YEAR_TO_DATE = ((150, 200, 6), (240, 290, 9))

# A fiscal year is a period an annual report covers. A 10-Q balance has no span
# to reject, and some filers' 10-Qs report twelve-month totals that pass the span
# check, so either would read as a fiscal year. The value is still the latest
# filing of any form: a recast after a split or spin-off counts from its filing.
_ANNUAL_FORMS = ("10-K", "20-F", "40-F")

# A 10-Q table back to 2018 is ~28 columns, oldest first, and the model then
# misreads which period is which. Keep a short, latest-first window.
_RECENT_QUARTERS = 8
_RECENT_YEARS = 4
# ~15 months: a live run whose latest us-gaap period is older than this has
# nothing recent to offer, so the router should try the next vendor.
_STALE_AFTER = timedelta(days=456)
_BLANK = "not filed"
_YOY_ROWS = ("Revenue", "Net Income")
_COMBINED_CASH_TAGS = ("CashCashEquivalentsAndShortTermInvestments",)


def _user_agent() -> str:
    """Who SEC sees. No account or key exists; callers identify themselves.

    www.sec.gov, which serves the ticker map, refuses a User-Agent carrying no
    contact address: a client name alone or with a project URL gets 403, one
    with an address gets 200. So the default carries a placeholder address and
    the package version. Set SEC_EDGAR_USER_AGENT to your own name and address
    so SEC can reach you about your traffic rather than the project.
    """
    configured = os.getenv("SEC_EDGAR_USER_AGENT", "").strip()
    return configured or f"TradingAgents/{__version__} (contact@example.com)"




def _fetch_json(url: str) -> dict:
    """Read a public EDGAR document, respecting SEC's identification rule."""
    try:
        response = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # Every failure here is "this vendor cannot serve it now", so the router
        # moves on instead of seeing a transport exception it has no rule for.
        raise VendorUnavailableError(f"SEC EDGAR request failed ({status or type(exc).__name__})") from exc
    except ValueError as exc:
        raise VendorUnavailableError("SEC EDGAR returned an unreadable response") from exc


def _cached_json(url: str, name: str) -> dict:
    path = Path(get_config()["data_cache_dir"]) / "sec_edgar" / name
    if path.exists() and time.time() - path.stat().st_mtime < _CACHE_TTL_SECONDS:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass  # a truncated file is a miss, not a failure
    data = _fetch_json(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_file(path, lambda temp: Path(temp).write_text(json.dumps(data), encoding="utf-8"))
    return data


def cik_for(ticker: str) -> str | None:
    """The filer's CIK, or None when the ticker is not a US filer."""
    table = _cached_json(_TICKERS_URL, "company_tickers.json")
    wanted = ticker.strip().upper()
    for entry in table.values():
        if entry.get("ticker", "").upper() == wanted:
            return f"{int(entry['cik_str']):010d}"
    return None


def _span_index(fact: dict, spans: tuple[tuple[int, int], ...]) -> int | None:
    """Which of ``spans`` a duration fact covers (0 for an instant fact), or None."""
    if "start" not in fact:
        return 0
    days = (date.fromisoformat(fact["end"]) - date.fromisoformat(fact["start"])).days
    return next((i for i, (low, high) in enumerate(spans) if low <= days <= high), None)


def _as_of(facts: dict, tags: tuple[str, ...], as_of_date: str, spans: tuple[tuple[int, int], ...],
           forms: tuple[str, ...] = ()) -> tuple[dict, str, dict]:
    """({(period end, span index): value}, unit, meta) for the tags the filer reports.

    A period reported more than once takes its latest filing on or before the
    date, so an amendment counts from the day it was filed and not before. The
    unit comes from the latest period's filing: most lines are USD, earnings
    per share are USD/shares, and scaling those alike would print a real figure
    as zero. A duration fact (revenue, cash flow) must cover one of ``spans``;
    an instant fact (a balance) has no span and serves any.
    """
    values: dict[tuple[str, int], float] = {}
    units: dict[tuple[str, int], str] = {}
    meta: dict[tuple[str, int], dict] = {}
    # Tags are tried in order and a period keeps the first one that reports it:
    # filers renamed lines over the years, so one tag covers only part of the
    # history. Values are never added across tags, which would double count.
    for tag in tags:
        for unit, unit_values in ((facts.get(tag) or {}).get("units", {})).items():
            latest: dict[tuple[str, int], dict] = {}
            covered: set[tuple[str, int]] = set()   # periods a filing of ``forms`` reports
            for fact in unit_values:
                index = _span_index(fact, spans)
                key = (fact["end"], index)
                if fact["filed"] > as_of_date or index is None or key in values:
                    continue
                if not forms or fact.get("form", "").startswith(forms):
                    covered.add(key)
                seen = latest.get(key)
                if seen is None or fact["filed"] >= seen["filed"]:
                    latest[key] = fact
            latest = {key: fact for key, fact in latest.items() if key in covered}
            if latest:
                for key, fact in latest.items():
                    values[key] = fact["val"]
                    units[key] = unit
                    meta[key] = {"fy": fact.get("fy"), "fp": fact.get("fp"),
                                 "form": fact.get("form") or ""}
    chosen_unit = units[max(units)] if units else "USD"
    return dict(sorted(values.items())), chosen_unit, meta


def _fiscal_year(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _column_label(end: str, index: int, names: list[str], info: dict) -> str:
    """Fiscal period plus end date, with no comma so the CSV stays aligned."""
    span = names[index] if index < len(names) else ""
    fy, fp = _fiscal_year(info.get("fy")), info.get("fp")
    if fy is not None and fp:
        fp = str(fp).upper()
        head = f"FY{fy}" if fp == "FY" else f"{fp} FY{fy}"
        return f"{head} ended {end}{span}"
    return f"{end}{span}"


def _info_for(end: str, index: int, metas: dict[str, dict]) -> dict:
    for meta in metas.values():
        info = meta.get((end, index)) or {}
        if info.get("fy") is not None or info.get("fp") or info.get("form"):
            return info
    return {}


def _format_value(value, unit: str) -> str:
    if value is None:
        return _BLANK
    if not unit or "/" in unit:
        return f"{value:.2f}"
    return f"{value / 1e6:.0f}"


def _header_unit(lines: dict[str, tuple[dict, str]]) -> str:
    for values, unit in lines.values():
        if values and unit and "/" not in unit:
            return unit
    return "USD"


def _limit_periods(periods: list[tuple[str, int]], quarterly: bool) -> list[tuple[str, int]]:
    """Latest ``_RECENT_QUARTERS`` (or ``_RECENT_YEARS``) period-ends, newest first."""
    max_ends = _RECENT_QUARTERS if quarterly else _RECENT_YEARS
    ends: list[str] = []
    for end, _index in sorted(periods, reverse=True):
        if end not in ends:
            ends.append(end)
        if len(ends) == max_ends:
            break
    keep = set(ends)
    selected = [period for period in periods if period[0] in keep]
    # Newest end first; within an end, the discrete quarter before year-to-date.
    return sorted(selected, key=lambda period: (period[0], -period[1]), reverse=True)


def _only_annual_forms(periods: list[tuple[str, int]], metas: dict[str, dict]) -> bool:
    """Whether every selected fact comes from a 10-K / 20-F / 40-F."""
    selected = set(periods)
    forms = [info.get("form") or ""
             for meta in metas.values()
             for key, info in meta.items()
             if key in selected and info.get("form")]
    return bool(forms) and all(form.startswith(_ANNUAL_FORMS) for form in forms)


def _value_at(values: dict, chosen: dict, end: str, index: int):
    return values.get((end, index)) if chosen.get(end) == index else None


def _yoy_block(lines: dict, metas: dict, chosen: dict, periods: list[tuple[str, int]],
               names: list[str]) -> str:
    """Same-fiscal-period YoY so the model does not have to align columns."""
    displayed = [(end, index) for end, index in periods if index == 0] or list(periods)
    out: list[str] = []
    for label in _YOY_ROWS:
        if label not in lines:
            continue
        values, unit = lines[label]
        meta = metas[label]
        for end, index in displayed:
            current = _value_at(values, chosen[label], end, index)
            if current is None:
                continue
            prior = _prior_year_value(end, index, values, meta)
            if prior is None:
                continue
            prior_end, prior_val = prior
            if prior_val == 0:
                continue
            pct = (current - prior_val) / abs(prior_val) * 100
            curr_l = _column_label(end, index, names, meta.get((end, index)) or {})
            prior_l = _column_label(prior_end, index, names, meta.get((prior_end, index)) or {})
            sign = "+" if pct >= 0 else ""
            out.append(
                f"{label}: {curr_l} vs {prior_l}: "
                f"{_format_value(current, unit)} vs {_format_value(prior_val, unit)} "
                f"({sign}{pct:.1f}%)"
            )
    if not out:
        return ""
    return ("# YoY on matching fiscal periods; use these rather than aligning columns yourself\n"
            + "\n".join(out) + "\n")


def _prior_year_value(end: str, index: int, values: dict, meta: dict):
    """The same fiscal quarter/year one year earlier, from the untruncated series."""
    info = meta.get((end, index)) or {}
    fy, fp = _fiscal_year(info.get("fy")), info.get("fp")
    if fy is not None and fp:
        want_fp = str(fp).upper()
        for (prior_end, prior_index), prior_info in meta.items():
            if prior_index != index:
                continue
            if (_fiscal_year(prior_info.get("fy")) == fy - 1
                    and str(prior_info.get("fp") or "").upper() == want_fp
                    and (prior_end, prior_index) in values):
                return prior_end, values[(prior_end, prior_index)]
    try:
        target = date.fromisoformat(end).replace(year=date.fromisoformat(end).year - 1)
    except ValueError:
        return None
    best = None
    for prior_end, prior_index in values:
        if prior_index != index:
            continue
        try:
            delta = abs((date.fromisoformat(prior_end) - target).days)
        except ValueError:
            continue
        if delta <= 15 and (best is None or delta < best[0]):
            best = (delta, prior_end)
    if best:
        return best[1], values[(best[1], index)]
    return None


def _combined_cash_row(us_gaap: dict, as_of_date: str, spans, forms, lines, chosen, periods) -> str:
    """Cash plus short-term investments: the combined tag, else the sum."""
    tagged, unit, _meta = _as_of(us_gaap, _COMBINED_CASH_TAGS, as_of_date, spans, forms)
    cash_vals, cash_unit = lines["Cash and Equivalents"]
    sti_vals, _ = lines["Short-Term Investments"]
    cells = []
    for end, index in periods:
        tagged_val = tagged.get((end, index))
        cash = _value_at(cash_vals, chosen["Cash and Equivalents"], end, index)
        sti = _value_at(sti_vals, chosen["Short-Term Investments"], end, index)
        if tagged_val is not None:
            value, row_unit = tagged_val, unit
        elif cash is not None and sti is not None:
            value, row_unit = cash + sti, cash_unit
        elif cash is not None and not sti_vals:
            value, row_unit = cash, cash_unit
        else:
            value, row_unit = None, cash_unit
        cells.append(_format_value(value, row_unit))
    return ",".join(["Cash and Short-Term Investments"] + cells)


def _statement(kind: str, ticker: str, freq: str, as_of_date: str, title: str) -> str:
    as_of_date = as_of_date or datetime.now().strftime("%Y-%m-%d")
    cik = cik_for(ticker)
    if cik is None:
        raise NoMarketDataError(ticker, ticker, "not a US SEC filer")

    facts = _cached_json(_FACTS_URL.format(cik=cik), f"CIK{cik}.json")
    namespaces = (facts.get("facts") or {})
    us_gaap = namespaces.get("us-gaap")
    if not us_gaap:
        detail = "US filer with no us-gaap facts"
        if "ifrs-full" in namespaces:
            detail += " (IFRS facts only; falling through)"
        raise NoMarketDataError(ticker, ticker, detail)

    quarterly = freq.lower() == "quarterly"
    if quarterly:
        spans = (_SPANS["quarterly"], *((low, high) for low, high, _ in _YEAR_TO_DATE))
        names = ["", *(f" ({months} months)" for _, _, months in _YEAR_TO_DATE)]
    else:
        spans, names = (_SPANS["annual"],), [""]
    forms = () if quarterly else _ANNUAL_FORMS
    lines: dict[str, tuple[dict, str]] = {}
    metas: dict[str, dict] = {}
    for label, tags in _STATEMENTS[kind]:
        values, unit, meta = _as_of(us_gaap, tags, as_of_date, spans, forms)
        lines[label] = (values, unit)
        metas[label] = meta
    # Each row takes the shortest span it reports for a period, and a column
    # holds one span of one period, so a row filed only to date keeps its figure
    # beside a row filed by quarter.
    chosen = {label: {} for label in lines}
    for label, (values, _) in lines.items():
        for end, index in values:
            chosen[label][end] = min(index, chosen[label].get(end, index))
    periods = sorted({(end, index) for spans_of in chosen.values() for end, index in spans_of.items()})
    if not periods:
        raise NoMarketDataError(ticker, ticker, f"no {freq} {title.lower()} filed by {as_of_date}")

    latest_end = max(end for end, _index in periods)
    if date.fromisoformat(as_of_date) - date.fromisoformat(latest_end) > _STALE_AFTER:
        raise NoMarketDataError(
            ticker, ticker,
            f"no recent {freq} {title.lower()} filed by {as_of_date} "
            f"(latest period ended {latest_end})",
        )
    if quarterly and _only_annual_forms(periods, metas):
        raise NoMarketDataError(ticker, ticker, f"no {freq} {title.lower()} filed by {as_of_date}")

    periods = _limit_periods(periods, quarterly)
    unit = _header_unit(lines)
    header = (
        f"# {title} for {ticker.upper()} ({freq}), {unit} in millions unless the row says otherwise\n"
        f"# SEC EDGAR facts filed on or before {as_of_date}, at the values filed then. "
        f"Latest periods first; missing cells say '{_BLANK}'\n\n"
    )
    columns = [_column_label(end, index, names, _info_for(end, index, metas))
               for end, index in periods]
    rows = [",".join([""] + columns)]
    for label, (values, row_unit) in lines.items():
        # Every row spans the same columns, or a reader lines the table up wrong.
        if not values:
            rows.append(",".join([label] + ["unavailable (not tagged by this filer)"] * len(periods)))
            continue
        name = label if row_unit == "USD" else f"{label} ({row_unit})"
        # Plain numbers: a thousands separator would split the CSV field.
        cells = [_format_value(_value_at(values, chosen[label], end, index), row_unit)
                 for end, index in periods]
        rows.append(",".join([name] + cells))
    if kind == "balance_sheet":
        rows.append(_combined_cash_row(us_gaap, as_of_date, spans, forms, lines, chosen, periods))
    body = header + "\n".join(rows) + "\n"
    return body + _yoy_block(lines, metas, chosen, periods, names)


def get_balance_sheet(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Balance sheet as filed on or before ``as_of_date``."""
    return _statement("balance_sheet", ticker, freq, as_of_date, "Balance Sheet")


def get_income_statement(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Income statement as filed on or before ``as_of_date``.

    A fourth quarter is never derived: filers report it only inside the annual
    figure, and subtracting three separately filed quarters would invent a number
    with no filing date behind it.
    """
    return _statement("income_statement", ticker, freq, as_of_date, "Income Statement")


def get_cashflow(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Cash flow statement as filed on or before ``as_of_date``."""
    return _statement("cashflow", ticker, freq, as_of_date, "Cash Flow Statement")
