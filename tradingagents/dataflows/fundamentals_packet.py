"""Point-in-time fundamentals packet: every figure labelled, all maths in Python.

The fundamentals analyst calls ``get_fundamentals_packet`` once and quotes this
output. It does not sum, subtract or divide. One source per field; a second
source that disagrees by more than 2% is a CONFLICT flag, not a second number.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.sec_8k import SubsequentEvent, load_subsequent_events
from tradingagents.dataflows.statement_tables import (
    PeriodCol,
    StatementTable,
    fill_fiscal,
    fy_end_month_from_columns,
    matching_prior_index,
    parse_period_label,
    period_label,
    standalone_quarters,
    streak_positive,
    ttm,
    uses_start_year_naming,
    yoy_pair,
)

logger = logging.getLogger(__name__)

_ETF_TYPES = {"ETF", "MUTUALFUND"}
_CONFLICT_TOL = 0.02
_MCAP_SHARE_TOL = 0.15
_ADR_RATIO = {"SONY": 1, "SNE": 1}

# Yahoo rows that are duplicates, vendor-computed, or over-granular (SONY coverage).
_YAHOO_DROP = {
    "Accounts Receivable", "Accumulated Depreciation", "Available For Sale Securities",
    "Buildings And Improvements", "Capital Lease Obligations", "Cash Equivalents",
    "Cash Financial", "Common Stock", "Common Stock Equity", "Construction In Progress",
    "Current Deferred Revenue", "Finished Goods", "Gross PPE", "Income Tax Payable",
    "Invested Capital", "Land And Improvements", "Machinery Furniture Equipment",
    "Net Debt", "Net Tangible Assets", "Non Current Deferred Taxes Assets",
    "Non Current Deferred Taxes Liabilities",
    "Non Current Pension And Other Postretirement Benefit Plans",
    "Other Current Borrowings", "Other Equity Adjustments", "Other Properties",
    "Payables", "Properties", "Raw Materials", "Tangible Book Value",
    "Total Capitalization", "Work In Process", "Working Capital",
    "Change In Income Tax Payable", "Change In Other Working Capital", "Change In Payable",
    "Changes In Account Receivables", "Common Stock Dividend Paid", "Common Stock Payments",
    "Deferred Income Tax", "Depreciation", "Depreciation And Amortization", "Free Cash Flow",
    "Long Term Debt Payments", "Net Business Purchase And Sale", "Net Common Stock Issuance",
    "Net Income From Continuing Operations", "Net Investment Purchase And Sale",
    "Net Long Term Debt Issuance", "Net PPE Purchase And Sale", "Purchase Of PPE",
    "Short Term Debt Issuance", "Short Term Debt Payments",
    "Average Dilution Earnings", "Diluted NI Availto Com Stockholders", "EBIT", "EBITDA",
    "Interest Expense Non Operating", "Interest Income Non Operating", "Net Income",
    "Net Income Common Stockholders", "Net Interest Income",
    "Net Non Operating Interest Income Expense", "Normalized EBITDA", "Normalized Income",
    "Operating Revenue", "Other Non Operating Income Expenses", "Reconciled Cost Of Revenue",
    "Reconciled Depreciation", "Tax Effect Of Unusual Items", "Tax Rate For Calcs",
    "Total Expenses", "Total Unusual Items", "Total Unusual Items Excluding Goodwill",
}

_EPS_HINTS = ("eps", "per share")
_SHARE_HINTS = ("share", "shares")

_ROW_ALIASES = {
    "Revenue": ("Revenue", "Total Revenue", "Operating Revenue"),
    "Gross Profit": ("Gross Profit",),
    "Operating Income": ("Operating Income", "Operating Income (as reported)",
                         "Total Operating Income As Reported"),
    "Operating Income As Reported": ("Total Operating Income As Reported",),
    "Net Income": ("Net Income", "Net Income From Continuing And Discontinued Operation",
                   "Net Income Common Stockholders"),
    "Net Income Continuing": ("Net Income (continuing)", "Net Income, continuing",
                              "Net Income From Continuing Operation Net Minority Interest",
                              "Net Income Continuous Operations"),
    "Net Income Discontinued": ("Net Income (discontinued)", "Net Income, discontinued",
                                "Net Income Discontinuous Operations"),
    "Diluted EPS": ("Diluted EPS", "Diluted EPS (USD/shares)"),
    "Diluted Average Shares": ("Diluted Average Shares",),
    "SG&A": ("SG&A", "Selling General And Administration"),
    "R&D": ("R&D", "Research Development"),
    "Pretax Income": ("Pretax Income",),
    "Tax Provision": ("Tax Provision",),
    "Operating Cash Flow": ("Operating Cash Flow",),
    "Operating Cash Flow Continuing": ("Cash Flow From Continuing Operating Activities",),
    "Investing Cash Flow": ("Investing Cash Flow",),
    "Financing Cash Flow": ("Financing Cash Flow",),
    "Capex": ("Capital Expenditure",),
    "D&A": ("Depreciation and Amortization", "Depreciation Amortization Depletion"),
    "SBC": ("Share-based Compensation", "Stock Based Compensation"),
    "Buybacks": ("Buybacks", "Repurchase Of Capital Stock"),
    "Dividends Paid": ("Dividends Paid", "Cash Dividends Paid"),
    "Equity Issuance": ("Equity Issuance", "Issuance Of Capital Stock"),
    "Cash": ("Cash and Equivalents", "Cash And Cash Equivalents"),
    "STI": ("Short-Term Investments", "Other Short Term Investments"),
    "Cash+STI": ("Cash and Short-Term Investments",
                 "Cash Cash Equivalents And Short Term Investments"),
    "Inventory": ("Inventory",),
    "Receivables": ("Receivables",),
    "Total Assets": ("Total Assets",),
    "Current Assets": ("Current Assets",),
    "Total Liabilities": ("Total Liabilities", "Total Liabilities Net Minority Interest"),
    "Current Liabilities": ("Current Liabilities",),
    "Current Debt": ("Current Debt",),
    "Long-Term Debt": ("Long-Term Debt", "Long Term Debt"),
    "Total Debt": ("Total Debt",),
    "Lease Current": ("Operating Lease Liabilities (current)",
                      "Operating Lease Liabilities, current", "Current Capital Lease Obligation"),
    "Lease Noncurrent": ("Operating Lease Liabilities (noncurrent)",
                         "Operating Lease Liabilities, noncurrent",
                         "Long Term Capital Lease Obligation"),
    "Equity": ("Stockholders Equity",),
    "NCI": ("Minority Interest",),
    "Shares Outstanding": ("Shares Outstanding", "Ordinary Shares Number"),
}


@dataclass
class PricePoint:
    close: float
    day: str
    previous: float | None = None
    previous_day: str | None = None
    currency: str = "USD"
    high_52w: float | None = None
    low_52w: float | None = None
    ma50: float | None = None
    ma200: float | None = None


@dataclass
class EarningsRow:
    day: str
    reported: float | None = None
    estimate: float | None = None
    surprise_pct: float | None = None


@dataclass
class InsiderSummary:
    net_buys: int = 0
    net_sells: int = 0
    buy_value: float | None = None
    sell_value: float | None = None
    top: list[str] = field(default_factory=list)
    raw_missing: bool = False


@dataclass
class FundSnapshot:
    name: str | None = None
    family: str | None = None
    category: str | None = None
    expense_ratio: float | None = None
    aum: float | None = None
    yield_ttm: float | None = None
    holdings_count: int | None = None
    top_holdings: list[tuple[str, float | None]] = field(default_factory=list)
    duration: float | None = None
    ytm: float | None = None
    nav: float | None = None
    inception: str | None = None
    quote_type: str = "ETF"


@dataclass
class PacketInputs:
    ticker: str
    as_of: str
    name: str | None = None
    quote_type: str = "EQUITY"
    country: str | None = None
    currency_price: str = "USD"
    currency_report: str = "USD"
    statements: dict[str, StatementTable] = field(default_factory=dict)
    yahoo_info: dict[str, Any] = field(default_factory=dict)
    price: PricePoint | None = None
    fx_usd: float | None = None  # reporting units per 1 valuation unit (e.g. JPY per USD)
    fx_pair: str | None = None
    fx_day: str | None = None
    fx_source: str | None = None
    home_price: PricePoint | None = None
    next_earnings: str | None = None
    surprises: list[EarningsRow] = field(default_factory=list)
    insiders: InsiderSummary | None = None
    fund: FundSnapshot | None = None
    subsequent_events: list[SubsequentEvent] = field(default_factory=list)
    extra_flags: list[str] = field(default_factory=list)


@dataclass
class Packet:
    ticker: str
    as_of: str
    markdown: str
    summary: str
    numbers: list[float]
    quote_type: str = "EQUITY"
    metrics: dict[str, float | None] = field(default_factory=dict)


def _is_eps(name: str) -> bool:
    lower = name.lower()
    return any(h in lower for h in _EPS_HINTS)


def _is_shares(name: str) -> bool:
    lower = name.lower()
    return any(h in lower for h in _SHARE_HINTS) and "based" not in lower and "compensation" not in lower


def lookup_row(table: StatementTable | None, key: str) -> list[float | None] | None:
    if table is None:
        return None
    for name in _ROW_ALIASES.get(key, (key,)):
        if name in table.rows:
            return table.native_row(name)
        for actual in table.rows:
            if actual.lower() == name.lower():
                return table.native_row(actual)
    return None


def lookup_scalar(table: StatementTable | None, key: str, index: int = 0) -> float | None:
    row = lookup_row(table, key)
    if row is None or index >= len(row):
        return None
    return row[index]


def _as_spend(values: list[float | None]) -> list[float | None]:
    nums = [v for v in values if v is not None]
    if nums and sum(1 for v in nums if v < 0) > len(nums) / 2:
        return [None if v is None else -v for v in values]
    return values


def _as_outflow_positive(values: list[float | None]) -> list[float | None]:
    """Buybacks/dividends as positive amounts spent."""
    return _as_spend(values)


_CCY_ALIASES = {"RMB": "CNY", "CNH": "CNY"}
_MINOR_TO_MAJOR = {
    "GBp": ("GBP", 100.0),
    "GBX": ("GBP", 100.0),
    "ZAc": ("ZAR", 100.0),
    "ILA": ("ILS", 100.0),
}
# Yahoo ``CCY=X`` for these is USD per 1 CCY (EURUSD-style). Others are CCY per 1 USD.
_USD_IS_QUOTE = frozenset({"EUR", "GBP", "AUD", "NZD"})
_SUFFIX_QUOTE = {
    ".TO": "CAD", ".V": "CAD", ".L": "GBp", ".AS": "EUR", ".PA": "EUR",
    ".DE": "EUR", ".F": "EUR", ".T": "JPY", ".HK": "HKD", ".AX": "AUD",
    ".CO": "DKK", ".ST": "SEK", ".SW": "CHF", ".OL": "NOK",
}


def normalize_ccy(raw: str | None) -> str | None:
    """ISO code (or Yahoo minor unit like GBp). ``iso4217:CNY`` and RMB → CNY."""
    if not raw:
        return None
    text = str(raw).strip()
    if ":" in text:
        text = text.split(":")[-1].strip()
    if text in _MINOR_TO_MAJOR:
        return text
    if not text:
        return None
    upper = text.upper()
    return _CCY_ALIASES.get(upper, upper)


def major_unit(ccy: str | None) -> str:
    code = normalize_ccy(ccy) or "USD"
    if code in _MINOR_TO_MAJOR:
        return _MINOR_TO_MAJOR[code][0]
    return code


def quote_to_major(amount: float, ccy: str | None) -> float:
    """Pence/cents quote → major unit (GBp 2500 → 25 GBP)."""
    code = normalize_ccy(ccy) or "USD"
    if code in _MINOR_TO_MAJOR:
        return amount / _MINOR_TO_MAJOR[code][1]
    return amount


def display_unit(currency: str) -> tuple[float, int, str]:
    code = normalize_ccy(currency) or "USD"
    if code in {"JPY", "KRW"}:
        return 1e9, 1, "bn"
    return 1e6, 0, "M"


def unit_label(currency: str) -> str:
    """e.g. ``CNY M``, ``EUR M``, ``JPY bn``, ``USD M``."""
    code = normalize_ccy(currency) or "USD"
    _scale, _decimals, suffix = display_unit(code)
    return f"{code} {suffix}"


def fmt_amount(value: float | None, currency: str, *, signed: bool = True) -> str:
    if value is None:
        return "–"
    code = normalize_ccy(currency) or (currency or "USD")
    scale, decimals, suffix = display_unit(code)
    scaled = value / scale
    number = f"{scaled:,.{decimals}f}"
    return f"{code} {number}{suffix}" if suffix else f"{code} {number}"


def fmt_plain(value: float | None, currency: str) -> str:
    """Amount without currency code, for table cells already labelled by the header."""
    if value is None:
        return "–"
    scale, decimals, _suffix = display_unit(currency)
    return f"{value / scale:,.{decimals}f}"


def reporting_currency(inputs: PacketInputs) -> str:
    """XBRL/Yahoo statement unit first; then ``financialCurrency``. Never the quote ccy."""
    for key in (
        "income_statement:quarterly", "income_statement:annual",
        "balance_sheet:quarterly", "balance_sheet:annual",
        "cashflow:quarterly", "cashflow:annual",
    ):
        table = inputs.statements.get(key)
        if table is None:
            continue
        code = normalize_ccy(table.currency)
        if code:
            return code
    return normalize_ccy(inputs.currency_report) or "USD"


def quote_currency(ticker: str, info: dict | None = None) -> str:
    info = info or {}
    code = normalize_ccy(info.get("currency"))
    if code:
        return code
    upper = (ticker or "").upper()
    for suffix, ccy in _SUFFIX_QUOTE.items():
        if upper.endswith(suffix):
            return ccy
    return "USD"


def _fx_report_per_val(inputs: PacketInputs) -> float | None:
    """Reporting-currency units per 1 valuation (major quote) unit."""
    rate = inputs.fx_usd
    if rate is None or rate == 0:
        return None
    return float(rate)


def _to_valuation(amount: float | None, inputs: PacketInputs, report: str, val_ccy: str) -> float | None:
    if amount is None:
        return None
    if (normalize_ccy(report) or report) == val_ccy:
        return amount
    rate = _fx_report_per_val(inputs)
    if rate is None:
        return None
    return amount / rate


def _pct(value: float | None) -> str:
    if value is None:
        return "–"
    return f"{value:.1f}%"


# --- tool-dump loader (hermetic fixtures) -----------------------------------

_SECTION_RE = re.compile(r"^######## (\S+) \((.*)\)\s*$")


def load_tool_dump(text: str, ticker: str, as_of: str) -> PacketInputs:
    """Parse a captured ``*.tools.txt`` dump into packet inputs (no network)."""
    inputs = PacketInputs(ticker=ticker, as_of=as_of)
    sections = _split_sections(text)
    for key, body in sections.items():
        name, rest = key[0], key[1]
        if name == "get_fundamentals":
            inputs.yahoo_info, inputs.name = _parse_info_block(body)
            curr = str(inputs.yahoo_info.get("Revenue (TTM)") or "")
            if "JPY" in curr:
                inputs.currency_report = "JPY"
            elif "USD" in curr:
                inputs.currency_report = "USD"
        elif name in {"get_balance_sheet", "get_cashflow", "get_income_statement"}:
            kind = {"get_balance_sheet": "balance_sheet", "get_cashflow": "cashflow",
                    "get_income_statement": "income_statement"}[name]
            freq = "annual" if "'annual'" in rest or '"annual"' in rest else "quarterly"
            table = _parse_statement_csv(body, kind, freq)
            if table is not None:
                inputs.statements[f"{kind}:{freq}"] = table
                if table.currency:
                    inputs.currency_report = table.currency
        elif name == "get_earnings_context":
            inputs.next_earnings, inputs.surprises = _parse_earnings(body)
        elif name == "get_insider_transactions":
            inputs.insiders = _parse_insiders(body)
    return inputs


def _split_sections(text: str) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    current: tuple[str, str] | None = None
    buf: list[str] = []
    for line in text.splitlines():
        match = _SECTION_RE.match(line)
        if match:
            if current is not None:
                out[current] = "\n".join(buf)
            current = (match.group(1), match.group(2))
            buf = []
        else:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf)
    return out


def _parse_info_block(body: str) -> tuple[dict[str, Any], str | None]:
    info: dict[str, Any] = {}
    name = None
    for line in body.splitlines():
        if ":" not in line or line.startswith("#"):
            continue
        label, _, raw = line.partition(":")
        value = raw.strip()
        info[label.strip()] = value
        if label.strip() == "Name":
            name = value
        num = _parse_info_number(value)
        if num is not None:
            info[f"{label.strip()}__num"] = num
    return info, name


def _parse_info_number(value: str) -> float | None:
    text = value.replace(",", "").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_statement_csv(body: str, kind: str, freq: str) -> StatementTable | None:
    comments = [line for line in body.splitlines() if line.startswith("#")]
    currency = "JPY" if any("JPY" in c for c in comments) else "USD"
    scale = 1e6 if any("in millions" in c for c in comments) else 1.0
    source = "edgar" if any("SEC EDGAR" in c for c in comments) else "yahoo"
    table_lines = [line for line in body.splitlines()
                   if line and not line.startswith("#") and "," in line]
    if not table_lines:
        return None
    reader = csv.reader(table_lines)
    rows_list = list(reader)
    if not rows_list or not rows_list[0]:
        return None
    header = rows_list[0]
    col_labels = header[1:]
    instant = kind == "balance_sheet"
    columns = [parse_period_label(label, instant=instant) for label in col_labels]
    if freq == "annual":
        for col in columns:
            if not instant:
                col.span_months = 12
                col.fp = col.fp or "FY"
    rows: dict[str, list[float | None]] = {}
    unscaled: set[str] = set()
    for raw in rows_list[1:]:
        if not raw or not raw[0]:
            continue
        name = raw[0]
        if name.startswith("Revenue:") or name.startswith("Net Income:"):
            continue  # YoY prose, not a row
        cells: list[float | None] = []
        for cell in raw[1:len(columns) + 1]:
            cells.append(_csv_number(cell))
        while len(cells) < len(columns):
            cells.append(None)
        rows[name] = cells
        if _is_eps(name) or _is_shares(name):
            unscaled.add(name)
    return StatementTable(
        kind=kind, freq=freq, source=source, currency=currency, scale=scale,
        columns=columns, rows=rows, unscaled_rows=unscaled,
    )


def _csv_number(cell: str | None) -> float | None:
    if cell is None:
        return None
    text = cell.strip()
    if text in {"", "nan", "not filed", "not tagged", "–", "-", "unavailable (not tagged by this filer)"}:
        return None
    if "unavailable" in text or "not tagged" in text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _parse_earnings(body: str) -> tuple[str | None, list[EarningsRow]]:
    next_date = None
    rows: list[EarningsRow] = []
    for line in body.splitlines():
        if line.startswith("Next earnings date:"):
            rest = line.split(":", 1)[1].strip()
            if rest and not rest.lower().startswith("not"):
                next_date = rest.split()[0]
        match = re.match(
            r"(\d{4}-\d{2}-\d{2}):(?:.*reported EPS\s+([-+]?\d+(?:\.\d+)?))?"
            r"(?:.*estimate\s+([-+]?\d+(?:\.\d+)?))?"
            r"(?:.*surprise\s+([-+]?\d+(?:\.\d+)?)%)?",
            line, re.IGNORECASE,
        )
        if match:
            rows.append(EarningsRow(
                day=match.group(1),
                reported=float(match.group(2)) if match.group(2) else None,
                estimate=float(match.group(3)) if match.group(3) else None,
                surprise_pct=float(match.group(4)) if match.group(4) else None,
            ))
    return next_date, rows


def _parse_insiders(body: str) -> InsiderSummary:
    if "No insider" in body or "withheld" in body.lower():
        return InsiderSummary(raw_missing=True)
    try:
        start = body.find("\n")
        frame_text = body[start + 1:] if start >= 0 else body
        reader = csv.DictReader(io.StringIO(frame_text))
        rows = list(reader)
    except (csv.Error, ValueError):
        return InsiderSummary(raw_missing=True)
    if not rows:
        return InsiderSummary(raw_missing=True)
    # Best-effort: keep the three largest |value| rows as captions.
    top = []
    for row in rows[:20]:
        bits = ", ".join(f"{k}={v}" for k, v in list(row.items())[:4] if v)
        if bits:
            top.append(bits)
        if len(top) == 3:
            break
    return InsiderSummary(top=top)


def merge_extra_rows(inputs: PacketInputs, extra: dict[str, dict[str, dict[str, float]]]) -> None:
    """Merge extra {table_key: {row: {period_end: value}}} in the table's own scale."""
    for key, row_map in extra.items():
        table = inputs.statements.get(key)
        if table is None:
            continue
        for row_name, by_end in row_map.items():
            cells = table.rows.get(row_name) or [None] * len(table.columns)
            if len(cells) < len(table.columns):
                cells = cells + [None] * (len(table.columns) - len(cells))
            for i, col in enumerate(table.columns):
                if col.end in by_end:
                    cells[i] = by_end[col.end]
                elif col.source_label in by_end:
                    cells[i] = by_end[col.source_label]
            table.rows[row_name] = cells
            if _is_eps(row_name) or _is_shares(row_name):
                table.unscaled_rows.add(row_name)


# --- build -----------------------------------------------------------------

def build_packet(inputs: PacketInputs) -> Packet:
    if (inputs.quote_type or "").upper() in _ETF_TYPES or inputs.fund is not None:
        return _build_fund(inputs)
    return _build_equity(inputs)


def _fy_meta(inputs: PacketInputs) -> tuple[int, str]:
    annual = (inputs.statements.get("income_statement:annual")
              or inputs.statements.get("balance_sheet:annual")
              or inputs.statements.get("cashflow:annual"))
    month = None
    if annual and annual.columns:
        try:
            month = date.fromisoformat(annual.columns[0].end).month
        except ValueError:
            month = None
    if month is None:
        cols: list[PeriodCol] = []
        for key in ("income_statement:quarterly", "balance_sheet:quarterly"):
            table = inputs.statements.get(key)
            if table:
                cols.extend(table.columns)
                break
        month = fy_end_month_from_columns(cols, default=12)
    country = inputs.country or ("Japan" if inputs.ticker.upper() in {"SONY", "SNE"} else None)
    naming = "start" if uses_start_year_naming(inputs.ticker, country, month) else "end"
    return month, naming


def _stand(inputs: PacketInputs, kind: str, flow: bool) -> StatementTable | None:
    quarterly = inputs.statements.get(f"{kind}:quarterly")
    if quarterly is None:
        return None
    annual = inputs.statements.get(f"{kind}:annual")
    month, naming = _fy_meta(inputs)
    table = standalone_quarters(quarterly, annual, fy_end_month=month, naming=naming, flow=flow)
    # Keep the most recent 8 quarter-ends.
    if len(table.columns) > 8:
        keep = 8
        table.columns = table.columns[:keep]
        table.rows = {n: vals[:keep] for n, vals in table.rows.items()}
        table.derived_mask = {n: m[:keep] for n, m in table.derived_mask.items()}
    return table


def _annual_filled(inputs: PacketInputs, kind: str) -> StatementTable | None:
    table = inputs.statements.get(f"{kind}:annual")
    if table is None:
        return None
    month, naming = _fy_meta(inputs)
    cols = fill_fiscal(table.columns, month, naming)
    native = {n: table.native_row(n) for n in table.rows}
    return StatementTable(
        kind=table.kind, freq=table.freq, source=table.source, currency=table.currency,
        scale=1.0, columns=cols, rows=native, unscaled_rows=set(table.rows),
    )


def _build_equity(inputs: PacketInputs) -> Packet:
    currency = reporting_currency(inputs)
    inputs.currency_report = currency
    month, naming = _fy_meta(inputs)
    isq = _stand(inputs, "income_statement", True)
    cfq = _stand(inputs, "cashflow", True)
    bsq = _stand(inputs, "balance_sheet", False)
    isa = _annual_filled(inputs, "income_statement")
    cfa = _annual_filled(inputs, "cashflow")
    bsa = _annual_filled(inputs, "balance_sheet")

    rev = lookup_row(isq, "Revenue") or []
    gp = lookup_row(isq, "Gross Profit") or []
    oi_rep = lookup_row(isq, "Operating Income As Reported") or lookup_row(isq, "Operating Income") or []
    ni_total = lookup_row(isq, "Net Income") or []
    ni_cont = lookup_row(isq, "Net Income Continuing") or []
    ni_disc = lookup_row(isq, "Net Income Discontinued") or []
    eps = lookup_row(isq, "Diluted EPS") or []
    shares_dil = lookup_row(isq, "Diluted Average Shares") or []
    sga = lookup_row(isq, "SG&A") or []
    rnd = lookup_row(isq, "R&D") or []
    pretax = lookup_row(isq, "Pretax Income") or []
    tax = lookup_row(isq, "Tax Provision") or []

    ocf = lookup_row(cfq, "Operating Cash Flow") or []
    ocfc = lookup_row(cfq, "Operating Cash Flow Continuing") or []
    capex_raw = lookup_row(cfq, "Capex") or []
    capex = _as_spend(capex_raw)
    da = lookup_row(cfq, "D&A") or []
    sbc = lookup_row(cfq, "SBC") or []
    buy = _as_outflow_positive(lookup_row(cfq, "Buybacks") or [])
    divs = _as_outflow_positive(lookup_row(cfq, "Dividends Paid") or [])
    equity_raised = lookup_row(cfq, "Equity Issuance") or []
    fcf = _pair_sub(ocf, capex)
    fcfc = _pair_sub(ocfc, capex) if ocfc else fcf
    ebitda = _pair_add(oi_rep, da)

    cash_sti = lookup_row(bsq, "Cash+STI") or _pair_add(
        lookup_row(bsq, "Cash") or [], lookup_row(bsq, "STI") or [])
    cash = lookup_row(bsq, "Cash") or []
    sti = lookup_row(bsq, "STI") or []
    inventory = lookup_row(bsq, "Inventory") or []
    receivables = lookup_row(bsq, "Receivables") or []
    borrow = _pair_add(lookup_row(bsq, "Current Debt") or [], lookup_row(bsq, "Long-Term Debt") or [])
    leases = _pair_add(lookup_row(bsq, "Lease Current") or [], lookup_row(bsq, "Lease Noncurrent") or [])
    total_debt = lookup_row(bsq, "Total Debt") or _pair_add(borrow, leases)
    equity = lookup_row(bsq, "Equity") or []
    nci = lookup_row(bsq, "NCI") or []
    assets = lookup_row(bsq, "Total Assets") or []
    cass = lookup_row(bsq, "Current Assets") or []
    liab = lookup_row(bsq, "Total Liabilities") or []
    cliab = lookup_row(bsq, "Current Liabilities") or []

    ttm_rev = ttm(rev)
    ttm_gp = ttm(gp)
    ttm_oi = ttm(oi_rep)
    ttm_ni = ttm(ni_total)
    ttm_nic = ttm(ni_cont) if ni_cont and any(v is not None for v in ni_cont[:4]) else ttm_ni
    ttm_nid = ttm(ni_disc) if ni_disc and any(v is not None for v in ni_disc[:4]) else 0.0
    ttm_ocf = ttm(ocf)
    ttm_ocfc = ttm(ocfc) if ocfc and any(v is not None for v in ocfc[:4]) else ttm_ocf
    ttm_capex = ttm(capex)
    ttm_fcf = ttm(fcf)
    ttm_fcfc = ttm(fcfc) if fcfc else ttm_fcf
    ttm_da = ttm(da)
    ttm_ebitda = ttm(ebitda) if ebitda else None
    ttm_buy = ttm(buy) if buy and any(v is not None for v in buy[:4]) else 0.0
    ttm_div = ttm(divs) if divs and any(v is not None for v in divs[:4]) else 0.0
    ttm_eps = ttm(eps)
    ttm_sga = ttm(sga)
    ttm_rnd = ttm(rnd)

    # Continuing EPS per quarter = continuing NI / diluted shares when both exist.
    cont_eps = []
    if ni_cont and shares_dil:
        for ni, sh in zip(ni_cont, shares_dil, strict=False):
            if ni is None or not sh:
                cont_eps.append(None)
            else:
                # shares may be native counts or already in millions
                share_n = sh if sh > 1e4 else sh * 1e6
                cont_eps.append(ni / share_n)
    ttm_ceps = ttm(cont_eps) if cont_eps else None

    gm_ttm = (ttm_gp / ttm_rev * 100) if ttm_rev and ttm_gp is not None else None
    om_ttm = (ttm_oi / ttm_rev * 100) if ttm_rev and ttm_oi is not None else None
    om_mrq = (oi_rep[0] / rev[0] * 100) if oi_rep and rev and oi_rep[0] is not None and rev[0] else None
    gm_q = [(g / r * 100) if g is not None and r else None for g, r in zip(gp, rev, strict=False)]
    om_q = [(o / r * 100) if o is not None and r else None for o, r in zip(oi_rep, rev, strict=False)]

    cash0 = cash_sti[0] if cash_sti else None
    debt0 = (borrow[0] if borrow and borrow[0] is not None else (total_debt[0] if total_debt else None))
    lease0 = leases[0] if leases else None
    net_ex = None if cash0 is None or debt0 is None else cash0 - debt0
    net_in = None if cash0 is None or not total_debt or total_debt[0] is None else cash0 - total_debt[0]
    inv0 = inventory[0] if inventory else None

    # H1 SBC: 6-month YTD on the raw quarterly table, else Q1+Q2.
    sbc_h1 = _h1_from_raw(inputs, "cashflow", "SBC")
    if sbc_h1 is None and sbc:
        sbc_h1 = _sum_known(sbc[:2])

    runway_ttm = None
    runway_mrq = None
    runway_pf = None
    cash_in = sum(
        (e.amount or 0.0) for e in inputs.subsequent_events if e.cash_inflow and e.amount
    )
    if cash0 and ttm_fcf is not None and ttm_fcf < 0:
        burn_q = abs(ttm_fcf) / 4
        if burn_q:
            runway_ttm = cash0 / burn_q
            if cash_in:
                runway_pf = (cash0 + cash_in) / burn_q
    if cash0 and fcf and fcf[0] is not None and fcf[0] < 0:
        runway_mrq = cash0 / abs(fcf[0])

    gp_streak = streak_positive(gp)
    fcf_streak = streak_positive(fcf)

    yoy_rev = None
    yoy_rev_pair = None
    if isq is not None and rev:
        prior_i = matching_prior_index(isq.columns, 0)
        if prior_i is not None:
            yoy_rev = yoy_pair(rev[0], rev[prior_i])
            yoy_rev_pair = (rev[0], rev[prior_i])

    flags = list(inputs.extra_flags)
    flags.extend(_quality_flags(inputs, isq, cfq, bsq, ttm_fcf, ttm_ni, ttm_nic, pretax, ocf))
    quote = normalize_ccy((inputs.price.currency if inputs.price else inputs.currency_price) or "USD") or "USD"
    val_ccy = major_unit(quote)
    if currency != val_ccy and _fx_report_per_val(inputs) is None:
        flags.append(
            f"CROSS-CURRENCY: statements in {currency}, price in {quote}; "
            f"no FX rate on or before {inputs.as_of}. Valuation ratios omitted."
        )

    metrics = {
        "ttm_fcf": ttm_fcf,
        "ttm_revenue": ttm_rev,
        "ttm_net_income": ttm_ni,
        "ttm_net_income_continuing": ttm_nic,
        "ttm_buybacks": ttm_buy,
        "ttm_dividends": ttm_div,
        "capital_return_ttm": (ttm_buy or 0) + (ttm_div or 0),
        "cash_sti": cash0,
        "inventory": inv0,
        "sbc_h1": sbc_h1,
        "runway_ttm_quarters": runway_ttm,
        "runway_pro_forma_quarters": runway_pf,
        "subsequent_cash_in": cash_in or None,
    }
    val_figs = _valuation_figures(
        inputs, currency, ttm_rev, ttm_ni, ttm_nic, ttm_fcf, ttm_ebitda,
        cash0, total_debt[0] if total_debt else None, equity[0] if equity else None, ttm_ceps,
    )
    metrics.update(val_figs)

    markdown, summary, numbers = _render_equity(
        inputs, currency, month, naming, isq, cfq, bsq, isa, cfa, bsa,
        rev, gp, gm_q, oi_rep, om_q, ni_total, ni_cont, ni_disc, eps, cont_eps,
        sga, rnd, pretax, tax, ocf, ocfc, capex, fcf, fcfc, da, ebitda, buy, divs,
        cash, sti, cash_sti, inventory, receivables, borrow, leases, total_debt,
        equity, nci, assets, cass, liab, cliab, shares_dil, equity_raised, sbc,
        ttm_rev, ttm_gp, gm_ttm, ttm_oi, om_ttm, om_mrq, ttm_ni, ttm_nic, ttm_nid,
        ttm_eps, ttm_ceps, ttm_ocf, ttm_ocfc, ttm_capex, ttm_fcf, ttm_fcfc,
        ttm_ebitda, ttm_da, ttm_buy, ttm_div, ttm_sga, ttm_rnd, sbc_h1,
        cash0, debt0, lease0, net_ex, net_in, inv0, runway_ttm, runway_mrq,
        gp_streak, fcf_streak, yoy_rev, yoy_rev_pair, flags, metrics,
    )
    return Packet(
        ticker=inputs.ticker, as_of=inputs.as_of, markdown=markdown, summary=summary,
        numbers=numbers, quote_type="EQUITY", metrics=metrics,
    )


def _pair_sub(a: list[float | None], b: list[float | None]) -> list[float | None]:
    n = max(len(a), len(b))
    out = []
    for i in range(n):
        x = a[i] if i < len(a) else None
        y = b[i] if i < len(b) else None
        out.append(None if x is None or y is None else x - y)
    return out


def _pair_add(a: list[float | None], b: list[float | None]) -> list[float | None]:
    n = max(len(a), len(b))
    out = []
    for i in range(n):
        x = a[i] if i < len(a) else None
        y = b[i] if i < len(b) else None
        if x is None and y is None:
            out.append(None)
        else:
            out.append((x or 0) + (y or 0))
    return out


def _sum_known(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) if nums else None


def _h1_from_raw(inputs: PacketInputs, kind: str, key: str) -> float | None:
    raw = inputs.statements.get(f"{kind}:quarterly")
    if raw is None:
        return None
    month, naming = _fy_meta(inputs)
    cols = fill_fiscal(raw.columns, month, naming)
    row = lookup_row(raw, key)
    if row is None:
        return None
    latest_fy = next((c.fy for c in cols if c.fy is not None), None)
    if latest_fy is None:
        return None
    for i, col in enumerate(cols):
        if col.fy == latest_fy and col.fp == "Q2" and col.span_months == 6:
            return row[i]
    return None


def _quality_flags(inputs, isq, cfq, bsq, ttm_fcf, ttm_ni, ttm_nic, pretax, ocf) -> list[str]:
    flags = []
    info = inputs.yahoo_info
    yahoo_fcf = info.get("Free Cash Flow__num")
    if (yahoo_fcf is not None and ttm_fcf is not None and abs(ttm_fcf) > 0
            and abs(yahoo_fcf - ttm_fcf) / abs(ttm_fcf) > _CONFLICT_TOL):
        flags.append(
            f"CONFLICT: Yahoo freeCashflow = {fmt_amount(yahoo_fcf, inputs.currency_report)} "
            f"is Yahoo's own computed FCF. Statements give "
            f"{fmt_amount(ttm_fcf, inputs.currency_report)} TTM; the packet uses the statements."
        )
    yahoo_ni = info.get("Net Income__num")
    if (yahoo_ni is not None and ttm_ni is not None and ttm_nic is not None
            and abs(yahoo_ni - ttm_ni) > abs(ttm_ni) * _CONFLICT_TOL):
        flags.append(
            f"CONFLICT: Yahoo \"Net Income\" {fmt_amount(yahoo_ni, inputs.currency_report)} "
            f"is continuing ops (packet continuing "
            f"{fmt_amount(ttm_nic, inputs.currency_report)}; total "
            f"{fmt_amount(ttm_ni, inputs.currency_report)})."
        )
    intang = lookup_row(bsq, "Other Intangibles") if bsq else None
    # alias
    if bsq is not None:
        for name in bsq.rows:
            if "intangible" in name.lower() and "goodwill" not in name.lower():
                intang = bsq.native_row(name)
                break
    if (intang and len(intang) >= 3 and all(intang[i] is not None for i in range(3))
            and intang[1] > 2 * max(intang[0], intang[2], 1)):
        flags.append(
            "RECLASSIFICATION ARTIFACT: other intangibles jump in the year-end column "
            "and revert the next quarter — a classification difference, not a divestiture."
        )
    hfs = None
    if bsq is not None:
        for name in bsq.rows:
            if "held for sale" in name.lower():
                hfs = bsq.native_row(name)
                break
    if hfs and any(v and v > 0 for v in hfs) and any(not v for v in hfs[:2]):
        flags.append(
            "STRUCTURAL BREAK: assets held for sale appear then vanish. Totals before and "
            "after are not comparable (discontinued operations)."
        )
    pretax_row = pretax
    if pretax_row and cfq is not None:
        for name in cfq.rows:
            if name.lower().startswith("net income from continuing"):
                cf_ni = cfq.native_row(name)
                if (cf_ni and pretax_row and cf_ni[0] is not None and pretax_row[0] is not None
                        and abs(cf_ni[0] - pretax_row[0]) <= max(1.0, 0.002 * abs(pretax_row[0]))):
                    flags.append(
                        "MISLABEL (Yahoo): the cash-flow row \"Net Income From Continuing "
                        "Operations\" equals pretax income. The cash flow starts from income "
                        "before taxes."
                    )
                break
    flags.append(
        "DERIVED (D): standalone quarters from YTD differences and Q4 = FY − 9M; "
        "TTM sums, FCF, margins, runway, continuing EPS; pro forma runway from "
        "subsequent equity inflows."
    )
    if inputs.yahoo_info:
        flags.append(
            "NOT POINT-IN-TIME: shares outstanding, beta, forward EPS and 52-week range "
            "from a current Yahoo snapshot are flagged; the last close is from daily history."
        )
    shares, share_src = _resolve_share_count(inputs)
    share_flag = _share_count_flag(inputs, shares, share_src)
    if share_flag:
        flags.append(share_flag)
    return flags


# --- render ----------------------------------------------------------------

def _q_headers(table: StatementTable | None, month: int, naming: str, n: int | None = None) -> list[str]:
    if table is None:
        return []
    cols = table.columns[:n] if n else table.columns
    return [period_label(c, month, naming) for c in cols]


def _md_table(headers: list[str], rows: list[tuple[str, list[str]]]) -> str:
    if not headers:
        return ""
    head = "| | " + " | ".join(headers) + " |"
    sep = "|---|" + "---|" * len(headers)
    lines = [head, sep]
    for name, cells in rows:
        padded = list(cells) + ["–"] * (len(headers) - len(cells))
        lines.append("| " + name + " | " + " | ".join(padded[:len(headers)]) + " |")
    return "\n".join(lines)


def _cell(values, i, currency, derived=False, kind="amount"):
    if not values or i >= len(values) or values[i] is None:
        return "–"
    if kind == "pct":
        text = _pct(values[i])
    elif kind == "eps":
        text = f"{values[i]:,.2f}"
    elif kind == "shares":
        text = f"{values[i] / 1e6:,.1f}" if abs(values[i]) > 1e4 else f"{values[i]:,.1f}"
    else:
        text = fmt_plain(values[i], currency)
    if derived:
        text += " D"
    return text


def _derived(table: StatementTable | None, name: str, i: int) -> bool:
    if table is None:
        return False
    mask = table.derived_mask.get(name)
    if mask and i < len(mask):
        return mask[i]
    col = table.columns[i] if i < len(table.columns) else None
    return bool(col and col.derived)


def _render_equity(inputs, currency, month, naming, isq, cfq, bsq, isa, cfa, bsa,
                   rev, gp, gm_q, oi_rep, om_q, ni_total, ni_cont, ni_disc, eps, cont_eps,
                   sga, rnd, pretax, tax, ocf, ocfc, capex, fcf, fcfc, da, ebitda, buy, divs,
                   cash, sti, cash_sti, inventory, receivables, borrow, leases, total_debt,
                   equity, nci, assets, cass, liab, cliab, shares_dil, equity_raised, sbc,
                   ttm_rev, ttm_gp, gm_ttm, ttm_oi, om_ttm, om_mrq, ttm_ni, ttm_nic, ttm_nid,
                   ttm_eps, ttm_ceps, ttm_ocf, ttm_ocfc, ttm_capex, ttm_fcf, ttm_fcfc,
                   ttm_ebitda, ttm_da, ttm_buy, ttm_div, ttm_sga, ttm_rnd, sbc_h1,
                   cash0, debt0, lease0, net_ex, net_in, inv0, runway_ttm, runway_mrq,
                   gp_streak, fcf_streak, yoy_rev, yoy_rev_pair, flags, metrics) -> tuple[str, str, list[float]]:
    ticker = inputs.ticker
    as_of = inputs.as_of
    unit_name = unit_label(currency)
    qh = _q_headers(isq or cfq or bsq, month, naming)
    price = inputs.price
    last_close = f"**{price.currency} {price.close:.2f} on {price.day}**" if price else "n/a"
    if price and price.previous is not None:
        last_close += f" (previous {price.previous:.2f} on {price.previous_day})"

    fy_blurb = (
        f"April–March, **named by start year: FY year N = Apr N–Mar N+1**. "
        f"Latest quarter: **{qh[0] if qh else 'n/a'}**"
        if naming == "start" else
        f"Named by end year (US convention). Latest quarter: **{qh[0] if qh else 'n/a'}**"
    )

    yoy_text = "n/a"
    if yoy_rev is not None and yoy_rev_pair:
        yoy_text = (
            f"{yoy_rev:+.1f}% ({fmt_plain(yoy_rev_pair[0], currency)} vs "
            f"{fmt_plain(yoy_rev_pair[1], currency)})"
        )

    if ttm_fcf is not None and ttm_fcf < 0 and runway_ttm is not None:
        runway_line = (
            f"**{runway_ttm:.1f} quarters (≈{runway_ttm / 4:.1f} yrs)** | from "
            f"{(bsq.columns[0].end if bsq else as_of)} | "
            f"{fmt_plain(cash0, currency)} ÷ ({fmt_plain(abs(ttm_fcf), currency)} / 4)"
        )
    elif ttm_fcf is not None and ttm_fcf >= 0:
        runway_line = "n/a: FCF positive | | "
    else:
        runway_line = "n/a | | "

    cash_in = metrics.get("subsequent_cash_in")
    runway_pf = metrics.get("runway_pro_forma_quarters")
    if cash_in and ttm_fcf is not None and ttm_fcf < 0 and runway_pf is not None:
        pf_line = (
            f"| **Runway, TTM FCF burn (pro forma incl. subsequent raises)** D | "
            f"**{runway_pf:.1f} quarters (≈{runway_pf / 4:.1f} yrs)** | from "
            f"{(bsq.columns[0].end if bsq else as_of)} | "
            f"({fmt_plain(cash0, currency)} + {fmt_plain(cash_in, currency)}) ÷ "
            f"({fmt_plain(abs(ttm_fcf), currency)} / 4) |\n"
        )
    else:
        pf_line = ""

    # Header + key metrics (summary for debate agents is this block).
    header = f"""## 0. Header
| Field | Value |
|---|---|
| As-of date | {as_of} (point-in-time: filings and prices on or before this date) |
| Last close | {last_close} |
| Currencies | {_currency_blurb(inputs, currency, unit_name)} |
| Fiscal year | {fy_blurb} |
| Next earnings | {inputs.next_earnings or "n/a"} |
| Sources | Statements: {((isq or cfq or bsq).source if (isq or cfq or bsq) else "n/a")}. Price/estimates: yfinance |
"""
    if ticker.upper() in _ADR_RATIO:
        adr = _ADR_RATIO[ticker.upper()]
        header += f"| ADR ratio | {adr} ADR = {adr} share |\n"
    if inputs.home_price:
        header += (f"| Home listing | {inputs.home_price.close:,.0f} {inputs.home_price.currency} "
                   f"on {inputs.home_price.day} |\n")

    ttm_label = "TTM (sum of 4 standalone quarters)"
    key = f"""## 1. Key metrics (each has a period and a basis; do not recompute)
| Metric | Value | Period | Basis / formula |
|---|---|---|---|
| Revenue TTM | {fmt_amount(ttm_rev, currency)} | {ttm_label} | Sum of standalone quarters |
| Revenue YoY, latest quarter | {yoy_text} | latest vs same quarter prior year | matching fiscal periods only |
| Gross profit TTM | {fmt_amount(ttm_gp, currency)} ({_pct(gm_ttm)} margin) | {ttm_label} | |
| Operating income TTM | {fmt_amount(ttm_oi, currency)} (op margin **{_pct(om_ttm)} TTM**) | {ttm_label} | |
| Operating margin, latest quarter (MRQ) | **{_pct(om_mrq)}** | latest quarter | |
| Net income TTM (total) | **{fmt_amount(ttm_ni, currency)}** | {ttm_label} | includes discontinued ops {fmt_amount(ttm_nid, currency)} |
| Net income TTM (continuing) | {fmt_amount(ttm_nic, currency)} | {ttm_label} | |
| Operating cash flow TTM | {fmt_amount(ttm_ocf, currency)} | {ttm_label} | |
| Capex TTM | {fmt_amount(ttm_capex, currency)} | {ttm_label} | spend (positive) |
| **Free cash flow TTM** | **{fmt_amount(ttm_fcf, currency)}** | {ttm_label} | OCF − capex, from statements |
| Cash + short-term investments | **{fmt_amount(cash0, currency)}** | at {(bsq.columns[0].end if bsq else "n/a")} | |
| Inventory | {fmt_amount(inv0, currency)} | at {(bsq.columns[0].end if bsq else "n/a")} | |
| Total assets | {fmt_amount(assets[0] if assets else None, currency)} | at {(bsq.columns[0].end if bsq else "n/a")} | |
| Current assets | {fmt_amount(cass[0] if cass else None, currency)} | at {(bsq.columns[0].end if bsq else "n/a")} | |
| Total liabilities | {fmt_amount(liab[0] if liab else None, currency)} | at {(bsq.columns[0].end if bsq else "n/a")} | |
| Current liabilities | {fmt_amount(cliab[0] if cliab else None, currency)} | at {(bsq.columns[0].end if bsq else "n/a")} | |
| Debt (borrowings) | {fmt_amount(debt0, currency)} | at {(bsq.columns[0].end if bsq else "n/a")} | current + long-term, ex leases |
| Lease liabilities | {fmt_amount(lease0, currency)} | at {(bsq.columns[0].end if bsq else "n/a")} | |
| Net cash (ex leases) | {fmt_amount(net_ex, currency)} | at {(bsq.columns[0].end if bsq else "n/a")} | cash+STI − borrowings |
| Investing cash flow TTM | {fmt_amount(ttm(lookup_row(cfq, "Investing Cash Flow") or []), currency) if cfq else "–"} | {ttm_label} | |
| R&D TTM | {fmt_amount(ttm_rnd, currency)} | {ttm_label} | |
| SG&A TTM | {fmt_amount(ttm_sga, currency)} | {ttm_label} | |
| SBC H1 | {fmt_amount(sbc_h1, currency)} | first half of latest FY | 6-month YTD or Q1+Q2 |
| Diluted average shares (latest) | {_cell(shares_dil, 0, currency, kind="shares")} | latest quarter | |
| **Runway, TTM FCF burn** | {runway_line} |
{pf_line}| Runway, latest-quarter burn | {f"{runway_mrq:.1f} quarters" if runway_mrq else "n/a"} | | cash+STI ÷ |MRQ FCF| |
| Diluted EPS TTM (GAAP/IFRS) | {f"{ttm_eps:,.2f}" if ttm_eps is not None else "–"} | {ttm_label} | street/adjusted EPS is labelled separately |
"""
    summary = header + "\n" + key

    n = min(8, len(qh))
    def cells(vals, kind="amount", names_for_d=None):
        out = []
        src = isq if names_for_d == "is" else cfq if names_for_d == "cf" else bsq
        for i in range(n):
            derived = bool(src and i < len(src.columns) and src.columns[i].derived)
            out.append(_cell(vals, i, currency, derived=derived, kind=kind))
        return out

    q_rows = [
        ("Revenue", cells(rev, names_for_d="is")),
        ("Gross profit", cells(gp, names_for_d="is")),
        ("Gross margin", cells(gm_q, kind="pct")),
        ("R&D", cells(rnd, names_for_d="is")),
        ("SG&A", cells(sga, names_for_d="is")),
        ("Operating income", cells(oi_rep, names_for_d="is")),
        ("Operating margin", cells(om_q, kind="pct")),
        ("Pretax income", cells(pretax, names_for_d="is")),
        ("Tax provision", cells(tax, names_for_d="is")),
        ("Net income attrib., continuing", cells(ni_cont, names_for_d="is")),
        ("Net income, discontinued ops", cells(ni_disc, names_for_d="is")),
        ("**Net income attrib., total**", cells(ni_total, names_for_d="is")),
        ("Diluted EPS total", cells(eps, kind="eps")),
        ("Diluted EPS continuing", cells(cont_eps, kind="eps")),
        ("Operating cash flow (total)", cells(ocf, names_for_d="cf")),
        ("Operating cash flow (continuing)", cells(ocfc, names_for_d="cf")),
        ("Capex (spend)", cells(capex, names_for_d="cf")),
        ("**Free cash flow**", cells(fcf, names_for_d="cf")),
        ("Investing cash flow", cells(lookup_row(cfq, "Investing Cash Flow") or [], names_for_d="cf")),
        ("Financing cash flow", cells(lookup_row(cfq, "Financing Cash Flow") or [], names_for_d="cf")),
        ("D&A", cells(da, names_for_d="cf")),
        ("SBC", cells(sbc, names_for_d="cf")),
        ("Buybacks", cells(buy, names_for_d="cf")),
        ("Dividends paid", cells(divs, names_for_d="cf")),
        ("Equity raised", cells(equity_raised, names_for_d="cf")),
        ("Cash + STI (end of period)", cells(cash_sti, names_for_d="bs")),
        ("Inventory", cells(inventory, names_for_d="bs")),
        ("Receivables", cells(receivables, names_for_d="bs")),
        ("Total assets", cells(assets, names_for_d="bs")),
        ("Current assets", cells(cass, names_for_d="bs")),
        ("Total liabilities", cells(liab, names_for_d="bs")),
        ("Current liabilities", cells(cliab, names_for_d="bs")),
        ("Borrowings (end)", cells(borrow, names_for_d="bs")),
        ("Equity attributable (end)", cells(equity, names_for_d="bs")),
        ("Diluted average shares", cells(shares_dil, kind="shares")),
    ]
    q_table = _md_table(qh[:n], q_rows)

    ah = _q_headers(isa or cfa, month, naming)
    def a_cells(table, key, kind="amount"):
        vals = lookup_row(table, key) or []
        return [_cell(vals, i, currency, kind=kind) for i in range(len(ah))]
    a_fcf = None
    a_cap: list[float | None] = []
    if cfa is not None:
        a_ocf = lookup_row(cfa, "Operating Cash Flow") or []
        a_cap = _as_spend(lookup_row(cfa, "Capex") or [])
        a_fcf = _pair_sub(a_ocf, a_cap)
    a_rows = [
        ("Revenue", a_cells(isa, "Revenue")),
        ("Gross profit", a_cells(isa, "Gross Profit")),
        ("Operating income", a_cells(isa, "Operating Income As Reported") if lookup_row(isa, "Operating Income As Reported") else a_cells(isa, "Operating Income")),
        ("Net income attrib., continuing", a_cells(isa, "Net Income Continuing")),
        ("Net income, discontinued ops", a_cells(isa, "Net Income Discontinued")),
        ("**Net income attrib., total**", a_cells(isa, "Net Income")),
        ("Diluted EPS total", a_cells(isa, "Diluted EPS", kind="eps")),
        ("Operating cash flow", a_cells(cfa, "Operating Cash Flow")),
        ("Capex (spend)", [_cell(a_cap, i, currency) for i in range(len(ah))]),
        ("FCF total", [_cell(a_fcf or [], i, currency, derived=True) for i in range(len(ah))]),
        ("Buybacks", [_cell(_as_outflow_positive(lookup_row(cfa, "Buybacks") or []), i, currency) for i in range(len(ah))]),
        ("Dividends paid", [_cell(_as_outflow_positive(lookup_row(cfa, "Dividends Paid") or []), i, currency) for i in range(len(ah))]),
        ("D&A", a_cells(cfa, "D&A")),
    ]
    a_table = _md_table(ah, a_rows)

    eps_table = _eps_section(inputs, isq, eps, cont_eps, month, naming)
    val_table = _valuation_section(inputs, currency, metrics)
    cap_table = _capital_section(qh[:4], buy, divs, ttm_buy, ttm_div, currency, ttm_fcfc)
    insider = _insider_section(inputs)
    flags_md = "## 9. Data-quality flags\n" + "\n".join(f"- {f}" for f in flags)
    appendix = _appendix(inputs, isq, cfq, bsq, isa, cfa, bsa, currency, month, naming,
                         fcf, ebitda, borrow, cash_sti)
    subsequent = _subsequent_section(inputs)

    core = f"""# Fundamentals packet: {ticker}{f" ({inputs.name})" if inputs.name else ""}

{header}
{key}
## 2. Standalone quarterly ({unit_name}, latest first). D = derived (Q4 = FY − 9M, or quarter = YTD − prior YTD)
{q_table}

Precomputed streaks: positive gross profit for **{gp_streak} consecutive quarters**; positive FCF for **{fcf_streak}**.
QoQ revenue: {_pct(yoy_pair(rev[0], rev[1]) if rev and len(rev) > 1 else None)} latest vs prior quarter.

## 3. Annual ({unit_name}, latest first)
{a_table}

{eps_table}

{val_table}

{cap_table}

{insider}

{subsequent}

{flags_md}
"""
    full = core + "\n" + appendix
    numbers = _collect_numbers(full)
    return full, summary, numbers


def _eps_section(inputs, isq, eps, cont_eps, month, naming) -> str:
    lines = ["## 4. EPS: GAAP/IFRS vs street (Yahoo reported is street/adjusted, labelled as such)",
             "| Report date | Quarter | GAAP/IFRS diluted EPS | Yahoo \"reported\" EPS (street / adjusted) | Estimate | Surprise |",
             "|---|---|---|---|---|---|"]
    for i, row in enumerate(inputs.surprises[:4]):
        label = period_label(isq.columns[i], month, naming) if isq and i < len(isq.columns) else row.day
        ga = f"{eps[i]:,.2f}" if eps and i < len(eps) and eps[i] is not None else "–"
        ce = f"{cont_eps[i]:,.2f}" if cont_eps and i < len(cont_eps) and cont_eps[i] is not None else "–"
        street = f"{row.reported:.2f}" if row.reported is not None else "–"
        warn = ""
        if (row.reported is not None and eps and i < len(eps) and eps[i] is not None
                and abs(row.reported - eps[i]) > 0.02):
            warn = " ⚠ basis differs (street/adjusted vs GAAP)"
        est = f"{row.estimate:.2f}" if row.estimate is not None else "–"
        sur = f"{row.surprise_pct:+.1f}%" if row.surprise_pct is not None else "–"
        lines.append(f"| {row.day} | {label} | {ga} (continuing {ce}) | {street}{warn} | {est} | {sur} |")
    if len(lines) == 3:
        lines.append("| n/a | | | | | |")
    return "\n".join(lines)


def _currency_blurb(inputs: PacketInputs, report: str, unit_name: str) -> str:
    quote = normalize_ccy((inputs.price.currency if inputs.price else inputs.currency_price) or "USD") or "USD"
    val_ccy = major_unit(quote)
    bits = [f"Statements in {report} ({unit_name})", f"share price in {quote}"]
    if quote in _MINOR_TO_MAJOR:
        major, scale = _MINOR_TO_MAJOR[quote]
        bits.append(f"FX used: {scale:.0f} {quote} = 1 {major} (minor quote unit)")
        if report != val_ccy:
            bits.append(_fx_used_text(inputs, report, val_ccy))
    elif report == val_ccy:
        bits.append("FX used: none (same currency)")
    else:
        bits.append(_fx_used_text(inputs, report, val_ccy))
    return "; ".join(bits)


def _fx_used_text(inputs: PacketInputs, report: str, val_ccy: str) -> str:
    rate = _fx_report_per_val(inputs)
    if rate is None:
        return "FX used: unavailable (valuation ratios omitted)"
    day = inputs.fx_day or inputs.as_of
    source = inputs.fx_source or (inputs.fx_pair or "yfinance")
    return f"FX used: 1 {val_ccy} = {rate:,.4f} {report} on {day} ({source})"


def _yahoo_market_cap(info: dict) -> float | None:
    return _maybe_float(
        info.get("Market Cap__num") or info.get("Market Cap") or info.get("marketCap")
    )


def _as_share_count(value) -> float | None:
    number = _maybe_float(value)
    if number is None or number <= 0:
        return None
    # Statement rows sometimes store millions; Yahoo / EDGAR native counts do not.
    return number if number > 1e4 else number * 1e6


def _statement_share_count(inputs: PacketInputs) -> float | None:
    for key in ("balance_sheet:quarterly", "balance_sheet:annual"):
        table = inputs.statements.get(key)
        if table is None:
            continue
        row = lookup_row(table, "Shares Outstanding")
        if not row:
            continue
        for value in row:
            count = _as_share_count(value)
            if count:
                return count
    return None


def _implied_share_count(inputs: PacketInputs) -> float | None:
    info = inputs.yahoo_info
    from_info = _as_share_count(
        info.get("impliedSharesOutstanding") or info.get("impliedSharesOutstanding__num")
    )
    mcap = _yahoo_market_cap(info)
    price = inputs.price
    if mcap and price and price.close:
        px = quote_to_major(price.close, price.currency)
        if px:
            return mcap / px
    return from_info


def _resolve_share_count(inputs: PacketInputs) -> tuple[float | None, str]:
    """Total shares for market cap: sum of classes, not one class.

    Yahoo ``sharesOutstanding`` and a single EDGAR context often omit a class
    (NVO A vs B). Prefer a statement total that agrees with Yahoo market cap,
    else implied shares = Yahoo market cap / last close.
    """
    info = inputs.yahoo_info
    yahoo = _as_share_count(info.get("sharesOutstanding") or info.get("Shares Outstanding__num"))
    stmt = _statement_share_count(inputs)
    implied = _implied_share_count(inputs)
    if implied:
        for candidate, source in (
            (stmt, "statement class total"),
            (yahoo, "yahoo sharesOutstanding"),
        ):
            if candidate and abs(candidate - implied) / implied <= _MCAP_SHARE_TOL:
                return candidate, source
        return implied, "Yahoo market cap / last close"
    if stmt and yahoo:
        chosen = stmt if stmt >= yahoo else yahoo
        return chosen, "statement class total" if chosen == stmt else "yahoo sharesOutstanding"
    if stmt:
        return stmt, "statement class total"
    if yahoo:
        return yahoo, "yahoo sharesOutstanding"
    return None, ""


def _share_count_flag(inputs: PacketInputs, shares: float | None, source: str) -> str | None:
    """Flag when price × shares and Yahoo market cap disagree by more than 15%."""
    yahoo_mcap = _yahoo_market_cap(inputs.yahoo_info)
    price = inputs.price
    if not yahoo_mcap or not price or not price.close or not shares:
        return None
    px = quote_to_major(price.close, price.currency)
    if not px:
        return None
    quote = major_unit(normalize_ccy(price.currency) or price.currency or "USD")
    packet_mcap = px * shares
    gap_packet = abs(packet_mcap - yahoo_mcap) / abs(yahoo_mcap)
    yahoo_sh = _as_share_count(
        inputs.yahoo_info.get("sharesOutstanding") or inputs.yahoo_info.get("Shares Outstanding__num")
    )
    yahoo_from_sh = px * yahoo_sh if yahoo_sh else None
    gap_yahoo = (
        abs(yahoo_from_sh - yahoo_mcap) / abs(yahoo_mcap) if yahoo_from_sh else None
    )
    if gap_packet > _MCAP_SHARE_TOL:
        return (
            f"SHARE COUNT: packet market cap {fmt_amount(packet_mcap, quote)} vs "
            f"Yahoo market cap {fmt_amount(yahoo_mcap, quote)} differ by {gap_packet:.0%} "
            f"(packet uses {shares / 1e6:,.1f}M shares, {source})."
        )
    if gap_yahoo is not None and gap_yahoo > _MCAP_SHARE_TOL:
        return (
            f"SHARE COUNT: Yahoo sharesOutstanding {yahoo_sh / 1e6:,.1f}M × price → "
            f"{fmt_amount(yahoo_from_sh, quote)} vs Yahoo market cap "
            f"{fmt_amount(yahoo_mcap, quote)} (differ by {gap_yahoo:.0%}). "
            f"Packet uses {shares / 1e6:,.1f}M ({source}) so share classes are not omitted."
        )
    return None


def _valuation_figures(inputs, report, ttm_rev, ttm_ni, ttm_nic, ttm_fcf, ttm_ebitda,
                       cash0, total_debt0, equity0, ttm_ceps) -> dict[str, Any]:
    price = inputs.price
    shares, share_src = _resolve_share_count(inputs)
    quote = normalize_ccy((price.currency if price else inputs.currency_price) or "USD") or "USD"
    val_ccy = major_unit(quote)
    out: dict[str, Any] = {
        "mcap": None, "ev": None, "pe_ttm": None, "ev_ebitda": None, "ev_sales": None,
        "ps": None, "pb": None, "fcf_yield": None, "cash_in_price_ccy": None,
        "net_cash_per_share": None, "shares": None, "share_source": share_src,
    }
    if price is None or shares in (None, 0):
        return out
    share_n = float(shares)
    out["shares"] = share_n
    px = quote_to_major(price.close, quote)
    mcap = px * share_n
    out["mcap"] = mcap
    same = (normalize_ccy(report) or report) == val_ccy
    if not (same or _fx_report_per_val(inputs) is not None):
        return out
    cash_v = _to_valuation(cash0, inputs, report, val_ccy)
    debt_v = _to_valuation(total_debt0, inputs, report, val_ccy)
    eq_v = _to_valuation(equity0, inputs, report, val_ccy)
    rev_v = _to_valuation(ttm_rev, inputs, report, val_ccy)
    ni_src = ttm_nic if ttm_nic and ttm_nic > 0 else ttm_ni
    ni_v = _to_valuation(ni_src, inputs, report, val_ccy)
    ebitda_v = _to_valuation(ttm_ebitda, inputs, report, val_ccy)
    fcf_v = _to_valuation(ttm_fcf, inputs, report, val_ccy)
    out["cash_in_price_ccy"] = cash_v
    if cash_v is not None and debt_v is not None:
        ev = mcap + debt_v - cash_v
        out["ev"] = ev
        if rev_v:
            out["ev_sales"] = ev / rev_v
        if ebitda_v and ebitda_v > 0:
            out["ev_ebitda"] = ev / ebitda_v
        if share_n:
            out["net_cash_per_share"] = (cash_v - debt_v) / share_n
    if rev_v:
        out["ps"] = mcap / rev_v
    if eq_v:
        out["pb"] = mcap / eq_v
    if ni_v and ni_v > 0:
        out["pe_ttm"] = mcap / ni_v
    elif ttm_ceps and ttm_ceps > 0:
        eps_v = _to_valuation(ttm_ceps, inputs, report, val_ccy)
        if eps_v:
            out["pe_ttm"] = px / eps_v
    if fcf_v is not None and mcap:
        out["fcf_yield"] = fcf_v / mcap * 100
    return out


def _valuation_section(inputs, currency, metrics) -> str:
    price = inputs.price
    info = inputs.yahoo_info
    shares = metrics.get("shares")
    share_src = metrics.get("share_source") or "snapshot"
    close = price.close if price else None
    quote = normalize_ccy((price.currency if price else inputs.currency_price) or "USD") or "USD"
    val_ccy = major_unit(quote)
    lines = ["## 5. Valuation at the point-in-time close",
             "| Metric | Value | Formula / source |",
             "|---|---|---|"]
    if close is None:
        lines.append("| Last close | n/a | |")
        return "\n".join(lines)
    lines.append(
        f"| Last close | {quote} {close:.2f} on {price.day} | "
        f"daily history; do not use the 50-day average as the price |"
    )
    if quote in _MINOR_TO_MAJOR:
        major, scale = _MINOR_TO_MAJOR[quote]
        lines.append(
            f"| Last close ({major}) | {quote_to_major(close, quote):.4f} {major} | "
            f"{quote} / {scale:.0f} |"
        )
    if shares and close:
        lines.append(
            f"| Shares outstanding | {float(shares) / 1e6:,.1f}M | "
            f"{share_src}; snapshot ⚠ not point-in-time |"
        )
    mcap = metrics.get("mcap")
    if mcap is not None:
        lines.append(f"| Market cap | {fmt_amount(mcap, val_ccy)} | price × shares ({val_ccy}) |")
    convertible = (normalize_ccy(currency) or currency) == val_ccy or _fx_report_per_val(inputs) is not None
    if not convertible:
        lines.append(
            "| EV, P/E, P/B, FCF yield | omitted | "
            "CROSS-CURRENCY: no FX on or before as-of; not mixing currencies |"
        )
    else:
        fx_note = ""
        if (normalize_ccy(currency) or currency) != val_ccy:
            rate = _fx_report_per_val(inputs)
            fx_note = f"; statements {currency} → {val_ccy} at {rate:,.4f}"
        cash_v = metrics.get("cash_in_price_ccy")
        if cash_v is not None:
            lines.append(
                f"| Cash + STI ({val_ccy}) | {fmt_amount(cash_v, val_ccy)} | "
                f"as-reported converted{fx_note} |"
            )
        if metrics.get("net_cash_per_share") is not None:
            lines.append(
                f"| Net cash per share | {metrics['net_cash_per_share']:.2f} {val_ccy} | "
                f"(cash+STI − debt) / shares{fx_note} |"
            )
        if metrics.get("ev") is not None:
            lines.append(
                f"| Enterprise value | {fmt_amount(metrics['ev'], val_ccy)} | "
                f"mcap + debt − cash+STI{fx_note} |"
            )
        if metrics.get("ev_sales") is not None:
            lines.append(f"| EV / sales TTM | {metrics['ev_sales']:.2f}x | |")
        if metrics.get("ev_ebitda") is not None:
            lines.append(f"| EV / EBITDA TTM | {metrics['ev_ebitda']:.1f}x | |")
        if metrics.get("ps") is not None:
            lines.append(f"| Price / sales TTM | {metrics['ps']:.2f}x | |")
        if metrics.get("pb") is not None:
            lines.append(f"| Price / book | {metrics['pb']:.2f}x | |")
        pe = metrics.get("pe_ttm")
        lines.append(
            f"| P/E TTM | {f'{pe:.1f}x' if pe else 'n/m (negative earnings)'} | "
            f"continuing when discontinued ≠ 0{fx_note} |"
        )
        if metrics.get("fcf_yield") is not None:
            lines.append(f"| FCF yield TTM | {metrics['fcf_yield']:.1f}% | |")
    if price and price.high_52w:
        dist_h = (close / price.high_52w - 1) * 100
        dist_l = (close / price.low_52w - 1) * 100 if price.low_52w else None
        lines.append(f"| 52-week range | {price.low_52w:.2f}–{price.high_52w:.2f} | close {dist_h:+.1f}% vs high"
                     + (f", {dist_l:+.1f}% vs low" if dist_l is not None else ""))
    if price and price.ma50:
        lines.append(
            f"| 50-day / 200-day average | {price.ma50:.2f} / {price.ma200 or float('nan'):.2f} | "
            f"close vs 50-day {(close / price.ma50 - 1) * 100:+.1f}%; not a substitute for last close |"
        )
    beta = info.get("Beta") or info.get("Beta__num")
    if beta:
        lines.append(f"| Beta | {beta} | yfinance snapshot |")
    fwd = info.get("Forward EPS__num") or info.get("Forward EPS")
    if fwd:
        lines.append(f"| Forward EPS | {fwd} | Yahoo consensus (street), labelled as such |")
    return "\n".join(lines)


def _capital_section(headers, buy, divs, ttm_buy, ttm_div, currency, ttm_fcfc) -> str:
    total = (ttm_buy or 0) + (ttm_div or 0)
    if not any(v for v in (buy or [])[:4]) and not any(v for v in (divs or [])[:4]):
        return ("## 6. Capital returns\nNone in the periods shown. Financing inflows are equity "
                "and debt raises where present; they are not treated as FCF.\n")
    cols = list(headers[:4]) + ["**TTM**"]
    rows = [
        ("Buybacks", [fmt_plain(v, currency) for v in (buy or [])[:4]] + [f"**{fmt_plain(ttm_buy, currency)}**"]),
        ("Dividends paid", [fmt_plain(v, currency) for v in (divs or [])[:4]] + [f"**{fmt_plain(ttm_div, currency)}**"]),
        ("**Total**", [fmt_plain((buy[i] or 0) + (divs[i] or 0), currency)
                       if i < len(buy) or i < len(divs) else "–" for i in range(4)]
         + [f"**{fmt_plain(total, currency)}**"]),
    ]
    return "## 6. Capital returns\n" + _md_table(cols, rows) + "\n"


def _insider_section(inputs: PacketInputs) -> str:
    ins = inputs.insiders
    if ins is None or ins.raw_missing:
        return ("## 7. Insider / ownership\nNot available (foreign filers have no Form 4, "
                "or the feed was empty / withheld).\n")
    lines = ["## 7. Insider / ownership (summary; replaces the raw Form 4 dump)"]
    if ins.top:
        lines.append("Largest rows:")
        lines.extend(f"- {t}" for t in ins.top)
    else:
        lines.append("No material Form 4 rows in the window.")
    return "\n".join(lines) + "\n"


def _subsequent_section(inputs: PacketInputs) -> str:
    period_end = latest_period_end(inputs)
    since = period_end or "latest period"
    failed = any(f.startswith("SUBSEQUENT EVENTS:") for f in inputs.extra_flags)
    events = inputs.subsequent_events or []
    lines = [f"## 8. Subsequent events (since {since})"]
    if failed and not events:
        lines.append(
            "Omitted: 8-K filings could not be retrieved. See data-quality flags."
        )
        return "\n".join(lines)
    if not events:
        lines.append(
            "None in the window (filings after period end, on or before the as-of date)."
        )
        lines.append(
            "Subsequent raises are not mixed into period FCF. Runway as-reported uses "
            "cash+STI at period end."
        )
        return "\n".join(lines)
    for event in events:
        lines.append(f"- {event.display_line()}")
    lines.append(
        "Subsequent cash inflows are not mixed into period FCF. Pro forma runway "
        "adds only confident equity/net proceeds and is labelled D."
    )
    return "\n".join(lines)


def latest_period_end(inputs: PacketInputs) -> str | None:
    ends: list[str] = []
    for table in inputs.statements.values():
        for col in table.columns:
            if col.end and re.match(r"\d{4}-\d{2}-\d{2}", col.end):
                ends.append(col.end[:10])
    return max(ends) if ends else None


def _appendix(inputs, isq, cfq, bsq, isa, cfa, bsa, currency, month, naming,
              fcf, ebitda, borrow, cash_sti) -> str:
    parts = [
        "## 10. Statements appendix (standalone, latest first; – = not filed / not tagged)",
        "Sign convention: appendix cash-flow capex is as the source reports it "
        "(Yahoo: typically negative outflow; EDGAR: typically positive spend). "
        "Core Capex (spend) rows are normalized to positive spend.",
    ]
    for title, table, extra in (
        ("10.1 Balance sheet, quarterly", bsq, {"Cash + STI": cash_sti, "Borrowings": borrow}),
        ("10.2 Income statement, quarterly", isq, {"EBITDA (D)": ebitda}),
        ("10.3 Cash flow, quarterly", cfq, {"FCF (D)": fcf}),
        ("10.4 Income statement, annual", isa, {}),
        ("10.5 Cash flow, annual", cfa, {}),
        ("10.6 Balance sheet, annual", bsa, {}),
    ):
        if table is None:
            continue
        headers = _q_headers(table, month, naming)
        rows = []
        seen_vals: list[list[float | None]] = []
        for name in table.rows:
            if _drop_row(name, table.source):
                continue
            vals = table.native_row(name)
            if _duplicate_of(vals, seen_vals):
                continue
            seen_vals.append(vals)
            kind = "eps" if _is_eps(name) else ("shares" if _is_shares(name) else "amount")
            label = name
            if name.lower() in {"capital expenditure", "capex"}:
                conv = ("Yahoo typically negative outflow" if table.source == "yahoo"
                        else "EDGAR typically positive spend")
                label = f"{name} (source sign; {conv})"
            rows.append((label, [_cell(vals, i, currency, kind=kind) for i in range(len(headers))]))
        for name, vals in extra.items():
            if vals:
                rows.append((name, [_cell(vals, i, currency, derived=True) for i in range(len(headers))]))
        parts.append(f"### {title} ({unit_label(currency)})")
        parts.append(_md_table(headers, rows))
    return "\n\n".join(parts) + "\n"


def _drop_row(name: str, source: str) -> bool:
    return source == "yahoo" and name in _YAHOO_DROP


def _duplicate_of(vals: list[float | None], seen: list[list[float | None]]) -> bool:
    compact = tuple(None if v is None else round(v, 4) for v in vals)
    if all(v is None for v in compact):
        return True
    for other in seen:
        other_c = tuple(None if v is None else round(v, 4) for v in other)
        if compact == other_c:
            return True
    return False


def _collect_numbers(markdown: str) -> list[float]:
    found: list[float] = []
    for match in re.finditer(
        r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+\.\d+|[-+]?\d+", markdown
    ):
        try:
            found.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue
    return found


# --- ETF -------------------------------------------------------------------

def _build_fund(inputs: PacketInputs) -> Packet:
    fund = inputs.fund or FundSnapshot(name=inputs.name, quote_type=inputs.quote_type)
    price = inputs.price
    last = f"{price.currency} {price.close:.2f} on {price.day}" if price else "n/a"
    holdings = "\n".join(
        f"| {n} | {'' if w is None else f'{w:.2f}%'} |" for n, w in (fund.top_holdings or [])[:10]
    ) or "| n/a | |"
    md = f"""# Fund packet: {inputs.ticker}{f" ({fund.name})" if fund.name else ""}

This instrument is a **{fund.quote_type or 'fund'}**, not an operating company.
Equity multiples (P/E, book value, statements) are not meaningful here and are omitted.

## 0. Header
| Field | Value |
|---|---|
| As-of date | {inputs.as_of} |
| Last close | **{last}** |
| Issuer / family | {fund.family or "n/a"} |
| Category | {fund.category or "n/a"} |
| Inception | {fund.inception or "n/a"} |
| Sources | yfinance funds_data. ⚠ NOT POINT-IN-TIME (live snapshot) |

## 1. Fund metrics
| Metric | Value | Period | Basis |
|---|---|---|---|
| Expense ratio | {'' if fund.expense_ratio is None else f'{fund.expense_ratio:.2f}%'} | current | |
| AUM | {'' if fund.aum is None else f'{fund.aum:,.0f}'} | current | |
| Distribution yield (TTM) | {'' if fund.yield_ttm is None else f'{fund.yield_ttm:.2f}%'} | TTM | |
| Holdings count | {fund.holdings_count or "n/a"} | | |
| Duration | {'' if fund.duration is None else f'{fund.duration:.2f}'} | | bond funds |
| YTM | {'' if fund.ytm is None else f'{fund.ytm:.2f}%'} | | bond funds |
| NAV | {'' if fund.nav is None else f'{fund.nav:.2f}'} | | |
| 52-week range | {f'{price.low_52w:.2f}–{price.high_52w:.2f}' if price and price.high_52w else 'n/a'} | | |

## 2. Top holdings
| Holding | Weight |
|---|---|
{holdings}

## 9. Data-quality flags
- NOT POINT-IN-TIME: fund profile fields are a current Yahoo snapshot.
- No income statement, balance sheet, P/E or book value: those figures are undefined or nonsense for this quote type.
"""
    summary = md.split("## 2.")[0]
    return Packet(
        ticker=inputs.ticker, as_of=inputs.as_of, markdown=md, summary=summary,
        numbers=_collect_numbers(md), quote_type=fund.quote_type or "ETF",
        metrics={"expense_ratio": fund.expense_ratio, "yield_ttm": fund.yield_ttm},
    )


# --- live fetch ------------------------------------------------------------

def render_fundamentals_packet(ticker: str, as_of: str | None) -> str:
    """Build and render the packet for ``ticker`` as of ``as_of`` (YYYY-MM-DD)."""
    as_of = as_of or date.today().isoformat()
    try:
        inputs = fetch_packet_inputs(ticker, as_of)
    except NoMarketDataError:
        raise
    except VendorUnavailableError:
        raise
    packet = build_packet(inputs)
    return packet.markdown


def fetch_packet_inputs(ticker: str, as_of: str) -> PacketInputs:
    """Gather vendor data. Statements prefer EDGAR, then Yahoo."""
    from tradingagents.dataflows.date_window import withhold_live_profile
    from tradingagents.dataflows.vendors.sec_edgar import statement_table as edgar_table
    from tradingagents.dataflows.vendors.yahoo import fundamentals as yahoo
    from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

    info: dict[str, Any] = {}
    try:
        if withhold_live_profile(as_of, ticker) is None:
            info = yahoo.get_company_profile(ticker) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("profile fetch failed for %s: %s", ticker, exc)

    quote_type = str(info.get("quoteType") or "EQUITY")
    price_ccy = quote_currency(ticker, info)
    report_ccy = normalize_ccy(info.get("financialCurrency")) or ""
    inputs = PacketInputs(
        ticker=ticker, as_of=as_of, name=info.get("longName") or info.get("shortName"),
        quote_type=quote_type, country=info.get("country"),
        currency_price=price_ccy,
        currency_report=report_ccy or "USD",
        yahoo_info=_info_to_map(info),
    )
    if quote_type.upper() in _ETF_TYPES:
        inputs.fund = _fetch_fund(ticker, info)
        inputs.price = _price_from_history(ticker, as_of, load_ohlcv, currency=price_ccy)
        return inputs

    for kind in ("balance_sheet", "income_statement", "cashflow"):
        for freq in ("quarterly", "annual"):
            table = None
            try:
                table = edgar_table(kind, ticker, freq, as_of)
            except (NoMarketDataError, VendorUnavailableError, Exception) as exc:
                logger.debug("EDGAR %s %s %s: %s", ticker, kind, freq, exc)
                try:
                    yahoo_ccy = report_ccy or "USD"
                    table = yahoo.statement_table(
                        ticker, kind, freq, as_of, currency=yahoo_ccy,
                    )
                except (NoMarketDataError, VendorUnavailableError, Exception) as exc2:
                    logger.debug("Yahoo %s %s %s: %s", ticker, kind, freq, exc2)
            if table is not None:
                inputs.statements[f"{kind}:{freq}"] = table

    if not inputs.statements:
        raise NoMarketDataError(ticker, ticker, "no statements available for fundamentals packet")

    inputs.currency_report = reporting_currency(inputs)
    inputs.price = _price_from_history(ticker, as_of, load_ohlcv, currency=inputs.currency_price)
    _attach_fx(inputs, as_of, load_ohlcv)
    if ticker.upper() in {"SONY", "SNE"}:
        try:
            inputs.home_price = _price_from_history("6758.T", as_of, load_ohlcv, currency="JPY")
        except Exception:  # noqa: BLE001
            inputs.home_price = None
    try:
        earn = yahoo.get_earnings_context(ticker, as_of)
        inputs.next_earnings, inputs.surprises = _parse_earnings(earn)
    except Exception as exc:  # noqa: BLE001
        logger.debug("earnings context failed: %s", exc)
    try:
        raw = yahoo.get_insider_transactions(ticker, as_of)
        inputs.insiders = _parse_insiders(raw if isinstance(raw, str) else "")
    except Exception as exc:  # noqa: BLE001
        logger.debug("insiders failed: %s", exc)
        inputs.insiders = InsiderSummary(raw_missing=True)
    try:
        events, flags = load_subsequent_events(ticker, as_of, latest_period_end(inputs))
        inputs.subsequent_events = events
        inputs.extra_flags.extend(flags)
    except Exception as exc:  # noqa: BLE001
        logger.debug("subsequent events failed: %s", exc)
        inputs.extra_flags.append(
            f"SUBSEQUENT EVENTS: 8-K fetch failed ({type(exc).__name__}); section omitted"
        )
    return inputs


def _info_to_map(info: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    mapping = {
        "freeCashflow": "Free Cash Flow",
        "netIncomeToCommon": "Net Income",
        "totalRevenue": "Revenue (TTM)",
        "grossProfits": "Gross Profit",
        "ebitda": "EBITDA",
        "operatingMargins": "Operating Margin",
        "trailingPE": "PE Ratio (TTM)",
        "forwardPE": "Forward PE",
        "forwardEps": "Forward EPS",
        "trailingEps": "EPS (TTM)",
        "beta": "Beta",
        "dividendYield": "Dividend Yield",
        "fiftyTwoWeekHigh": "52 Week High",
        "fiftyTwoWeekLow": "52 Week Low",
        "fiftyDayAverage": "50 Day Average",
        "twoHundredDayAverage": "200 Day Average",
        "sharesOutstanding": "sharesOutstanding",
        "impliedSharesOutstanding": "impliedSharesOutstanding",
        "marketCap": "Market Cap",
        "priceToBook": "Price to Book",
        "bookValue": "Book Value",
    }
    for src, label in mapping.items():
        if src in info and info[src] is not None:
            out[label] = info[src]
            num = _maybe_float(info[src])
            if num is not None:
                out[f"{label}__num"] = num
    if "sharesOutstanding" in info:
        out["sharesOutstanding"] = info["sharesOutstanding"]
    if "impliedSharesOutstanding" in info:
        out["impliedSharesOutstanding"] = info["impliedSharesOutstanding"]
    return out


def _price_from_history(ticker: str, as_of: str, load_ohlcv, currency: str = "USD") -> PricePoint | None:
    try:
        frame = load_ohlcv(ticker, as_of, fill_gaps=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("ohlcv failed for %s: %s", ticker, exc)
        return None
    if frame is None or getattr(frame, "empty", True):
        return None
    import pandas as pd

    work = frame.copy()
    if "Date" not in work.columns:
        work = work.reset_index()
        if "Date" not in work.columns and "index" in work.columns:
            work = work.rename(columns={"index": "Date"})
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date"]).sort_values("Date")
    cutoff = pd.Timestamp(as_of)
    work = work[work["Date"] <= cutoff]
    if work.empty or "Close" not in work.columns:
        return None
    last = work.iloc[-1]
    prev = work.iloc[-2] if len(work) > 1 else None
    year = work[work["Date"] >= cutoff - pd.Timedelta(days=365)]
    window50 = work.tail(50)["Close"]
    window200 = work.tail(200)["Close"]
    return PricePoint(
        close=float(last["Close"]),
        day=last["Date"].strftime("%Y-%m-%d"),
        previous=float(prev["Close"]) if prev is not None else None,
        previous_day=prev["Date"].strftime("%Y-%m-%d") if prev is not None else None,
        currency=normalize_ccy(currency) or currency or "USD",
        high_52w=float(year["High"].max()) if "High" in year.columns and not year.empty else None,
        low_52w=float(year["Low"].min()) if "Low" in year.columns and not year.empty else None,
        ma50=float(window50.mean()) if len(window50) >= 20 else None,
        ma200=float(window200.mean()) if len(window200) >= 50 else None,
    )


def _fx_rate(symbol: str, as_of: str, load_ohlcv) -> tuple[float | None, str]:
    pt = _price_from_history(symbol, as_of, load_ohlcv)
    return (pt.close if pt else None), symbol


def _attach_fx(inputs: PacketInputs, as_of: str, load_ohlcv) -> None:
    report = reporting_currency(inputs)
    val_ccy = major_unit(inputs.currency_price)
    if report == val_ccy:
        return
    rate, pair, day, source = _lookup_fx(report, val_ccy, as_of, load_ohlcv)
    inputs.fx_usd = rate
    inputs.fx_pair = pair
    inputs.fx_day = day
    inputs.fx_source = source


def _lookup_fx(
    report: str, val: str, as_of: str, load_ohlcv,
) -> tuple[float | None, str | None, str | None, str | None]:
    """Reporting units per 1 valuation unit, as of ``as_of`` (no look-ahead)."""
    if report == val:
        return 1.0, f"{val}/{report}", as_of, "identity"

    def close_of(symbol: str) -> PricePoint | None:
        try:
            return _price_from_history(symbol, as_of, load_ohlcv)
        except Exception:  # noqa: BLE001
            return None

    # VALREPORT=X (USDCNY=X) is report per val.
    pt = close_of(f"{val}{report}=X")
    if pt and pt.close:
        return pt.close, f"{val}/{report}", pt.day, f"yfinance {val}{report}=X"
    # REPORTVAL=X (EURUSD=X) is val per report — invert.
    pt = close_of(f"{report}{val}=X")
    if pt and pt.close:
        return 1.0 / pt.close, f"{val}/{report}", pt.day, f"yfinance {report}{val}=X"

    report_per_usd, report_day, report_src = _ccy_per_usd(report, as_of, close_of)
    val_per_usd, val_day, val_src = _ccy_per_usd(val, as_of, close_of)
    if report_per_usd and val_per_usd:
        day = report_day or val_day
        src = report_src if val == "USD" else (val_src if report == "USD" else f"{report_src}; {val_src}")
        return report_per_usd / val_per_usd, f"{val}/{report}", day, src
    return None, None, None, None


def _ccy_per_usd(ccy: str, as_of: str, close_of) -> tuple[float | None, str | None, str | None]:
    if ccy == "USD":
        return 1.0, as_of, "identity"
    pt = close_of(f"{ccy}=X")
    if not pt or not pt.close:
        pt = close_of(f"USD{ccy}=X" if ccy not in _USD_IS_QUOTE else f"{ccy}USD=X")
    if not pt or not pt.close:
        return None, None, None
    if ccy in _USD_IS_QUOTE:
        # EUR=X / EURUSD=X: USD per 1 EUR → EUR per USD is the inverse.
        return 1.0 / pt.close, pt.day, f"yfinance {ccy}=X"
    return pt.close, pt.day, f"yfinance {ccy}=X"


def _fetch_fund(ticker: str, info: dict) -> FundSnapshot:
    import yfinance as yf

    from tradingagents.dataflows.vendors.yahoo.common import yf_retry

    snap = FundSnapshot(
        name=info.get("longName") or info.get("shortName"),
        family=info.get("fundFamily") or info.get("family"),
        category=info.get("category"),
        expense_ratio=_maybe_expense(
            info.get("annualReportExpenseRatio") or info.get("expenseRatio")
            or info.get("netExpenseRatio")
        ),
        aum=info.get("totalAssets"),
        yield_ttm=_maybe_pct(info.get("yield") or info.get("trailingAnnualDividendYield")),
        nav=info.get("navPrice") or info.get("previousClose"),
        quote_type=str(info.get("quoteType") or "ETF"),
    )
    try:
        funds = yf_retry(lambda: yf.Ticker(ticker).funds_data)
    except Exception as exc:  # noqa: BLE001
        logger.debug("funds_data failed: %s", exc)
        return snap
    if funds is None:
        return snap
    overview = getattr(funds, "fund_overview", None)
    if isinstance(overview, dict):
        snap.expense_ratio = snap.expense_ratio or _maybe_expense(
            overview.get("AnnualReportExpenseRatio")
        )
        snap.family = snap.family or overview.get("Family")
        snap.category = snap.category or overview.get("CategoryName")
    bond = getattr(funds, "bond_holdings", None)
    if isinstance(bond, dict):
        snap.duration = _maybe_float(bond.get("duration") or bond.get("Duration"))
        snap.ytm = _maybe_pct(bond.get("maturity") or bond.get("YTM") or bond.get("yieldToMaturity"))
    holdings = getattr(funds, "top_holdings", None)
    if holdings is not None and hasattr(holdings, "iterrows"):
        snap.top_holdings = _holdings_from_frame(holdings)
        snap.holdings_count = snap.holdings_count or len(holdings)
    return snap


def _holdings_from_frame(holdings) -> list[tuple[str, float | None]]:
    """yfinance ``holdingPercent`` is a fraction (0.05 = 5%); store as percent."""
    rows: list[tuple[str, float | None]] = []
    for idx, row in holdings.head(10).iterrows():
        weight = None
        for col in row.index:
            name = str(col).lower()
            if "holding" in name or "percent" in name or "weight" in name:
                weight = _maybe_pct(row[col])
                break
        if weight is None and len(row):
            weight = _maybe_pct(row.iloc[0])
        rows.append((str(idx), weight))
    return rows


def _maybe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_pct(value) -> float | None:
    number = _maybe_float(value)
    if number is None:
        return None
    return number * 100 if abs(number) <= 1 else number


def _maybe_expense(value) -> float | None:
    """Expense ratio: yfinance mixes fractions and already-percent values.

    ``annualReportExpenseRatio`` is a fraction (0.0015 → 0.15%).
    ``netExpenseRatio`` is already percent (TLT 0.15, TCHI 0.59). Values
    below 0.05 are treated as fractions; 0.15 must stay 0.15%, not 15%.
    """
    number = _maybe_float(value)
    if number is None:
        return None
    return number * 100 if abs(number) < 0.05 else number


def packet_summary_markdown(markdown: str) -> str:
    """§0–§1 of a rendered packet, for debate agents."""
    if "## 2." in markdown:
        return markdown.split("## 2.", 1)[0].strip()
    return markdown[:2000]
