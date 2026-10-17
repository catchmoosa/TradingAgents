"""Subsequent events from Form 8-K, extracted without sending filing text to the LLM.

US filers only. The packet lists 8-K / 8-K/A filings after the latest statement
period end and on or before the as-of date. Money facts (net proceeds, shares,
price, debt principal) come from conservative regexes on the primary document
and exhibit 99.1. Uncertain amounts are omitted; the event still appears with
its item codes. Failures never block the packet: they become a data-quality flag.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors import sec_edgar

logger = logging.getLogger(__name__)

_PACIFIC = ZoneInfo("America/Los_Angeles")
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}"

_MONEY_ITEMS = frozenset({"1.01", "2.01", "2.03", "3.02", "8.01"})
_NOTE_ITEMS = frozenset({"2.02"})
_EIGHT_K = frozenset({"8-K", "8-K/A"})
_MAX_MONEY_FETCH = 12

_ITEM_RE = re.compile(r"(?:Item\s*)?(\d+\.\d{2})", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"(?is)<script.*?</script>")
_STYLE_RE = re.compile(r"(?is)<style.*?</style>")
_WS_RE = re.compile(r"\s+")
_HEDGE_RE = re.compile(r"\b(?:expect(?:s|ed)?|contingent|up to)\b", re.I)
_DOC_RE = re.compile(r"(?is)<DOCUMENT>(.*?)</DOCUMENT>")
_TYPE_RE = re.compile(r"(?i)<TYPE>\s*([^\r\n<]+)")
_TEXT_RE = re.compile(r"(?is)<TEXT>(.*)</TEXT>")

_DOLLAR = r"(?:US)?\$\s*([\d,.]+)\s*(million|millions|billion|billions|bn)?"
_NET_BEFORE = re.compile(rf"net proceeds.{{0,160}}?{_DOLLAR}", re.I)
_NET_AFTER = re.compile(
    r"(?:US)?\$\s*([\d,.]+)\s*(million|millions|billion|billions|bn)"
    r"[^.?!]{0,80}net proceeds",
    re.I,
)
_SHARES_PRICE = re.compile(
    r"([\d,.]+)\s*(million|billion)?\s+shares.{0,140}?(?:US)?\$\s*([\d.]+)\s*per share",
    re.I,
)
_PRINCIPAL = re.compile(
    rf"(?:aggregate\s+)?principal(?:\s+amount)?.{{0,80}}?{_DOLLAR}",
    re.I,
)
_ISSUED_NOTES = re.compile(
    rf"issued.{{0,60}}?{_DOLLAR}.{{0,60}}?(?:notes|bonds|debentures)",
    re.I,
)
_EQUITY_CTX = re.compile(
    r"\b(common stock|ordinary shares|equity|underwritten|public offering|"
    r"follow[- ]on|at[- ]the[- ]market|registered direct)\b",
    re.I,
)
_DEBT_CTX = re.compile(
    r"\b(notes|debentures|credit (?:facility|agreement|line)|term loan|"
    r"indenture|senior notes|revolving)\b",
    re.I,
)


@dataclass
class MoneyFacts:
    amount: float | None = None
    shares: float | None = None
    price_per_share: float | None = None
    kind: str = "other"
    cash_inflow: bool = False


@dataclass
class SubsequentEvent:
    filed: str
    items: list[str] = field(default_factory=list)
    title: str = ""
    accession: str = ""
    url: str = ""
    kind: str = "other"
    amount: float | None = None
    shares: float | None = None
    price_per_share: float | None = None
    currency: str = "USD"
    cash_inflow: bool = False
    note: str | None = None

    def display_line(self) -> str:
        item_str = "/".join(self.items) if self.items else "?"
        head = self._headline()
        extra = ""
        if self.shares and self.price_per_share:
            share_n = self.shares / 1e6 if self.shares >= 1e4 else self.shares
            extra = f"; {share_n:,.1f}M shares at ${self.price_per_share:.2f}"
        link = f"[{self.accession}]({self.url})" if self.url else self.accession
        title = f" — {self.title}" if self.title else ""
        return f"{self.filed}: {head} (8-K items {item_str}){extra}{title} {link}"

    def _headline(self) -> str:
        if self.kind == "equity_raise" and self.amount is not None:
            return f"equity raise {self._fmt_amount()}"
        if self.kind == "debt" and self.amount is not None:
            return f"debt incurred {self._fmt_amount()} (not added to cash)"
        if self.kind == "earnings_release":
            return "earnings release (no figures taken from this filing)"
        if self.amount is not None:
            return self._fmt_amount()
        return self.title or "8-K"

    def _fmt_amount(self) -> str:
        if self.amount is None:
            return ""
        return f"${self.amount / 1e6:,.0f}M"


def filed_pacific_date(filing_date: str, acceptance_datetime: str | None) -> str:
    """Pacific calendar date of the filing, for look-ahead against as_of.

    ``acceptanceDateTime`` (UTC) converted to America/Los_Angeles when present;
    otherwise the SEC ``filingDate``.
    """
    raw = (acceptance_datetime or "").strip()
    if raw:
        try:
            iso = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_PACIFIC)
            return dt.astimezone(_PACIFIC).date().isoformat()
        except ValueError:
            logger.debug("unreadable acceptanceDateTime %r", raw)
    return (filing_date or "")[:10]


def parse_item_codes(raw: str | list | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        raw = ",".join(str(x) for x in raw)
    seen: list[str] = []
    for match in _ITEM_RE.finditer(str(raw)):
        code = match.group(1)
        if code not in seen:
            seen.append(code)
    return seen


def plain_text(blob: str) -> str:
    text = _SCRIPT_RE.sub(" ", blob or "")
    text = _STYLE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def extract_exhibit_99_1(complete: str) -> str | None:
    for block in _DOC_RE.findall(complete or ""):
        kind_match = _TYPE_RE.search(block)
        kind = kind_match.group(1).strip() if kind_match else ""
        upper = kind.upper()
        if upper.startswith("EX-99.1") or upper.startswith("EXHIBIT 99.1"):
            body = _TEXT_RE.search(block)
            return body.group(1) if body else block
    return None


def extract_money_facts(text: str, items: list[str] | None = None) -> MoneyFacts:
    """Best-effort money facts. Empty amount means 'not confident'."""
    items = items or []
    blob = plain_text(text)
    if not blob:
        return MoneyFacts()

    proceeds = _first_scaled(_NET_BEFORE, blob) or _first_scaled(_NET_AFTER, blob)
    principal = _first_scaled(_PRINCIPAL, blob) or _first_scaled(_ISSUED_NOTES, blob)
    shares, price = _shares_and_price(blob)

    equity_ctx = bool(_EQUITY_CTX.search(blob))
    debt_ctx = bool(_DEBT_CTX.search(blob))
    item_set = set(items)
    debt_item = "2.03" in item_set and "3.02" not in item_set

    facts = MoneyFacts(shares=shares, price_per_share=price)

    if proceeds is not None and not debt_item and (equity_ctx or "3.02" in item_set):
        facts.amount = proceeds
        facts.kind = "equity_raise"
        facts.cash_inflow = True
        return facts
    if principal is not None or (proceeds is not None and (debt_ctx or debt_item)):
        facts.amount = principal if principal is not None else proceeds
        facts.kind = "debt"
        facts.cash_inflow = False
        return facts
    if proceeds is not None and equity_ctx:
        facts.amount = proceeds
        facts.kind = "equity_raise"
        facts.cash_inflow = True
        return facts
    return facts


def select_8k_filings(
    payload: dict,
    *,
    cik: str,
    period_end: str,
    as_of: str,
) -> list[dict]:
    """8-K / 8-K/A rows with money or 2.02 items in (period_end, as_of]."""
    rows = []
    for raw in _recent_rows(payload):
        if raw["form"].strip().upper() not in _EIGHT_K:
            continue
        items = parse_item_codes(raw["items"])
        wanted = [c for c in items if c in _MONEY_ITEMS or c in _NOTE_ITEMS]
        if not wanted:
            continue
        filed = filed_pacific_date(raw["filing_date"], raw["acceptance"])
        if not filed or not (period_end < filed <= as_of):
            continue
        rows.append({
            "filed": filed,
            "form": raw["form"].strip(),
            "items": wanted,
            "all_items": items,
            "accession": raw["accession"],
            "primary_document": raw["primary"],
            "title": raw["description"] or raw["form"].strip(),
            "url": _primary_url(cik, raw["accession"], raw["primary"]),
            "complete_url": _complete_url(cik, raw["accession"]),
            "cik": cik,
        })
    money = [r for r in rows if set(r["items"]) & _MONEY_ITEMS]
    notes = [r for r in rows if set(r["items"]) <= _NOTE_ITEMS]
    return money[:_MAX_MONEY_FETCH] + notes


def load_subsequent_events(
    ticker: str,
    as_of: str,
    period_end: str | None,
) -> tuple[list[SubsequentEvent], list[str]]:
    """Return (events, data-quality flags). Never raises to the packet builder."""
    if not period_end:
        return [], []
    try:
        cik = sec_edgar.cik_for(ticker)
    except Exception as exc:  # noqa: BLE001
        return [], [_fail_flag(exc)]
    if not cik:
        return [], []
    try:
        payload = sec_edgar._cached_json(
            _SUBMISSIONS_URL.format(cik=cik),
            f"submissions_CIK{cik}.json",
        )
    except Exception as exc:  # noqa: BLE001
        return [], [_fail_flag(exc)]
    if not isinstance(payload, dict):
        return [], [_fail_flag(ValueError("submissions JSON was not an object"))]

    try:
        filings = select_8k_filings(payload, cik=cik, period_end=period_end, as_of=as_of)
    except Exception as exc:  # noqa: BLE001
        return [], [_fail_flag(exc)]

    events: list[SubsequentEvent] = []
    for row in filings:
        events.append(_event_from_filing(row))
    events.sort(key=lambda e: (e.filed, e.accession))
    return _dedup_events(events), []


def _fail_flag(exc: BaseException) -> str:
    name = type(exc).__name__
    if isinstance(exc, VendorUnavailableError):
        name = "VendorUnavailableError"
    return f"SUBSEQUENT EVENTS: 8-K fetch failed ({name}); section omitted"


def _event_from_filing(row: dict) -> SubsequentEvent:
    items = list(row["items"])
    note_only = set(items) <= _NOTE_ITEMS and not (set(items) & _MONEY_ITEMS)
    event = SubsequentEvent(
        filed=row["filed"],
        items=items,
        title=row["title"],
        accession=row["accession"],
        url=row["url"],
        kind="earnings_release" if note_only else "other",
    )
    if note_only:
        return event
    blob = _filing_text(row)
    if not blob:
        return event
    facts = extract_money_facts(blob, items)
    event.amount = facts.amount
    event.shares = facts.shares
    event.price_per_share = facts.price_per_share
    event.kind = facts.kind if facts.kind != "other" else event.kind
    event.cash_inflow = facts.cash_inflow
    return event


def _filing_text(row: dict) -> str:
    primary = ""
    if row.get("primary_document") and row.get("url"):
        primary = _read_doc(row["url"], f"8k_{row['accession']}_{Path(row['primary_document']).name}")
    facts = extract_money_facts(primary, row.get("items") or []) if primary else MoneyFacts()
    if facts.amount is not None or not row.get("complete_url"):
        return primary
    complete = _read_doc(row["complete_url"], f"8k_{row['accession']}_complete.txt")
    exhibit = extract_exhibit_99_1(complete) if complete else None
    if exhibit:
        return exhibit
    return complete or primary


def _read_doc(url: str, cache_name: str) -> str:
    try:
        return sec_edgar._cached_text(url, cache_name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("8-K document fetch failed %s: %s", url, exc)
        return ""


def _recent_rows(payload: dict) -> list[dict]:
    recent = (payload or {}).get("filings", {}).get("recent") or {}
    accessions = recent.get("accessionNumber") or []
    n = len(accessions)

    def col(name: str, default: str = "") -> list:
        vals = recent.get(name) or []
        return [vals[i] if i < len(vals) else default for i in range(n)]

    filing_dates = col("filingDate")
    acceptances = col("acceptanceDateTime")
    forms = col("form")
    items = col("items")
    primaries = col("primaryDocument")
    descriptions = col("primaryDocDescription")
    rows = []
    for i, accession in enumerate(accessions):
        if not accession:
            continue
        rows.append({
            "accession": str(accession),
            "filing_date": str(filing_dates[i] or ""),
            "acceptance": str(acceptances[i] or "") or None,
            "form": str(forms[i] or ""),
            "items": str(items[i] or ""),
            "primary": str(primaries[i] or ""),
            "description": str(descriptions[i] or ""),
        })
    return rows


def _primary_url(cik: str, accession: str, primary: str) -> str:
    base = _ARCHIVE.format(cik_int=int(cik), acc_nodash=accession.replace("-", ""))
    if primary:
        return f"{base}/{primary.lstrip('/')}"
    return f"{base}/{accession}-index.html"


def _complete_url(cik: str, accession: str) -> str:
    base = _ARCHIVE.format(cik_int=int(cik), acc_nodash=accession.replace("-", ""))
    return f"{base}/{accession}.txt"


def _first_scaled(pattern: re.Pattern, text: str) -> float | None:
    for match in pattern.finditer(text):
        if _HEDGE_RE.search(text[max(0, match.start() - 100):match.start()]):
            continue
        groups = match.groups()
        number = _to_float(groups[0])
        unit = groups[1] if len(groups) > 1 else None
        if number is None:
            continue
        scaled = _scale_amount(number, unit)
        if scaled is not None:
            return scaled
    return None


def _shares_and_price(text: str) -> tuple[float | None, float | None]:
    if not _EQUITY_CTX.search(text):
        return None, None
    match = _SHARES_PRICE.search(text)
    if not match:
        return None, None
    if _HEDGE_RE.search(text[max(0, match.start() - 100):match.start()]):
        return None, None
    count = _to_float(match.group(1))
    unit = match.group(2)
    price = _to_float(match.group(3))
    if count is None or price is None:
        return None, None
    if unit and unit.lower().startswith("billion"):
        count *= 1e9
    elif unit and unit.lower().startswith("million"):
        count *= 1e6
    if count < 1e4 and unit is None:
        return None, None
    return count, price


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw.replace(",", "").strip())
    except ValueError:
        return None


def _scale_amount(number: float, unit: str | None) -> float | None:
    token = (unit or "").strip().lower()
    if token in {"billion", "billions", "bn"}:
        return number * 1e9
    if token in {"million", "millions", "mn"}:
        return number * 1e6
    if number >= 1e8:
        return number
    return None


def _dedup_events(events: list[SubsequentEvent]) -> list[SubsequentEvent]:
    kept: list[SubsequentEvent] = []
    for event in events:
        if event.cash_inflow and event.amount:
            duplicate = False
            for prior in kept:
                if not (prior.cash_inflow and prior.amount):
                    continue
                if abs(event.amount - prior.amount) / max(prior.amount, 1.0) > 0.05:
                    continue
                if abs(_day_delta(event.filed, prior.filed)) > 14:
                    continue
                for item in event.items:
                    if item not in prior.items:
                        prior.items.append(item)
                duplicate = True
                break
            if duplicate:
                continue
        kept.append(event)
    funded = [e for e in kept if e.cash_inflow and e.amount]
    trimmed: list[SubsequentEvent] = []
    for event in kept:
        if event.amount or event.kind in {"earnings_release", "debt"}:
            trimmed.append(event)
            continue
        redundant = False
        for fund in funded:
            if abs(_day_delta(event.filed, fund.filed)) <= 14:
                for item in event.items:
                    if item not in fund.items:
                        fund.items.append(item)
                redundant = True
                break
        if not redundant:
            trimmed.append(event)
    return trimmed


def _day_delta(a: str, b: str) -> int:
    try:
        return (date.fromisoformat(a[:10]) - date.fromisoformat(b[:10])).days
    except ValueError:
        return 10_000
