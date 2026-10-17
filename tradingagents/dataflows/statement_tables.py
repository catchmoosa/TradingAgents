"""Standalone fiscal periods: YTD differences, Q4 = FY − 9M, and company naming.

Statement vendors return mixed spans (a 10-Q cash-flow column is often 6 or 9
months). The packet never asks the model to de-cumulate those: this module
turns them into discrete quarters, flags derived cells, and labels periods in
the issuer's own fiscal convention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

# Japanese issuers (and a March year-end in Japan) name the fiscal year by its
# start: Sony FY2026 = Apr 2026–Mar 2027. US issuers name it by the end year.
_START_YEAR_TICKERS = {
    "SONY", "SNE", "TM", "HMC", "6758.T", "7203.T", "7267.T",
    "MUFG", "SMFG", "MFG", "NMR", "TAK", "HTHIY", "SFTBY", "NTDOY",
}

_END_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_QFY_RE = re.compile(r"\b(Q[1-4])\s+FY(\d{4})\b", re.IGNORECASE)
_FY_RE = re.compile(r"\bFY(\d{4})\b", re.IGNORECASE)
_MONTHS_RE = re.compile(r"\((\d+)\s+months\)", re.IGNORECASE)


@dataclass
class PeriodCol:
    end: str
    fy: int | None = None
    fp: str | None = None  # Q1, Q2, Q3, Q4, FY
    span_months: int = 3
    filed: str | None = None
    form: str | None = None
    derived: bool = False
    source_label: str = ""

    @property
    def is_instant(self) -> bool:
        return self.span_months == 0


@dataclass
class StatementTable:
    """One statement (one frequency) as labelled columns × named rows."""

    kind: str
    freq: str
    source: str
    currency: str
    scale: float
    columns: list[PeriodCol]
    rows: dict[str, list[float | None]]
    # Rows whose values are already per-share or share counts: do not apply scale.
    unscaled_rows: set[str] = field(default_factory=set)
    derived_mask: dict[str, list[bool]] = field(default_factory=dict)

    def native(self, name: str, index: int) -> float | None:
        values = self.rows.get(name)
        if values is None or index >= len(values):
            return None
        value = values[index]
        if value is None:
            return None
        if name in self.unscaled_rows:
            return float(value)
        return float(value) * self.scale

    def native_row(self, name: str) -> list[float | None]:
        return [self.native(name, i) for i in range(len(self.columns))]


def parse_period_label(label: str, *, instant: bool = False) -> PeriodCol:
    """Read an EDGAR or Yahoo column header into a period."""
    text = (label or "").strip()
    found = _END_RE.search(text)
    end = found.group(1) if found else text
    fy = None
    fp = None
    qfy = _QFY_RE.search(text)
    if qfy:
        fp = qfy.group(1).upper()
        fy = int(qfy.group(2))
    else:
        fy_only = _FY_RE.search(text)
        if fy_only:
            fp = "FY"
            fy = int(fy_only.group(1))
    months_m = _MONTHS_RE.search(text)
    months = int(months_m.group(1)) if months_m else None
    if instant:
        span = 0
    elif months:
        span = months
    elif fp == "FY":
        span = 12
    else:
        span = 3
    return PeriodCol(end=end, fy=fy, fp=fp, span_months=span, source_label=text)


def fy_end_month_from_columns(columns: list[PeriodCol], default: int = 12) -> int:
    """Month the fiscal year ends, from annual (or Q4) period-end dates."""
    for col in columns:
        if col.fp in {"FY", "Q4"} or col.span_months == 12:
            try:
                return date.fromisoformat(col.end).month
            except ValueError:
                continue
    for col in columns:
        try:
            return date.fromisoformat(col.end).month
        except ValueError:
            continue
    return default


def uses_start_year_naming(ticker: str, country: str | None, fy_end_month: int) -> bool:
    """Whether the issuer names the fiscal year by its start (Sony convention)."""
    key = ticker.strip().upper()
    if key in _START_YEAR_TICKERS or key.split(".")[0] in _START_YEAR_TICKERS:
        return True
    nation = (country or "").strip().lower()
    return nation in {"japan", "jp"} and fy_end_month == 3


def infer_fiscal(end: str, fy_end_month: int, naming: str) -> tuple[int, str]:
    """(fiscal year number, Q1–Q4/FY) for a period-end date."""
    day = date.fromisoformat(end)
    fy_end_year = day.year if day.month <= fy_end_month else day.year + 1
    start_month = (fy_end_month % 12) + 1
    months_in = (day.month - start_month) % 12
    quarter = months_in // 3 + 1
    fp = "FY" if quarter == 4 and day.month == fy_end_month else f"Q{quarter}"
    fy = fy_end_year - 1 if naming == "start" and fy_end_month != 12 else fy_end_year
    if fp.startswith("Q") and quarter == 4:
        fp = "Q4"
    return fy, fp


def fill_fiscal(columns: list[PeriodCol], fy_end_month: int, naming: str) -> list[PeriodCol]:
    """Fill missing fy/fp from the period-end date and the issuer's convention."""
    filled = []
    for col in columns:
        fy, fp = col.fy, col.fp
        if fy is None or not fp:
            inferred_fy, inferred_fp = infer_fiscal(col.end, fy_end_month, naming)
            fy = fy if fy is not None else inferred_fy
            fp = fp or inferred_fp
        filled.append(PeriodCol(
            end=col.end, fy=fy, fp=fp, span_months=col.span_months,
            filed=col.filed, form=col.form, derived=col.derived,
            source_label=col.source_label,
        ))
    return filled


def period_label(col: PeriodCol, fy_end_month: int, naming: str) -> str:
    """Display label in the company's own naming, with a month span for Japan."""
    fy, fp = col.fy, col.fp or ""
    try:
        end = date.fromisoformat(col.end)
    except ValueError:
        end = None
    mark = "†" if col.derived and fp == "Q4" else (" D" if col.derived else "")
    if fp == "FY" or (fp == "Q4" and col.span_months == 12):
        if naming == "start" and fy_end_month == 3 and fy is not None:
            return f"FY{fy} (Apr{fy % 100:02d}-Mar{(fy + 1) % 100:02d}){mark}"
        return f"FY{fy}" if fy is not None else (col.end + mark)
    if naming == "start" and fy_end_month == 3 and fy is not None and fp.startswith("Q"):
        qn = int(fp[1])
        spans = {1: "Apr-Jun", 2: "Jul-Sep", 3: "Oct-Dec", 4: "Jan-Mar"}
        year = fy if qn < 4 else fy + 1
        return f"{fp} FY{fy} ({spans[qn]} {year}){mark}"
    if fy is not None and fp:
        month = end.strftime("%b-%y") if end else col.end
        return f"{fp} FY{fy} ({month}){mark}"
    return (end.strftime("%Y-%m-%d") if end else col.end) + mark


def _sub(a: float | None, b: float | None) -> tuple[float | None, bool]:
    if a is None:
        return None, False
    if b is None:
        return a, False
    return a - b, True


def standalone_quarters(
    quarterly: StatementTable,
    annual: StatementTable | None,
    *,
    fy_end_month: int,
    naming: str,
    flow: bool,
) -> StatementTable:
    """Discrete quarters, latest first. Derived cells (YTD diffs, Q4 = FY − 9M) are flagged.

    ``flow`` is True for income and cash-flow (duration facts). Balance-sheet
    rows are point-in-time: Q4 is the annual snapshot when the quarter is absent.
    """
    qcols = fill_fiscal(quarterly.columns, fy_end_month, naming)
    # A quarterly column that ends on FYE is Q4, not FY (Yahoo date headers).
    qcols = [
        PeriodCol(
            end=c.end, fy=c.fy,
            fp="Q4" if c.fp == "FY" and c.span_months != 12 else c.fp,
            span_months=c.span_months, filed=c.filed, form=c.form,
            derived=c.derived, source_label=c.source_label,
        )
        for c in qcols
    ]
    annual_cols = fill_fiscal(annual.columns, fy_end_month, naming) if annual else []

    # Index original quarterly cells by (fy, fp, span_months, end).
    def q_value(name: str, fy: int, fp: str, span: int | None = None) -> float | None:
        for i, col in enumerate(qcols):
            if col.fy != fy or (col.fp or "") != fp:
                continue
            if span is not None and col.span_months != span:
                continue
            return quarterly.native(name, i)
        return None

    def a_value(name: str, fy: int) -> float | None:
        if annual is None:
            return None
        for i, col in enumerate(annual_cols):
            if col.fy == fy:
                return annual.native(name, i)
        return None

    # Target discrete columns: union of quarter-ends, plus derived Q4s.
    targets: list[PeriodCol] = []
    seen_ends: set[tuple[int, str]] = set()
    for col in qcols:
        if col.fy is None or not col.fp:
            continue
        fp = "Q4" if col.fp == "FY" and col.span_months != 12 else col.fp
        if fp == "FY":
            fp = "Q4"
        key = (col.fy, fp)
        if key in seen_ends:
            continue
        if col.span_months in (6, 9, 12) and fp != "Q1":
            # A YTD column still occupies that quarter's slot (to be decumulated).
            pass
        seen_ends.add(key)
        targets.append(PeriodCol(
            end=col.end, fy=col.fy, fp=fp, span_months=0 if not flow else 3,
            filed=col.filed, form=col.form, derived=False,
            source_label=col.source_label,
        ))
    have_quarters = {c.fy for c in targets if c.fp and c.fp.startswith("Q")}
    for col in annual_cols:
        if col.fy is None:
            continue
        key = (col.fy, "Q4")
        if key in seen_ends:
            continue
        # Don't slot a full-year column in as Q4 when we have no quarters for that FY.
        if col.fy not in have_quarters:
            continue
        seen_ends.add(key)
        targets.append(PeriodCol(
            end=col.end, fy=col.fy, fp="Q4", span_months=0 if not flow else 3,
            filed=col.filed, form=col.form, derived=True, source_label=col.source_label,
        ))

    def sort_key(col: PeriodCol):
        return (col.end, col.fy or 0, col.fp or "")

    targets = sorted(targets, key=sort_key, reverse=True)

    names = list(dict.fromkeys([*quarterly.rows, *(annual.rows if annual else {})]))
    new_rows: dict[str, list[float | None]] = {n: [] for n in names}
    derived_mask: dict[str, list[bool]] = {n: [] for n in names}
    out_cols: list[PeriodCol] = []

    for target in targets:
        fy, fp = target.fy, target.fp
        if fy is None or not fp:
            continue
        col_derived = False
        for name in names:
            value, was_derived = _discrete_value(
                name, fy, fp, flow, q_value, a_value, quarterly, qcols,
            )
            new_rows[name].append(value)
            derived_mask[name].append(was_derived)
            col_derived = col_derived or was_derived
        target.derived = target.derived or (flow and fp == "Q4" and col_derived)
        out_cols.append(target)

    # A column that is all-empty (e.g. Q4 FY2024 cash flow) is kept so labels
    # stay aligned; the renderer prints "–".
    unscaled = set(quarterly.unscaled_rows)
    if annual:
        unscaled |= annual.unscaled_rows
    # Values are already native; scale becomes 1.
    native_rows = {name: list(vals) for name, vals in new_rows.items()}
    return StatementTable(
        kind=quarterly.kind, freq="quarterly", source=quarterly.source,
        currency=quarterly.currency, scale=1.0, columns=out_cols,
        rows=native_rows, unscaled_rows=unscaled | set(native_rows),
        derived_mask=derived_mask,
    )


def _discrete_value(name, fy, fp, flow, q_value, a_value, quarterly, qcols):
    """One standalone quarter for ``name``."""
    if not flow:
        # Instant: prefer the quarter-end, else the annual snapshot for Q4.
        for span in (0, 3):
            got = q_value(name, fy, fp, span)
            if got is not None:
                return got, False
        got = q_value(name, fy, fp, None)
        if got is not None:
            return got, False
        if fp == "Q4":
            annual_v = a_value(name, fy)
            if annual_v is not None:
                return annual_v, True
        return None, False

    discrete = q_value(name, fy, fp, 3)
    if discrete is not None and fp != "FY":
        return discrete, False
    # Some vendors file Q2 as a discrete quarter under span 3; others only as 6M.
    if fp == "Q1":
        v = q_value(name, fy, "Q1", 3) or q_value(name, fy, "Q1", None)
        return v, False
    if fp == "Q2":
        ytd6 = q_value(name, fy, "Q2", 6) or q_value(name, fy, "Q2", None)
        q1 = q_value(name, fy, "Q1", 3) or q_value(name, fy, "Q1", None)
        if q_value(name, fy, "Q2", 3) is not None and (q_value(name, fy, "Q2", 6) is None):
            return q_value(name, fy, "Q2", 3), False
        value, derived = _sub(ytd6, q1)
        return value, derived
    if fp == "Q3":
        ytd9 = q_value(name, fy, "Q3", 9) or q_value(name, fy, "Q3", None)
        ytd6 = q_value(name, fy, "Q2", 6)
        q2_d = q_value(name, fy, "Q2", 3)
        q1 = q_value(name, fy, "Q1", 3) or q_value(name, fy, "Q1", None)
        if q_value(name, fy, "Q3", 3) is not None and q_value(name, fy, "Q3", 9) is None:
            return q_value(name, fy, "Q3", 3), False
        prior = ytd6
        if prior is None and q1 is not None and q2_d is not None:
            prior = q1 + q2_d
        elif prior is None:
            prior = q1
            if prior is not None and q2_d is not None:
                prior = prior + q2_d
        value, derived = _sub(ytd9, prior)
        return value, derived
    # Q4
    fy_v = a_value(name, fy) or q_value(name, fy, "FY", 12) or q_value(name, fy, "Q4", 12)
    ytd9 = q_value(name, fy, "Q3", 9)
    if ytd9 is None:
        parts = []
        for q in ("Q1", "Q2", "Q3"):
            piece, _ = _discrete_value(name, fy, q, True, q_value, a_value, quarterly, qcols)
            if piece is None:
                parts = None
                break
            parts.append(piece)
        ytd9 = sum(parts) if parts else None
    q4_filed = q_value(name, fy, "Q4", 3)
    if q4_filed is not None and fy_v is not None and ytd9 is not None:
        implied = fy_v - ytd9
        if abs(implied - q4_filed) <= max(1.0, 0.002 * abs(fy_v)):
            return q4_filed, True  # Yahoo-implied Q4
        return q4_filed, False
    if q4_filed is not None:
        return q4_filed, False
    if fy_v is None or ytd9 is None:
        return None, False
    return fy_v - ytd9, True


def ttm(values: list[float | None], n: int = 4) -> float | None:
    """Sum of the first ``n`` standalone quarters (latest first)."""
    window = values[:n]
    if len(window) < n or any(v is None for v in window):
        return None
    return sum(window)


def yoy_pair(current: float | None, prior: float | None) -> float | None:
    if current is None or prior in (None, 0):
        return None
    return (current - prior) / abs(prior) * 100


def matching_prior_index(columns: list[PeriodCol], index: int) -> int | None:
    """Index of the same fiscal quarter/year one year earlier, else None.

    Period ends must be about one year apart: matching on fy/fp alone once
    compared FY2022 to a FY2021 fact that ended in 2019.
    """
    col = columns[index]
    if col.fy is None or not col.fp:
        return None
    want_fy, want_fp = col.fy - 1, col.fp
    best = None
    for i, other in enumerate(columns):
        if i == index or other.fp != want_fp:
            continue
        if other.fy == want_fy and _year_apart(col.end, other.end):
            return i
        if other.fy is None and _year_apart(col.end, other.end):
            best = i if best is None else best
    return best


def _year_apart(end: str, prior_end: str, slop: int = 50) -> bool:
    try:
        delta = (date.fromisoformat(end) - date.fromisoformat(prior_end)).days
    except ValueError:
        return False
    return 365 - slop <= delta <= 366 + slop


def streak_positive(values: list[float | None]) -> int:
    """Consecutive latest-first positives; 0 if the latest is missing or ≤ 0."""
    n = 0
    for value in values:
        if value is None or value <= 0:
            break
        n += 1
    return n


def parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def from_yahoo_dataframe(df, kind: str, freq: str, currency: str) -> StatementTable:
    """Yahoo statement DataFrame (rows × period-end columns) as a StatementTable."""
    import pandas as pd

    instant = kind == "balance_sheet"
    span = 0 if instant else (12 if freq == "annual" else 3)
    fp = "FY" if freq == "annual" and not instant else None
    columns: list[PeriodCol] = []
    for col in df.columns:
        end = pd.Timestamp(col).strftime("%Y-%m-%d")
        columns.append(PeriodCol(end=end, fp=fp, span_months=span, source_label=end))
    rows: dict[str, list[float | None]] = {}
    unscaled: set[str] = set()
    for idx, series in df.iterrows():
        name = str(idx)
        vals: list[float | None] = []
        for value in series:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                vals.append(None)
            else:
                try:
                    vals.append(float(value))
                except (TypeError, ValueError):
                    vals.append(None)
        rows[name] = vals
        lower = name.lower()
        if "eps" in lower or "share" in lower:
            unscaled.add(name)
    return StatementTable(
        kind=kind, freq=freq, source="yahoo", currency=currency,
        scale=1.0, columns=columns, rows=rows, unscaled_rows=unscaled,
    )
