"""Statement currency labels and same-currency valuation. No network."""

from __future__ import annotations

import pytest

from tradingagents.dataflows.fundamentals_packet import (
    PacketInputs,
    PricePoint,
    build_packet,
    unit_label,
)
from tradingagents.dataflows.statement_tables import PeriodCol, StatementTable

AS_OF = "2026-10-17"


def _cols() -> list[PeriodCol]:
    return [
        PeriodCol(end="2026-06-30", fy=2026, fp="Q2", span_months=3),
        PeriodCol(end="2026-03-31", fy=2026, fp="Q1", span_months=3),
        PeriodCol(end="2025-12-31", fy=2025, fp="Q4", span_months=3),
        PeriodCol(end="2025-09-30", fy=2025, fp="Q3", span_months=3),
    ]


def _table(kind: str, currency: str, rows: dict) -> StatementTable:
    cols = _cols()
    n = len(cols)
    filled = {name: (vals if isinstance(vals, list) else [vals] * n) for name, vals in rows.items()}
    return StatementTable(
        kind=kind, freq="quarterly", source="edgar", currency=currency,
        scale=1.0, columns=cols, rows=filled, unscaled_rows=set(),
    )


def _filer(
    ticker: str,
    report: str,
    price_ccy: str,
    close: float,
    *,
    cash: float,
    ni_ttm: float,
    rev_ttm: float,
    ebitda_ttm: float,
    equity: float,
    debt: float = 0.0,
    shares: float = 1_000_000_000,
    fx_rate: float | None = None,
    fx_pair: str | None = None,
    include_gp: bool = True,
) -> PacketInputs:
    q_ni, q_rev, q_ebitda = ni_ttm / 4, rev_ttm / 4, ebitda_ttm / 4
    income = {
        "Revenue": q_rev,
        "Net Income": q_ni,
        "Operating Income": q_ebitda,
    }
    if include_gp:
        income["Gross Profit"] = q_rev * 0.4
    inputs = PacketInputs(
        ticker=ticker, as_of=AS_OF, quote_type="EQUITY",
        currency_report=report, currency_price=price_ccy,
        price=PricePoint(close=close, day=AS_OF, currency=price_ccy),
        fx_usd=fx_rate, fx_pair=fx_pair, fx_day=AS_OF,
        fx_source=f"yfinance {fx_pair}" if fx_pair else None,
        yahoo_info={"sharesOutstanding": shares},
        statements={
            "income_statement:quarterly": _table("income_statement", report, income),
            "cashflow:quarterly": _table("cashflow", report, {
                "Operating Cash Flow": q_ni, "Capital Expenditure": 0.0,
            }),
            "balance_sheet:quarterly": _table("balance_sheet", report, {
                "Cash and Short-Term Investments": cash,
                "Total Debt": debt,
                "Stockholders Equity": equity,
            }),
        },
    )
    return inputs


@pytest.mark.unit
def test_unit_labels_are_iso_plus_scale():
    assert unit_label("CNY") == "CNY M"
    assert unit_label("EUR") == "EUR M"
    assert unit_label("DKK") == "DKK M"
    assert unit_label("CAD") == "CAD M"
    assert unit_label("USD") == "USD M"
    assert unit_label("JPY") == "JPY bn"
    assert unit_label("RMB") == "CNY M"


@pytest.mark.unit
def test_cny_filer_labels_cny_and_converts_valuation_to_usd():
    # Live bug: RMB 265B cash labelled USD vs USD 271B mcap → fake P/E ~3.7x, EV/EBITDA ~0.2x.
    inputs = _filer(
        "BABA", "CNY", "USD", close=100.0,
        cash=265_000_000_000, ni_ttm=80_000_000_000, rev_ttm=900_000_000_000,
        ebitda_ttm=90_000_000_000, equity=800_000_000_000, debt=0.0,
        shares=2_710_000_000, fx_rate=7.125, fx_pair="USD/CNY",
    )
    packet = build_packet(inputs)
    assert "CNY M" in packet.markdown
    assert "Statements in CNY" in packet.markdown
    assert "share price in USD" in packet.markdown
    assert "1 USD =" in packet.markdown and "CNY" in packet.markdown.split("Currencies", 1)[1][:200]
    mcap = 100.0 * 2_710_000_000  # 271B USD
    cash_usd = 265_000_000_000 / 7.125
    ni_usd = 80_000_000_000 / 7.125
    ebitda_usd = 90_000_000_000 / 7.125
    ev = mcap - cash_usd
    assert packet.metrics["mcap"] == pytest.approx(mcap)
    assert packet.metrics["cash_in_price_ccy"] == pytest.approx(cash_usd)
    assert packet.metrics["pe_ttm"] == pytest.approx(mcap / ni_usd, rel=1e-6)
    assert packet.metrics["ev_ebitda"] == pytest.approx(ev / ebitda_usd, rel=1e-6)
    assert packet.metrics["pe_ttm"] > 10  # not the mixed-currency ~3.4x
    assert packet.metrics["ev_ebitda"] > 5  # not the mixed-currency ~0.07x
    val = packet.markdown.split("## 5.", 1)[1].split("## 6.", 1)[0]
    assert "3.4x" not in val and "0.2x" not in val
    q_head = packet.markdown.split("## 2. Standalone quarterly", 1)[1].split("\n", 1)[0]
    assert "CNY M" in q_head
    assert "USD M" not in q_head


@pytest.mark.unit
def test_eur_filer_labels_eur():
    inputs = _filer(
        "ASML", "EUR", "USD", close=800.0,
        cash=10_000_000_000, ni_ttm=8_000_000_000, rev_ttm=30_000_000_000,
        ebitda_ttm=12_000_000_000, equity=20_000_000_000, debt=2_000_000_000,
        shares=400_000_000, fx_rate=0.92, fx_pair="USD/EUR",
    )
    packet = build_packet(inputs)
    assert "EUR M" in packet.markdown
    assert "Statements in EUR" in packet.markdown
    ni_usd = 8_000_000_000 / 0.92
    mcap = 800.0 * 400_000_000
    assert packet.metrics["pe_ttm"] == pytest.approx(mcap / ni_usd, rel=1e-6)
    q_head = packet.markdown.split("## 2. Standalone quarterly", 1)[1].split("\n", 1)[0]
    assert "USD M" not in q_head


@pytest.mark.unit
def test_dkk_filer_labels_dkk():
    inputs = _filer(
        "NVO", "DKK", "USD", close=50.0,
        cash=80_000_000_000, ni_ttm=100_000_000_000, rev_ttm=300_000_000_000,
        ebitda_ttm=120_000_000_000, equity=150_000_000_000,
        shares=4_500_000_000, fx_rate=6.90, fx_pair="USD/DKK",
    )
    packet = build_packet(inputs)
    assert "DKK M" in packet.markdown
    assert "Statements in DKK" in packet.markdown
    assert packet.metrics["pe_ttm"] == pytest.approx(
        (50.0 * 4_500_000_000) / (100_000_000_000 / 6.90), rel=1e-6,
    )


@pytest.mark.unit
def test_cad_filer_is_labelled_cad_not_usd():
    inputs = _filer(
        "CAR-UN.TO", "CAD", "CAD", close=20.0,
        cash=1_000_000_000, ni_ttm=400_000_000, rev_ttm=2_000_000_000,
        ebitda_ttm=600_000_000, equity=5_000_000_000,
        shares=500_000_000, fx_rate=None,
    )
    packet = build_packet(inputs)
    assert "CAD M" in packet.markdown
    assert "Statements in CAD" in packet.markdown
    assert "FX used: none (same currency)" in packet.markdown
    q_head = packet.markdown.split("## 2. Standalone quarterly", 1)[1].split("\n", 1)[0]
    assert "USD M" not in q_head
    assert packet.metrics["pe_ttm"] == pytest.approx((20.0 * 500_000_000) / 400_000_000)


@pytest.mark.unit
def test_gbp_pence_quote_uses_pounds_for_valuation():
    inputs = _filer(
        "HIJP.L", "GBP", "GBp", close=2500.0,
        cash=2_000_000_000, ni_ttm=1_000_000_000, rev_ttm=5_000_000_000,
        ebitda_ttm=1_500_000_000, equity=8_000_000_000,
        shares=1_000_000_000, fx_rate=None,
    )
    packet = build_packet(inputs)
    assert "Statements in GBP" in packet.markdown
    assert "share price in GBp" in packet.markdown
    # 2500 pence = 25 pounds; mcap = 25 * 1e9; P/S = 25e9 / 5e9 = 5x
    assert packet.metrics["mcap"] == pytest.approx(25_000_000_000)
    assert packet.metrics["ps"] == pytest.approx(5.0)
    assert packet.metrics["pe_ttm"] == pytest.approx(25.0)


@pytest.mark.unit
def test_missing_fx_omits_cross_currency_ratios():
    inputs = _filer(
        "BABA", "CNY", "USD", close=100.0,
        cash=265_000_000_000, ni_ttm=80_000_000_000, rev_ttm=900_000_000_000,
        ebitda_ttm=90_000_000_000, equity=800_000_000_000,
        shares=2_710_000_000, fx_rate=None,
    )
    packet = build_packet(inputs)
    assert packet.metrics["pe_ttm"] is None
    assert packet.metrics["ev_ebitda"] is None
    assert packet.metrics["mcap"] == pytest.approx(271_000_000_000)
    assert "CROSS-CURRENCY" in packet.markdown
    assert "omitted" in packet.markdown.split("## 5.", 1)[1]
    assert "P/E TTM" not in packet.markdown.split("## 5.", 1)[1].split("## 6.", 1)[0] or "omitted" in packet.markdown


@pytest.mark.unit
def test_gross_profit_missing_skips_ttm_margin():
    inputs = _filer(
        "AMZN", "USD", "USD", close=200.0,
        cash=80_000_000_000, ni_ttm=40_000_000_000, rev_ttm=600_000_000_000,
        ebitda_ttm=90_000_000_000, equity=250_000_000_000,
        shares=10_000_000_000, include_gp=False,
    )
    packet = build_packet(inputs)
    key = packet.markdown.split("## 1.", 1)[1].split("## 2.", 1)[0]
    gp_line = next(line for line in key.splitlines() if "Gross profit TTM" in line)
    assert "–" in gp_line or "-" in gp_line
    assert "0.0%" not in gp_line


@pytest.mark.unit
def test_usd_packet_still_says_usd_m():
    inputs = _filer(
        "RIVN", "USD", "USD", close=12.0,
        cash=5_000_000_000, ni_ttm=-3_000_000_000, rev_ttm=6_000_000_000,
        ebitda_ttm=-1_000_000_000, equity=5_000_000_000,
        shares=1_000_000_000,
    )
    packet = build_packet(inputs)
    assert "USD M" in packet.markdown
    assert "Statements in USD" in packet.markdown
    assert "JPY bn" not in packet.markdown.split("## 2.", 1)[0]
