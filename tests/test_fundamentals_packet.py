"""Hermetic fundamentals packet: labelled figures computed in Python, no network."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.analysts.fundamentals_check import unmatched_figures
from tradingagents.dataflows.fundamentals_packet import (
    FundSnapshot,
    PacketInputs,
    PricePoint,
    _fetch_fund,
    _holdings_from_frame,
    _maybe_expense,
    _maybe_pct,
    build_packet,
    load_tool_dump,
    merge_extra_rows,
)
from tradingagents.dataflows.sec_8k import SubsequentEvent
from tradingagents.dataflows.statement_tables import PeriodCol, StatementTable
from tradingagents.dataflows.vendors import sec_edgar

FIXTURES = Path(__file__).parent / "fixtures" / "fundamentals"
AS_OF = "2026-10-16"


def _load(ticker: str) -> PacketInputs:
    text = (FIXTURES / f"{ticker}.tools.txt").read_text(encoding="utf-8")
    inputs = load_tool_dump(text, ticker, AS_OF)
    extra_path = FIXTURES / f"{ticker}_extra.json"
    if extra_path.exists():
        merge_extra_rows(inputs, json.loads(extra_path.read_text(encoding="utf-8")))
    if ticker == "RIVN":
        inputs.price = PricePoint(close=12.42, day=AS_OF, previous=13.02, previous_day="2026-10-15")
        inputs.currency_report = "USD"
    else:
        inputs.price = PricePoint(
            close=24.88, day=AS_OF, previous=24.69, previous_day="2026-10-15", currency="USD",
        )
        inputs.fx_usd = 157.99
        inputs.fx_pair = "USD/JPY"
        inputs.currency_report = "JPY"
        inputs.country = "Japan"
    return inputs


@pytest.mark.unit
def test_rivn_packet_required_figures():
    packet = build_packet(_load("RIVN"))
    m = packet.metrics
    assert m["ttm_fcf"] is not None
    assert round(m["ttm_fcf"] / 1e6) == -3489
    assert m["cash_sti"] is not None
    assert round(m["cash_sti"] / 1e6) == 5310
    assert m["inventory"] is not None
    assert round(m["inventory"] / 1e6) == 1661
    assert m["sbc_h1"] is not None
    assert round(m["sbc_h1"] / 1e6) == 433
    text = packet.markdown
    assert "2026-06-30" in text
    assert "−3,489" in text or "-3,489" in text
    assert "5,310" in text
    assert "1,661" in text
    assert "433" in text
    assert "USD M" in text
    assert "Statements in USD" in text


@pytest.mark.unit
def test_sony_packet_required_figures():
    packet = build_packet(_load("SONY"))
    m = packet.metrics
    assert m["ttm_net_income"] is not None
    assert round(m["ttm_net_income"] / 1e9, 1) == -221.6
    assert m["ttm_net_income_continuing"] is not None
    assert round(m["ttm_net_income_continuing"] / 1e9, 1) == 1114.0
    assert m["ttm_fcf"] is not None
    assert round(m["ttm_fcf"] / 1e9, 1) == 1590.3
    assert m["capital_return_ttm"] is not None
    assert round(m["capital_return_ttm"] / 1e9, 1) == 704.8
    assert "JPY bn" in packet.markdown
    assert "Statements in JPY" in packet.markdown
    assert "Q1 FY2026" in packet.markdown
    assert "Apr-Jun 2026" in packet.markdown or "Apr–Jun 2026" in packet.markdown
    # Year-end Yahoo columns are Q4, not a full-year figure parked in the quarter table.
    q_section = packet.markdown.split("## 3. Annual", 1)[0]
    assert "12,034.9" not in q_section
    assert "-12,062.9" not in q_section


@pytest.mark.unit
def test_etf_packet_has_no_forward_pe():
    inputs = PacketInputs(
        ticker="TLT", as_of=AS_OF, quote_type="ETF", name="iShares 20+ Year Treasury Bond ETF",
        price=PricePoint(close=88.0, day=AS_OF, high_52w=100.0, low_52w=80.0),
        fund=FundSnapshot(
            name="iShares 20+ Year Treasury Bond ETF", expense_ratio=0.15,
            yield_ttm=4.2, duration=16.5, ytm=4.8, quote_type="ETF",
            top_holdings=[("US Treasury", 90.0)],
        ),
    )
    packet = build_packet(inputs)
    assert packet.quote_type == "ETF"
    assert "Forward PE" not in packet.markdown
    assert "P/E" not in packet.markdown or "not meaningful" in packet.markdown.lower() or "omitted" in packet.markdown.lower()
    assert "0.15" in packet.markdown
    assert "Expense ratio" in packet.markdown
    assert "-4002" not in packet.markdown


@pytest.mark.unit
def test_edgar_tag_map_is_about_fifty_concepts():
    n = sum(len(rows) for rows in sec_edgar._STATEMENTS.values())
    assert n >= 50


@pytest.mark.unit
def test_yoy_does_not_compare_fiscal_years_that_are_years_apart(tmp_path, monkeypatch):
    """FY2022 vs a fy=2021 fact that ended in 2019 used to print −1485%."""

    def _fact(end, val, filed, form="10-K", fp="FY", start=None, fy=None):
        fact = {"end": end, "val": val, "filed": filed, "form": form,
                "fy": int(end[:4]) if fy is None else fy, "fp": fp}
        if start:
            fact["start"] = start
        return fact

    facts = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
        _fact("2022-12-31", -6_752_000_000, "2023-02-28", start="2022-01-01", fy=2022),
        _fact("2021-12-31", -5_000_000_000, "2022-02-28", start="2021-01-01", fy=2021),
        _fact("2019-12-31", -426_000_000, "2020-02-28", start="2019-01-01", fy=2021),
    ]}}}}}
    ticker_map = {"0": {"cik_str": 1874178, "ticker": "RIVN", "title": "Rivian"}}
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: ticker_map if "company_tickers" in url else facts)
    monkeypatch.setattr(sec_edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    # as_of must be within ~15 months of the latest period or EDGAR falls through.
    out = sec_edgar.get_income_statement("RIVN", "annual", "2023-03-15")
    yoy = "\n".join(line for line in out.splitlines() if line.startswith("Net Income:"))
    assert "FY2022 ended 2022-12-31 vs FY2021 ended 2021-12-31" in yoy
    assert "2019-12-31" not in yoy
    assert "1485" not in yoy


@pytest.mark.unit
def test_number_check_flags_price_that_is_not_in_the_packet():
    packet_md = "| Last close | $12.42 | |\n| FCF TTM | $-3,489M |"
    report = "The stock trades at $15, near the 50-day average, with FCF of $-3,489M."
    missing = unmatched_figures(report, packet_md)
    assert any("15" in m for m in missing)
    assert not any("3,489" in m or "3489" in m.replace(",", "") for m in missing)


@pytest.mark.unit
def test_packet_summary_is_header_and_key_metrics_only():
    packet = build_packet(_load("RIVN"))
    assert "## 0. Header" in packet.summary
    assert "## 1. Key metrics" in packet.summary
    assert "## 10." not in packet.summary


@pytest.mark.unit
def test_raw_tools_are_off_by_default_and_reenabled_by_flag():
    from tradingagents.agents.analysts import fundamentals_analyst
    from tradingagents.dataflows.config import run_config

    with run_config({"fundamentals_raw_tools": False}):
        names = [t.name for t in fundamentals_analyst.tools_for_run()]
    assert names == ["get_fundamentals_packet"]

    with run_config({"fundamentals_raw_tools": True}):
        names = [t.name for t in fundamentals_analyst.tools_for_run()]
    assert names[0] == "get_fundamentals_packet"
    assert "get_fundamentals" in names
    assert "get_balance_sheet" in names
    assert "get_insider_transactions" in names


@pytest.mark.unit
def test_analyst_forces_the_packet_when_the_model_skips_it(monkeypatch):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import Field

    from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst

    class SkipPacket(BaseChatModel):
        calls: list = Field(default_factory=list)

        @property
        def _llm_type(self) -> str:
            return "skip-packet"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            self.calls.append(1)
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="Hold. The packet was not consulted."),
            )])

    monkeypatch.setattr(
        "tradingagents.dataflows.fundamentals_packet.render_fundamentals_packet",
        lambda ticker, day: (
            f"# packet {ticker} {day}\n## 0. Header\n| As-of | {day} |\n"
            "## 1. Key metrics\n| FCF TTM | $-3,489M |\n## 2. Quarters\n"
        ),
    )
    llm = SkipPacket()
    node = create_fundamentals_analyst(llm)
    out = node({
        "trade_date": "2026-10-16",
        "company_of_interest": "RIVN",
        "messages": [HumanMessage(content="analyze RIVN")],
        "asset_type": "stock",
        "instrument_context": "RIVN equity",
    })
    assert len(llm.calls) >= 2
    assert "packet RIVN 2026-10-16" in out["fundamentals_packet_summary"]
    assert "## 2." not in out["fundamentals_packet_summary"]
    assert out["fundamentals_report"]


@pytest.mark.unit
def test_rivn_pro_forma_runway_includes_july_equity_raise():
    inputs = _load("RIVN")
    inputs.as_of = "2026-10-17"
    inputs.subsequent_events = [
        SubsequentEvent(
            filed="2026-07-22",
            items=["3.02", "8.01"],
            title="Follow-on offering",
            accession="0001874178-26-000045",
            url="https://www.sec.gov/Archives/edgar/data/1874178/000187417826000045/offer.htm",
            kind="equity_raise",
            amount=1_317_000_000,
            cash_inflow=True,
        ),
    ]
    packet = build_packet(inputs)
    assert packet.metrics["runway_ttm_quarters"] == pytest.approx(6.1, abs=0.05)
    assert packet.metrics["runway_pro_forma_quarters"] == pytest.approx(7.6, abs=0.05)
    assert packet.metrics["subsequent_cash_in"] == pytest.approx(1_317_000_000)
    section = packet.markdown.split("## 8.", 1)[1].split("## 9.", 1)[0]
    assert "2026-06-30" in section
    assert "1,317" in section
    assert "3.02" in section and "8.01" in section
    key = packet.markdown.split("## 2.", 1)[0]
    assert "pro forma" in key.lower()
    assert "7.6" in key
    assert "6.1" in key


@pytest.mark.unit
def test_debt_event_is_not_added_to_pro_forma_cash():
    inputs = _load("RIVN")
    inputs.subsequent_events = [
        SubsequentEvent(
            filed="2026-07-01",
            items=["2.03"],
            title="Senior notes",
            accession="0001874178-26-000040",
            url="https://www.sec.gov/example.htm",
            kind="debt",
            amount=750_000_000,
            cash_inflow=False,
        ),
    ]
    packet = build_packet(inputs)
    assert packet.metrics.get("runway_pro_forma_quarters") is None
    key = packet.markdown.split("## 2.", 1)[0]
    assert "pro forma" not in key.lower()
    assert "not added to cash" in packet.markdown


@pytest.mark.unit
def test_packet_omits_8k_section_when_fetch_flagged():
    inputs = _load("RIVN")
    inputs.extra_flags.append(
        "SUBSEQUENT EVENTS: 8-K fetch failed (VendorUnavailableError); section omitted"
    )
    packet = build_packet(inputs)
    section = packet.markdown.split("## 8.", 1)[1].split("## 9.", 1)[0]
    assert "Omitted" in section
    assert "SUBSEQUENT EVENTS" in packet.markdown


@pytest.mark.unit
def test_sony_annual_capex_is_positive_spend_in_core():
    packet = build_packet(_load("SONY"))
    annual = packet.markdown.split("## 3. Annual", 1)[1].split("## 4.", 1)[0]
    capex_line = next(line for line in annual.splitlines() if "Capex" in line)
    assert "Capex (spend)" in capex_line
    assert "-457.7" not in capex_line
    assert "457.7" in capex_line
    appendix = packet.markdown.split("10.5 Cash flow, annual", 1)[1].split("###", 1)[0]
    assert "source sign" in packet.markdown.split("## 10.", 1)[1].lower()
    app_line = next(
        line for line in appendix.splitlines() if "Capital Expenditure" in line
    )
    assert "-457.7" in app_line


def _nvo_like_inputs(*, statement_shares: float | None, yahoo_shares: float,
                     yahoo_mcap: float, price: float = 38.73) -> PacketInputs:
    n = 4
    cols = [
        PeriodCol(end="2026-06-30", fy=2026, fp="Q2", span_months=3),
        PeriodCol(end="2026-03-31", fy=2026, fp="Q1", span_months=3),
        PeriodCol(end="2025-12-31", fy=2025, fp="Q4", span_months=3),
        PeriodCol(end="2025-09-30", fy=2025, fp="Q3", span_months=3),
    ]

    def _tbl(kind, rows, unscaled=None):
        return StatementTable(
            kind=kind, freq="quarterly", source="yahoo", currency="DKK",
            scale=1.0, columns=cols, rows={k: (v if isinstance(v, list) else [v] * n)
                                           for k, v in rows.items()},
            unscaled_rows=unscaled or set(),
        )

    bs_rows: dict = {
        "Cash and Short-Term Investments": 80_000_000_000,
        "Total Debt": 10_000_000_000,
        "Stockholders Equity": 140_000_000_000,
    }
    unscaled = set()
    if statement_shares is not None:
        bs_rows["Shares Outstanding"] = statement_shares
        unscaled.add("Shares Outstanding")
    return PacketInputs(
        ticker="NVO", as_of="2026-10-17", quote_type="EQUITY",
        currency_report="DKK", currency_price="USD",
        price=PricePoint(close=price, day="2026-10-17", currency="USD"),
        fx_usd=6.45, fx_pair="USD/DKK", fx_day="2026-10-17",
        fx_source="yfinance USD/DKK",
        yahoo_info={
            "sharesOutstanding": yahoo_shares,
            "Market Cap": yahoo_mcap,
            "Market Cap__num": yahoo_mcap,
        },
        statements={
            "income_statement:quarterly": _tbl("income_statement", {
                "Revenue": 70_000_000_000, "Net Income": 20_000_000_000,
                "Operating Income": 25_000_000_000, "Gross Profit": 50_000_000_000,
            }),
            "cashflow:quarterly": _tbl("cashflow", {
                "Operating Cash Flow": 22_000_000_000, "Capital Expenditure": 0.0,
            }),
            "balance_sheet:quarterly": _tbl("balance_sheet", bs_rows, unscaled),
        },
    )


@pytest.mark.unit
def test_nvo_like_two_class_share_count_uses_the_total():
    """B shares 3,338M omit A shares; total ~4,425M matches Yahoo market cap."""
    a_shares, b_shares = 1_087_000_000, 3_338_000_000
    total = a_shares + b_shares
    price = 38.73
    yahoo_mcap = 171_400_000_000
    inputs = _nvo_like_inputs(
        statement_shares=total, yahoo_shares=b_shares, yahoo_mcap=yahoo_mcap, price=price,
    )
    packet = build_packet(inputs)
    assert packet.metrics["shares"] == pytest.approx(total)
    assert packet.metrics["mcap"] == pytest.approx(price * total)
    assert packet.metrics["mcap"] == pytest.approx(yahoo_mcap, rel=0.01)
    val = packet.markdown.split("## 5.", 1)[1].split("## 6.", 1)[0]
    assert "4,425.0M" in val
    assert "3,338.0M" not in val
    flags = packet.markdown.split("## 9.", 1)[1]
    assert "SHARE COUNT" in flags
    assert "3,338" in flags


@pytest.mark.unit
def test_share_count_falls_back_to_yahoo_mcap_over_price():
    b_shares = 3_338_000_000
    price = 38.73
    yahoo_mcap = 171_400_000_000
    inputs = _nvo_like_inputs(
        statement_shares=None, yahoo_shares=b_shares, yahoo_mcap=yahoo_mcap, price=price,
    )
    packet = build_packet(inputs)
    implied = yahoo_mcap / price
    assert packet.metrics["shares"] == pytest.approx(implied)
    assert packet.metrics["mcap"] == pytest.approx(yahoo_mcap)
    assert "SHARE COUNT" in packet.markdown.split("## 9.", 1)[1]


@pytest.mark.unit
def test_share_count_agrees_with_yahoo_mcap_without_a_flag():
    shares, price = 1_000_000_000, 100.0
    inputs = _nvo_like_inputs(
        statement_shares=shares, yahoo_shares=shares,
        yahoo_mcap=shares * price, price=price,
    )
    packet = build_packet(inputs)
    assert packet.metrics["shares"] == pytest.approx(shares)
    assert "SHARE COUNT:" not in packet.markdown


@pytest.mark.unit
def test_mcap_mismatch_over_15_percent_is_flagged_when_unresolved():
    """Safety net: packet mcap still >15% from Yahoo after share resolution."""
    from tradingagents.dataflows.fundamentals_packet import _share_count_flag

    inputs = _nvo_like_inputs(
        statement_shares=None, yahoo_shares=3_338_000_000,
        yahoo_mcap=171_400_000_000, price=38.73,
    )
    # Pretend resolution could not correct (used B shares only).
    flag = _share_count_flag(inputs, 3_338_000_000, "yahoo sharesOutstanding")
    assert flag is not None
    assert "SHARE COUNT" in flag
    assert "15%" in flag or "24%" in flag or "%" in flag


@pytest.mark.unit
def test_etf_holding_percent_fraction_is_scaled():
    """yfinance holdingPercent 0.05 means 5%, not 0.05%."""
    import pandas as pd

    frame = pd.DataFrame(
        {"holdingPercent": [0.0523, 0.041, 5.10]},
        index=["Tencent Holdings Ltd", "Alibaba Group", "AlreadyPercent"],
    )
    rows = _holdings_from_frame(frame)
    assert rows[0][0] == "Tencent Holdings Ltd"
    assert rows[0][1] == pytest.approx(5.23)
    assert rows[1][1] == pytest.approx(4.10)
    assert rows[2][1] == pytest.approx(5.10)
    assert _maybe_pct(0.0059) == pytest.approx(0.59)
    assert _maybe_pct(0.0231) == pytest.approx(2.31)
    assert _maybe_pct(0.048) == pytest.approx(4.8)
    assert _maybe_expense(0.0059) == pytest.approx(0.59)
    assert _maybe_expense(0.59) == pytest.approx(0.59)
    packet = build_packet(PacketInputs(
        ticker="TCHI", as_of="2026-10-17", quote_type="ETF",
        price=PricePoint(close=20.0, day="2026-10-17"),
        fund=FundSnapshot(
            name="iShares MSCI China ETF", quote_type="ETF",
            expense_ratio=_maybe_pct(0.0059), yield_ttm=_maybe_pct(0.0231),
            top_holdings=rows,
        ),
    ))
    holdings = packet.markdown.split("## 2. Top holdings", 1)[1].split("## 9.", 1)[0]
    assert "5.23%" in holdings
    assert "4.10%" in holdings
    assert "0.05%" not in holdings
    assert "0.59%" in packet.markdown
    assert "2.31%" in packet.markdown


@pytest.mark.unit
def test_equity_etf_bond_holdings_dataframe_does_not_break_fetch(monkeypatch):
    """yfinance may give an empty DataFrame for bond_holdings on equity ETFs."""
    import pandas as pd

    holdings = pd.DataFrame({"holdingPercent": [0.0496]}, index=["Tencent"])
    funds = MagicMock()
    funds.fund_overview = {"AnnualReportExpenseRatio": 0.0059}
    funds.bond_holdings = pd.DataFrame()
    funds.top_holdings = holdings
    ticker = MagicMock()
    ticker.funds_data = funds
    monkeypatch.setattr(
        "tradingagents.dataflows.vendors.yahoo.common.yf_retry",
        lambda fn: fn(),
    )
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", lambda _t: ticker)
    snap = _fetch_fund("TCHI", {
        "quoteType": "ETF", "netExpenseRatio": 0.59, "yield": 0.0231,
    })
    assert snap.top_holdings[0][1] == pytest.approx(4.96)
    assert snap.expense_ratio == pytest.approx(0.59)
    assert snap.yield_ttm == pytest.approx(2.31)
    assert snap.duration is None
