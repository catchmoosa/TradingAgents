"""SEC EDGAR fundamentals: statements as they were filed, not as they read today.

Every other fundamentals vendor serves the current value of a past period and
cuts on the fiscal period end, so a run sees figures the company had not yet
filed, and later restatements replace what was actually published. EDGAR carries
the filing date of every fact, so a run can be limited to what was on file by its
own date.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest import mock

import pytest

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors import sec_edgar

_REAL_FETCH = sec_edgar._fetch_json

TICKER_MAP = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}


def _fact(end, val, filed, form="10-K", fp="FY", start=None, fy=None):
    fact = {"end": end, "val": val, "filed": filed, "form": form,
            "fy": int(end[:4]) if fy is None else fy, "fp": fp}
    if start:
        fact["start"] = start
    return fact


FACTS = {
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {"us-gaap": {
        "Assets": {"units": {"USD": [
            _fact("2008-09-27", 39_572_000_000, "2008-11-05"),
            _fact("2008-09-27", 36_171_000_000, "2010-01-25", form="10-K/A"),
            _fact("2010-09-25", 75_183_000_000, "2010-10-27"),
            _fact("2022-03-26", 350_662_000_000, "2022-04-29", form="10-Q", fp="Q2"),
            _fact("2023-09-30", 352_583_000_000, "2023-11-03"),
            _fact("2024-03-30", 337_411_000_000, "2024-05-03", form="10-Q", fp="Q2"),
            _fact("2024-06-29", 331_612_000_000, "2024-08-02", form="10-Q", fp="Q3"),
            _fact("2024-09-28", 364_980_000_000, "2024-11-01"),
        ]}},
        "Liabilities": {"units": {"USD": [_fact("2024-09-28", 308_030_000_000, "2024-11-01")]}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": [
            _fact("2024-09-28", 6.08, "2024-11-01", start="2023-09-30"),
        ]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            # One filing reports the quarter and the year to date under one end date.
            _fact("2025-12-31", 81_300_000_000, "2026-01-29", form="10-Q", fp="Q2", start="2025-10-01"),
            _fact("2025-12-31", 158_900_000_000, "2026-01-29", form="10-Q", fp="Q2", start="2025-07-01"),
            _fact("2024-09-28", 391_035_000_000, "2024-11-01", start="2023-09-30"),
        ]}},
    }},
}


@pytest.fixture(autouse=True)
def _no_network_or_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sec_edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(sec_edgar, "_fetch_json", lambda url: TICKER_MAP if "company_tickers" in url else FACTS)


@pytest.mark.unit
def test_a_us_filer_resolves_to_its_cik():
    assert sec_edgar.cik_for("AAPL") == "0000320193"
    assert sec_edgar.cik_for("aapl") == "0000320193"


@pytest.mark.unit
def test_a_non_filer_is_reported_as_such_not_as_missing_data():
    with pytest.raises(NoMarketDataError, match="not a US SEC filer"):
        sec_edgar.get_balance_sheet("0700.HK", "annual", "2026-01-01")


@pytest.mark.unit
def test_a_restated_figure_reads_as_it_did_at_the_time():
    """The value published then, not the correction filed later."""
    as_filed = sec_edgar.get_balance_sheet("AAPL", "annual", "2009-06-30")
    restated = sec_edgar.get_balance_sheet("AAPL", "annual", "2011-01-01")
    assert "39572" in as_filed and "36171" not in as_filed
    assert "36171" in restated


@pytest.mark.unit
def test_a_period_that_ended_but_was_not_filed_yet_is_not_served():
    """The fiscal year ended 2024-09-28; it reached the public on 2024-11-01."""
    before = sec_edgar.get_balance_sheet("AAPL", "annual", "2024-10-15")
    after = sec_edgar.get_balance_sheet("AAPL", "annual", "2024-11-15")
    assert "2024-09-28" not in before
    assert "2024-09-28" in after and "364980" in after


@pytest.mark.unit
def test_the_quarter_is_not_confused_with_the_year_to_date():
    """One filing carries both spans under the same end date (#MSFT-shaped)."""
    out = sec_edgar.get_income_statement("AAPL", "quarterly", "2026-06-01")
    assert "81300" in out
    assert "158900" not in out


@pytest.mark.unit
def test_a_line_the_filer_does_not_tag_is_named_unavailable():
    out = sec_edgar.get_balance_sheet("AAPL", "annual", "2024-11-15")
    assert "Stockholders Equity" in out and "not tagged" in out
    assert "364980" in out  # the rest of the statement still returns


@pytest.mark.unit
def test_the_report_states_the_vintage_rule():
    out = sec_edgar.get_balance_sheet("AAPL", "annual", "2024-11-15")
    assert "filed on or before 2024-11-15" in out


@pytest.mark.unit
def test_a_filer_with_no_usable_facts_reads_differently_from_a_non_filer(monkeypatch):
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else {"facts": {}})
    with pytest.raises(NoMarketDataError, match="no us-gaap facts"):
        sec_edgar.get_balance_sheet("AAPL", "annual", "2026-01-01")


@pytest.mark.unit
def test_company_facts_are_fetched_once_per_company_not_once_per_date(tmp_path, monkeypatch):
    """A sweep asks for many dates; the filing history is the same file."""
    calls = []
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: calls.append(url) or (TICKER_MAP if "company_tickers" in url else FACTS))
    for as_of in ("2024-11-15", "2025-01-15", "2025-06-15"):
        sec_edgar.get_balance_sheet("AAPL", "annual", as_of)
    assert len([u for u in calls if "companyfacts" in u]) == 1


@pytest.mark.unit
def test_it_works_unconfigured_and_takes_the_caller_s_own_contact(monkeypatch):
    """SEC returns 403 for a User-Agent with no contact address, so the default
    carries one; a caller who sets their own replaces it."""
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    assert "@" in sec_edgar._user_agent()

    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "MyDesk research@example.com")
    assert sec_edgar._user_agent() == "MyDesk research@example.com"


@pytest.mark.unit
def test_a_throttle_lets_the_next_vendor_try(monkeypatch):
    """SEC throttles by refusing the request; the router then tries yfinance."""
    import requests

    from tradingagents.dataflows.errors import VendorUnavailableError

    def _throttled(*a, **k):
        raise requests.HTTPError(response=mock.Mock(status_code=429))

    monkeypatch.setattr(sec_edgar.requests, "get", _throttled)
    with pytest.raises(VendorUnavailableError):
        _REAL_FETCH("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json")


@pytest.mark.unit
def test_values_do_not_break_the_columns():
    """Figures run to the billions; a thousands separator would split the field."""
    out = sec_edgar.get_balance_sheet("AAPL", "annual", "2024-11-15")
    body = [row for row in out.splitlines() if row.startswith("Total Assets")][0]
    assert body.count(",") == out.splitlines()[3].count(",")


@pytest.mark.unit
def test_a_per_share_figure_keeps_its_own_unit():
    """Statements are reported in millions, but EPS is dollars per share: scaling
    it the same way prints a real figure as zero."""
    out = sec_edgar.get_income_statement("AAPL", "annual", "2024-11-15")
    row = [r for r in out.splitlines() if r.startswith("Diluted EPS")][0]
    assert "6.08" in row
    assert "USD/shares" in row or "per share" in row


@pytest.mark.unit
def test_every_row_has_one_cell_per_period():
    """An untagged line still has to line up with the columns, or the table is
    misread by position."""
    out = sec_edgar.get_balance_sheet("AAPL", "annual", "2024-11-15")
    table = [r for r in out.splitlines() if r and not r.startswith("#")]
    widths = {row.count(",") for row in table}
    assert len(widths) == 1, table


@pytest.mark.unit
def test_a_server_error_lets_the_next_vendor_try(monkeypatch):
    import requests

    from tradingagents.dataflows.errors import VendorError

    def _server_error(*a, **k):
        raise requests.HTTPError(response=mock.Mock(status_code=503))

    monkeypatch.setattr(sec_edgar.requests, "get", _server_error)
    with pytest.raises(VendorError):  # not a bare HTTPError
        _REAL_FETCH("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json")


@pytest.mark.unit
def test_older_periods_fall_back_to_the_tag_the_filer_used_then():
    """Filers renamed lines when the revenue standard changed, so one tag covers
    only recent years. Each period takes one tag, never a sum of two."""
    facts = {"Revenues": {"units": {"USD": [_fact("2015-09-26", 233_715_000_000, "2015-10-28",
                                                 start="2014-09-28")]}},
             "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                 _fact("2024-09-28", 391_035_000_000, "2024-11-01", start="2023-09-30")]}}}
    values, unit, _meta = sec_edgar._as_of(facts, ("RevenueFromContractWithCustomerExcludingAssessedTax",
                                                   "Revenues"), "2026-01-01", ((300, 400),))
    assert values == {("2015-09-26", 0): 233_715_000_000, ("2024-09-28", 0): 391_035_000_000}
    assert unit == "USD"


@pytest.mark.unit
def test_a_period_reported_under_two_tags_takes_the_preferred_one_not_both():
    facts = {"Revenues": {"units": {"USD": [_fact("2024-09-28", 111, "2024-11-01", start="2023-09-30")]}},
             "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                 _fact("2024-09-28", 999, "2024-11-01", start="2023-09-30")]}}}
    values, _, _ = sec_edgar._as_of(facts, ("RevenueFromContractWithCustomerExcludingAssessedTax",
                                           "Revenues"), "2026-01-01", ((300, 400),))
    assert values == {("2024-09-28", 0): 999}


@pytest.mark.unit
def test_the_default_identification_names_the_package_version(monkeypatch):
    """SEC asks automated clients to identify themselves; a release names its own version."""
    import tradingagents

    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    assert sec_edgar._user_agent() == f"TradingAgents/{tradingagents.__version__} (contact@example.com)"


@pytest.mark.unit
def test_capital_expenditure_is_found_under_either_tag_filers_use(monkeypatch):
    """NVIDIA and Amazon report purchases of productive assets, not of property and equipment."""
    facts = {"facts": {"us-gaap": {"PaymentsToAcquireProductiveAssets": {"units": {"USD": [
        _fact("2024-12-31", 70_000_000, "2025-02-10", start="2024-01-01")]}}}}}
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else facts)
    out = sec_edgar.get_cashflow("AAPL", "annual", "2025-03-01")
    assert [r for r in out.splitlines() if r.startswith("Capital Expenditure")] == ["Capital Expenditure,70"]


def _columns(out):
    return [line for line in out.splitlines() if line.startswith(",")][0].split(",")[1:]


@pytest.mark.unit
def test_an_annual_balance_sheet_has_no_quarter_end_columns():
    """A balance has no span, so a 10-Q's quarter-end balance passed as annual."""
    annual = " ".join(_columns(sec_edgar.get_balance_sheet("AAPL", "annual", "2024-11-15")))
    quarterly = " ".join(_columns(sec_edgar.get_balance_sheet("AAPL", "quarterly", "2024-11-15")))
    assert "2022-03-26" not in annual and "2024-09-28" in annual
    assert "2022-03-26" in quarterly


@pytest.mark.unit
def test_a_twelve_month_total_from_a_quarterly_report_is_not_a_fiscal_year(monkeypatch):
    """Amazon's 10-Qs report trailing twelve months, which passed the annual span
    check and read as fiscal years overlapping the real ones."""
    facts = {"facts": {"us-gaap": {"NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
        _fact("2024-12-31", 115_000_000_000, "2025-02-07", start="2024-01-01"),
        _fact("2025-03-31", 113_000_000_000, "2025-05-02", form="10-Q", fp="Q1", start="2024-04-01"),
    ]}}}}}
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else facts)
    assert _columns(sec_edgar.get_cashflow("AAPL", "annual", "2025-06-01")) == ["FY2024 ended 2024-12-31"]


@pytest.mark.unit
def test_a_recast_outside_the_annual_report_still_counts_from_its_filing(monkeypatch):
    """Filers recast past years in an 8-K after a split or spin-off. The annual
    report decides the columns; the value is the latest filing of any form."""
    facts = {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": [
        _fact("2017-03-31", 16.97, "2017-06-15", form="20-F", start="2016-04-01"),
        _fact("2018-03-31", 2.50, "2018-06-15", form="20-F", start="2017-04-01"),
        _fact("2019-03-31", 2.80, "2019-06-20", form="20-F", start="2018-04-01"),
        _fact("2017-03-31", 2.12, "2019-09-30", form="6-K", start="2016-04-01"),
    ]}}}}}
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else facts)

    def eps(as_of):
        out = sec_edgar.get_income_statement("AAPL", "annual", as_of)
        header, row = _columns(out), [line for line in out.splitlines()
                                      if line.startswith("Diluted EPS")][0].split(",")[1:]
        return next(val for label, val in zip(header, row, strict=True) if "2017-03-31" in label)

    assert eps("2019-01-01") == "16.97"
    assert eps("2020-01-01") == "2.12"


@pytest.mark.unit
def test_a_cash_flow_filed_year_to_date_is_served_with_its_span(monkeypatch):
    """Many filers tag a 10-Q's cash flows only year to date (3, 6, then 9 months):
    a quarterly table of discrete quarters alone would stop at each first quarter."""
    ytd = [
        _fact("2025-12-27", 30_000_000_000, "2026-01-30", form="10-Q", fp="Q1", start="2025-09-28"),
        _fact("2026-03-28", 53_000_000_000, "2026-05-01", form="10-Q", fp="Q2", start="2025-09-28"),
        _fact("2026-06-27", 81_000_000_000, "2026-08-01", form="10-Q", fp="Q3", start="2025-09-28"),
    ]
    facts = {"facts": {"us-gaap": {"NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": ytd}}}}}
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else facts)

    out = sec_edgar.get_cashflow("AAPL", "quarterly", "2026-09-28")

    assert _columns(out) == [
        "Q3 FY2026 ended 2026-06-27 (9 months)",
        "Q2 FY2026 ended 2026-03-28 (6 months)",
        "Q1 FY2025 ended 2025-12-27",
    ]
    assert [r for r in out.splitlines() if r.startswith("Operating Cash Flow")] == [
        "Operating Cash Flow,81000,53000,30000",
    ]


@pytest.mark.unit
def test_a_row_filed_only_year_to_date_keeps_its_figure_beside_a_row_filed_by_quarter(monkeypatch):
    facts = {"facts": {"us-gaap": {
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
            _fact("2026-06-30", 120_000_000, "2026-08-01", form="10-Q", fp="Q2", start="2026-01-01")]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
            _fact("2026-06-30", 10_000_000, "2026-08-01", form="10-Q", fp="Q2", start="2026-04-01")]}},
    }}}
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else facts)

    out = sec_edgar.get_cashflow("AAPL", "quarterly", "2026-09-28")

    assert _columns(out) == [
        "Q2 FY2026 ended 2026-06-30",
        "Q2 FY2026 ended 2026-06-30 (6 months)",
    ]
    rows = {r.split(",")[0]: r.split(",")[1:] for r in out.splitlines() if r and not r.startswith("#")}
    assert rows["Operating Cash Flow"] == ["not filed", "120"]
    assert rows["Capital Expenditure"] == ["10", "not filed"]


def _rows(out):
    return {line.split(",")[0]: line.split(",")[1:]
            for line in out.splitlines() if line and not line.startswith("#") and not line.startswith(",")}


RIVN_MAP = {"0": {"cik_str": 1874178, "ticker": "RIVN", "title": "Rivian Automotive, Inc."}}
SONY_MAP = {"0": {"cik_str": 313838, "ticker": "SONY", "title": "Sony Group Corporation"}}

# Twelve calendar quarters, newest first. The last four exist only to prove truncation.
_RIVN_PERIODS = [
    ("2026-06-30", 2026, "Q2", "10-Q", "2026-08-07"),
    ("2026-03-31", 2026, "Q1", "10-Q", "2026-05-06"),
    ("2025-12-31", 2025, "FY", "10-K", "2026-02-24"),
    ("2025-09-30", 2025, "Q3", "10-Q", "2025-11-05"),
    ("2025-06-30", 2025, "Q2", "10-Q", "2025-08-06"),
    ("2025-03-31", 2025, "Q1", "10-Q", "2025-05-06"),
    ("2024-12-31", 2024, "FY", "10-K", "2025-02-24"),
    ("2024-09-30", 2024, "Q3", "10-Q", "2024-11-05"),
    ("2024-06-30", 2024, "Q2", "10-Q", "2024-08-06"),
    ("2024-03-31", 2024, "Q1", "10-Q", "2024-05-07"),
    ("2023-12-31", 2023, "FY", "10-K", "2024-02-26"),
    ("2023-09-30", 2023, "Q3", "10-Q", "2023-11-07"),
]


def _rivn_line(tag, amounts, duration=False):
    facts = []
    for (end, fy, fp, form, filed), amount in zip(_RIVN_PERIODS, amounts, strict=True):
        start = (date.fromisoformat(end) - timedelta(days=91)).isoformat() if duration else None
        facts.append(_fact(end, amount, filed, form=form, fp=fp, fy=fy, start=start))
    return {tag: {"units": {"USD": facts}}}


def _rivn_facts():
    n = len(_RIVN_PERIODS)
    cash = [3_592_000_000, 2_845_000_000, 5_294_000_000, 6_000_000_000,
            7_000_000_000, 8_000_000_000, 8_200_000_000, 5_000_000_000,
            4_000_000_000, 3_000_000_000, 2_000_000_000, 1_000_000_000]
    sti = [1_718_000_000, 1_500_000_000, 1_200_000_000, 1_100_000_000,
           1_000_000_000, 900_000_000, 800_000_000, 700_000_000,
           600_000_000, 500_000_000, 400_000_000, 300_000_000]
    assets = [a + 10_000_000_000 for a in cash]
    revenue = [1_300_000_000, 1_100_000_000, 1_700_000_000, 1_200_000_000,
               1_000_000_000, 900_000_000, 1_400_000_000, 800_000_000,
               700_000_000, 600_000_000, 1_200_000_000, 500_000_000]
    net_income = [-200_000_000, -300_000_000, -400_000_000, -350_000_000,
                  -400_000_000, -450_000_000, -500_000_000, -550_000_000,
                  -600_000_000, -650_000_000, -700_000_000, -750_000_000]
    us_gaap = {}
    us_gaap.update(_rivn_line("Assets", assets))
    us_gaap.update(_rivn_line("CashAndCashEquivalentsAtCarryingValue", cash))
    us_gaap.update(_rivn_line("MarketableSecuritiesCurrent", sti))
    us_gaap.update(_rivn_line("RevenueFromContractWithCustomerExcludingAssessedTax", revenue, duration=True))
    us_gaap.update(_rivn_line("NetIncomeLoss", net_income, duration=True))
    assert n == 12
    return {"facts": {"us-gaap": us_gaap}}


def _patch_filer(monkeypatch, ticker_map, facts):
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: ticker_map if "company_tickers" in url else facts)


@pytest.mark.unit
def test_a_wide_statement_keeps_recent_periods_latest_first_with_fiscal_labels(monkeypatch):
    _patch_filer(monkeypatch, RIVN_MAP, _rivn_facts())
    out = sec_edgar.get_balance_sheet("RIVN", "quarterly", "2026-10-16")
    cols = _columns(out)
    assert len(cols) == 8
    assert cols[0].startswith("Q2 FY2026 ended 2026-06-30")
    assert cols[1].startswith("Q1 FY2026 ended 2026-03-31")
    assert "2024-06-30" not in out  # older than the last 8 quarters
    assert "2023-09-30" not in out
    cash = _rows(out)["Cash and Equivalents"]
    assert cash[0] == "3592" and cash[1] == "2845"


@pytest.mark.unit
def test_short_term_investments_are_on_the_balance_sheet_and_in_combined_cash(monkeypatch):
    _patch_filer(monkeypatch, RIVN_MAP, _rivn_facts())
    out = sec_edgar.get_balance_sheet("RIVN", "quarterly", "2026-10-16")
    rows = _rows(out)
    assert rows["Short-Term Investments"][0] == "1718"
    assert rows["Cash and Short-Term Investments"][0] == "5310"  # 3592 + 1718
    assert "not filed" in out


@pytest.mark.unit
def test_yoy_pairs_matching_fiscal_quarters_instead_of_adjacent_columns(monkeypatch):
    _patch_filer(monkeypatch, RIVN_MAP, _rivn_facts())
    out = sec_edgar.get_income_statement("RIVN", "quarterly", "2026-10-16")
    assert "Revenue: Q2 FY2026 ended 2026-06-30 vs Q2 FY2025 ended 2025-06-30: 1300 vs 1000 (+30.0%)" in out
    assert "Net Income: Q2 FY2026 ended 2026-06-30 vs Q2 FY2025 ended 2025-06-30: -200 vs -400 (+50.0%)" in out


@pytest.mark.unit
def test_stale_us_gaap_is_not_served_so_the_next_vendor_can_try(monkeypatch):
    facts = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        _fact("2021-03-31", 27_000_000_000, "2021-06-18", form="20-F"),
    ]}}}}}
    _patch_filer(monkeypatch, SONY_MAP, facts)
    with pytest.raises(NoMarketDataError, match="no recent annual"):
        sec_edgar.get_balance_sheet("SONY", "annual", "2026-10-16")


@pytest.mark.unit
def test_a_quarterly_request_does_not_silently_return_annual_20_f_columns(monkeypatch):
    facts = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        _fact("2026-03-31", 100_000_000, "2026-06-18", form="20-F"),
    ]}}}}}
    _patch_filer(monkeypatch, SONY_MAP, facts)
    with pytest.raises(NoMarketDataError, match="no quarterly"):
        sec_edgar.get_balance_sheet("SONY", "quarterly", "2026-10-16")
    # Annual of a recent 20-F is still served.
    assert "2026-03-31" in sec_edgar.get_balance_sheet("SONY", "annual", "2026-10-16")


@pytest.mark.unit
def test_a_jpy_statement_is_labelled_jpy_not_usd(monkeypatch):
    facts = {"facts": {"us-gaap": {"Assets": {"units": {"JPY": [
        _fact("2026-03-31", 35_000_000_000_000, "2026-06-18", form="20-F"),
    ]}}}}}
    _patch_filer(monkeypatch, SONY_MAP, facts)
    out = sec_edgar.get_balance_sheet("SONY", "annual", "2026-10-16")
    assert "JPY in millions" in out
    assert "USD in millions" not in out
    assert "Total Assets (JPY),35000000" in out


@pytest.mark.unit
def test_an_ifrs_filer_without_us_gaap_falls_through(monkeypatch):
    facts = {"facts": {"ifrs-full": {"Assets": {"units": {"JPY": [
        _fact("2026-03-31", 1, "2026-06-18", form="20-F"),
    ]}}}}}
    _patch_filer(monkeypatch, SONY_MAP, facts)
    with pytest.raises(NoMarketDataError, match="IFRS"):
        sec_edgar.get_balance_sheet("SONY", "annual", "2026-10-16")


@pytest.mark.unit
def test_stale_edgar_facts_let_the_configured_next_vendor_serve(monkeypatch):
    """Default chain is sec_edgar,yfinance; SONY's stale us-gaap must not stick."""
    from tradingagents.dataflows import router
    from tradingagents.dataflows.config import set_config

    facts = {"facts": {
        "us-gaap": {"Assets": {"units": {"JPY": [
            _fact("2021-03-31", 27_500_000_000_000, "2021-06-18", form="20-F"),
        ]}}},
        "ifrs-full": {"Assets": {"units": {"JPY": [
            _fact("2026-03-31", 35_000_000_000_000, "2026-06-18", form="20-F"),
        ]}}},
    }}
    _patch_filer(monkeypatch, SONY_MAP, facts)
    set_config({"data_vendors": {"fundamental_data": "sec_edgar,yfinance"}})
    with mock.patch.dict(router.VENDOR_METHODS["get_balance_sheet"], {
        "sec_edgar": sec_edgar.get_balance_sheet,
        "yfinance": lambda *a, **k: "# Balance Sheet data for SONY (quarterly)\nJPY figures from yfinance\n",
    }):
        out = router.route_to_vendor("get_balance_sheet", "SONY", "quarterly", "2026-10-16")
    assert "yfinance" in out
    assert "USD in millions" not in out
