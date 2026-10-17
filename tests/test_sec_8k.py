"""Hermetic 8-K subsequent-event selection and money-fact extraction. No network."""

from __future__ import annotations

import pytest

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.sec_8k import (
    extract_exhibit_99_1,
    extract_money_facts,
    filed_pacific_date,
    load_subsequent_events,
    select_8k_filings,
)
from tradingagents.dataflows.vendors import sec_edgar

OFFERING_HTML = """
<html><body>
<p>Item 8.01 Other Events</p>
<p>On July 22, 2026, Rivian Automotive, Inc. completed an underwritten public
offering of 105,000,000 shares of common stock at a public offering price of
$13.00 per share.</p>
<p>Net proceeds to the Company from the offering were approximately $1,317 million,
after deducting underwriting discounts and commissions and estimated offering
expenses payable by the Company.</p>
</body></html>
"""

EARNINGS_VW_HTML = """
<html><body>
<p>Item 2.02 Results of Operations and Financial Condition</p>
<p>On August 5, 2026, the Company issued a press release announcing financial
results. Volkswagen has committed a contingent additional investment of up to
$1 billion. The Company expects net proceeds of $1 billion if that commitment
is funded.</p>
</body></html>
"""

DEBT_HTML = """
<html><body>
<p>The Company issued $750 million aggregate principal amount of senior notes
under an indenture dated July 1, 2026.</p>
</body></html>
"""

COMPLETE_WITH_EXHIBIT = """
<DOCUMENT>
<TYPE>8-K
<TEXT>
<p>Cover page only. See exhibit.</p>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<FILENAME>ex99-1.htm
<TEXT>
<html><body>Net proceeds to the Company from the underwritten public offering of
common stock were approximately $1,317 million.</body></html>
</TEXT>
</DOCUMENT>
"""

CIK = "0001874178"


def _payload(rows: list[dict]) -> dict:
    def col(key: str, default: str = "") -> list:
        return [r.get(key, default) for r in rows]

    return {"filings": {"recent": {
        "accessionNumber": col("accession"),
        "filingDate": col("filingDate"),
        "acceptanceDateTime": col("acceptanceDateTime"),
        "form": col("form"),
        "items": col("items"),
        "primaryDocument": col("primaryDocument"),
        "primaryDocDescription": col("primaryDocDescription"),
    }}}


@pytest.mark.unit
def test_pacific_acceptance_converts_utc_morning_to_prior_calendar_day():
    # 2026-10-18 06:00Z is 2026-10-17 23:00 PDT.
    assert filed_pacific_date("2026-10-18", "2026-10-18T06:00:00.000Z") == "2026-10-17"
    assert filed_pacific_date("2026-10-18", "2026-10-18T08:00:00.000Z") == "2026-10-18"
    assert filed_pacific_date("2026-07-22", None) == "2026-07-22"


@pytest.mark.unit
def test_extract_offering_net_proceeds_and_share_price():
    facts = extract_money_facts(OFFERING_HTML, ["3.02", "8.01"])
    assert facts.amount == pytest.approx(1_317_000_000)
    assert facts.cash_inflow is True
    assert facts.kind == "equity_raise"
    assert facts.shares == pytest.approx(105_000_000)
    assert facts.price_per_share == pytest.approx(13.0)


@pytest.mark.unit
def test_extract_does_not_take_contingent_vw_billion():
    facts = extract_money_facts(EARNINGS_VW_HTML, ["2.02"])
    assert facts.amount is None
    assert facts.cash_inflow is False


RIVN_PRICING_8K = """
in connection with the offering, issuance and sale by the Company of 75,000,000 shares of the
Company's Class A common stock, par value $0.001 per share (the "Class A common stock"), at an offering
price of $15.50 per share. In addition, under the terms of the Underwriting Agreement, the Company granted
the Underwriters the option, for 30 days, to purchase up to 11,250,000 additional shares of Class A common
stock at the public offering price, which the Underwriters exercised in full on July 8, 2026. The
net proceeds from the offering will be approximately $1.32 billion, after deducting underwriting discounts
and commissions and estimated offering expenses payable by the Company.
"""

RIVN_TERM_LOAN_8K = """
Item 2.03 Creation of a Direct Financial Obligation.
On October 7, 2026 the Company announced the funding of the committed $1.0 billion, 10-year term loan
facility from Volkswagen Group. The facility is secured by the 50% equity interest in the Joint Venture.
The principal under each Loan will be repaid each year, payable in installments of $50.0 million.
On the Funding Date, the term loans under the Loan Agreements were funded in the full committed amount of
$1.0 billion.
"""


@pytest.mark.unit
def test_extract_ignores_par_value_and_reads_offering_price():
    facts = extract_money_facts(RIVN_PRICING_8K, ["8.01"])
    assert facts.amount == pytest.approx(1.32e9)
    assert facts.cash_inflow is True
    assert facts.shares == pytest.approx(75_000_000)
    assert facts.price_per_share == pytest.approx(15.50)


@pytest.mark.unit
def test_extract_debt_uses_facility_not_amortization_installment():
    facts = extract_money_facts(RIVN_TERM_LOAN_8K, ["2.03"])
    assert facts.amount == pytest.approx(1e9)
    assert facts.kind == "debt"
    assert facts.cash_inflow is False


@pytest.mark.unit
def test_extract_debt_principal_is_not_a_cash_inflow():
    facts = extract_money_facts(DEBT_HTML, ["2.03"])
    assert facts.amount == pytest.approx(750_000_000)
    assert facts.kind == "debt"
    assert facts.cash_inflow is False


@pytest.mark.unit
def test_exhibit_99_1_is_read_from_complete_submission():
    exhibit = extract_exhibit_99_1(COMPLETE_WITH_EXHIBIT)
    assert exhibit is not None
    facts = extract_money_facts(exhibit, ["8.01"])
    assert facts.amount == pytest.approx(1_317_000_000)
    assert facts.cash_inflow is True


@pytest.mark.unit
def test_select_8k_window_and_pacific_lookahead():
    payload = _payload([
        {  # 10-Q: ignore
            "accession": "0001-26-000010", "filingDate": "2026-08-01",
            "acceptanceDateTime": "2026-08-01T12:00:00.000Z", "form": "10-Q",
            "items": "", "primaryDocument": "q.htm", "primaryDocDescription": "10-Q",
        },
        {  # on period end: exclude
            "accession": "0001-26-000011", "filingDate": "2026-06-30",
            "acceptanceDateTime": "2026-06-30T20:00:00.000Z", "form": "8-K",
            "items": "8.01", "primaryDocument": "end.htm", "primaryDocDescription": "too early",
        },
        {  # after as_of by filingDate, no acceptance: exclude
            "accession": "0001-26-000012", "filingDate": "2026-10-18",
            "acceptanceDateTime": "", "form": "8-K",
            "items": "8.01", "primaryDocument": "late.htm", "primaryDocDescription": "late",
        },
        {  # 06:00Z 18 Oct → 17 Oct PDT: include
            "accession": "0001-26-000013", "filingDate": "2026-10-18",
            "acceptanceDateTime": "2026-10-18T06:00:00.000Z", "form": "8-K",
            "items": "8.01", "primaryDocument": "edge.htm", "primaryDocDescription": "edge",
        },
        {  # 08:00Z 18 Oct → 18 Oct PDT: exclude
            "accession": "0001-26-000014", "filingDate": "2026-10-18",
            "acceptanceDateTime": "2026-10-18T08:00:00.000Z", "form": "8-K",
            "items": "8.01", "primaryDocument": "after.htm", "primaryDocDescription": "after",
        },
        {  # in window
            "accession": "0001874178-26-000045", "filingDate": "2026-07-22",
            "acceptanceDateTime": "2026-07-22T21:00:00.000Z", "form": "8-K",
            "items": "3.02,8.01", "primaryDocument": "offer.htm",
            "primaryDocDescription": "Follow-on offering",
        },
        {  # 2.02 note
            "accession": "0001874178-26-000050", "filingDate": "2026-08-05",
            "acceptanceDateTime": "2026-08-05T20:00:00.000Z", "form": "8-K",
            "items": "2.02,9.01", "primaryDocument": "earn.htm",
            "primaryDocDescription": "Earnings release",
        },
    ])
    selected = select_8k_filings(
        payload, cik=CIK, period_end="2026-06-30", as_of="2026-10-17",
    )
    accessions = {row["accession"] for row in selected}
    assert "0001874178-26-000045" in accessions
    assert "0001874178-26-000050" in accessions
    assert "0001-26-000013" in accessions
    assert "0001-26-000010" not in accessions
    assert "0001-26-000011" not in accessions
    assert "0001-26-000012" not in accessions
    assert "0001-26-000014" not in accessions
    earn = next(r for r in selected if r["accession"] == "0001874178-26-000050")
    assert earn["items"] == ["2.02"]


@pytest.mark.unit
def test_load_skips_document_fetch_for_202_only(monkeypatch):
    payload = _payload([{
        "accession": "0001874178-26-000050", "filingDate": "2026-08-05",
        "acceptanceDateTime": "2026-08-05T20:00:00.000Z", "form": "8-K",
        "items": "2.02", "primaryDocument": "earn.htm",
        "primaryDocDescription": "Earnings release",
    }])
    monkeypatch.setattr(sec_edgar, "cik_for", lambda ticker: CIK)
    monkeypatch.setattr(sec_edgar, "_cached_json", lambda url, name: payload)

    def boom(url, name):
        raise AssertionError(f"must not fetch 2.02 document {url}")

    monkeypatch.setattr(sec_edgar, "_cached_text", boom)
    events, flags = load_subsequent_events("RIVN", "2026-10-17", "2026-06-30")
    assert flags == []
    assert len(events) == 1
    assert events[0].kind == "earnings_release"
    assert events[0].amount is None
    assert events[0].cash_inflow is False


@pytest.mark.unit
def test_load_extracts_offering_from_primary_document(monkeypatch):
    payload = _payload([{
        "accession": "0001874178-26-000045", "filingDate": "2026-07-22",
        "acceptanceDateTime": "2026-07-22T21:00:00.000Z", "form": "8-K",
        "items": "3.02,8.01", "primaryDocument": "offer.htm",
        "primaryDocDescription": "Follow-on offering",
    }])
    monkeypatch.setattr(sec_edgar, "cik_for", lambda ticker: CIK)
    monkeypatch.setattr(sec_edgar, "_cached_json", lambda url, name: payload)
    monkeypatch.setattr(sec_edgar, "_cached_text", lambda url, name: OFFERING_HTML)
    events, flags = load_subsequent_events("RIVN", "2026-10-17", "2026-06-30")
    assert flags == []
    assert len(events) == 1
    assert events[0].cash_inflow is True
    assert events[0].amount == pytest.approx(1_317_000_000)
    assert events[0].kind == "equity_raise"
    assert "3.02" in events[0].items and "8.01" in events[0].items


@pytest.mark.unit
def test_load_falls_back_to_exhibit_99_1(monkeypatch):
    payload = _payload([{
        "accession": "0001874178-26-000045", "filingDate": "2026-07-22",
        "acceptanceDateTime": "2026-07-22T21:00:00.000Z", "form": "8-K",
        "items": "8.01", "primaryDocument": "cover.htm",
        "primaryDocDescription": "8-K",
    }])
    monkeypatch.setattr(sec_edgar, "cik_for", lambda ticker: CIK)
    monkeypatch.setattr(sec_edgar, "_cached_json", lambda url, name: payload)

    def fake_text(url, name):
        if url.endswith(".txt"):
            return COMPLETE_WITH_EXHIBIT
        return "<html><body>Cover page only.</body></html>"

    monkeypatch.setattr(sec_edgar, "_cached_text", fake_text)
    events, flags = load_subsequent_events("RIVN", "2026-10-17", "2026-06-30")
    assert flags == []
    assert events[0].amount == pytest.approx(1_317_000_000)
    assert events[0].cash_inflow is True


@pytest.mark.unit
def test_load_fail_soft_on_submissions_error(monkeypatch):
    monkeypatch.setattr(sec_edgar, "cik_for", lambda ticker: CIK)

    def boom(url, name):
        raise VendorUnavailableError("SEC down")

    monkeypatch.setattr(sec_edgar, "_cached_json", boom)
    events, flags = load_subsequent_events("RIVN", "2026-10-17", "2026-06-30")
    assert events == []
    assert flags and "8-K fetch failed" in flags[0]
    assert "section omitted" in flags[0]


@pytest.mark.unit
def test_non_filer_skips_silently(monkeypatch):
    monkeypatch.setattr(sec_edgar, "cik_for", lambda ticker: None)
    events, flags = load_subsequent_events("0700.HK", "2026-10-17", "2026-06-30")
    assert events == []
    assert flags == []
