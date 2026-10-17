"""Next earnings date and surprise history, point-in-time from yfinance."""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from tradingagents.dataflows import date_window
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors.yahoo import fundamentals

TODAY = "2026-10-16"


def _frame(*rows):
    """rows: (date, estimate, reported, surprise)."""
    index = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="America/New_York") for d, *_ in rows],
        name="Earnings Date",
    )
    return pd.DataFrame(
        {
            "EPS Estimate": [row[1] for row in rows],
            "Reported EPS": [row[2] for row in rows],
            "Surprise(%)": [row[3] for row in rows],
        },
        index=index,
    )


RIVN_DATES = _frame(
    ("2026-10-29", -0.41, None, None),
    ("2026-08-05", -0.47, -0.44, 6.4),
    ("2026-05-06", -0.52, -0.41, 21.2),
    ("2026-02-13", -0.41, -0.46, -12.2),
    ("2025-11-07", -0.44, -0.41, 6.8),
    ("2024-08-06", -0.50, -0.48, 4.0),
)


def _context(frame, as_of, today=TODAY):
    with mock.patch.object(date_window, "get_current_date", return_value=today), \
            mock.patch.object(fundamentals, "yf_retry", lambda fn: frame):
        return fundamentals.get_earnings_context("RIVN", as_of)


@pytest.mark.unit
def test_next_earnings_date_and_recent_surprises():
    out = _context(RIVN_DATES, TODAY)
    assert "Next earnings date: 2026-10-29" in out
    assert "2026-08-05: reported EPS -0.44; estimate -0.47; surprise +6.4%" in out
    assert "2026-10-29" in out.split("Recent earnings surprises")[0]
    assert "reported EPS" not in out.split("Recent earnings surprises")[0]


@pytest.mark.unit
def test_a_historical_run_does_not_see_later_surprises():
    out = _context(RIVN_DATES, "2026-06-01")
    _head, _sep, surprises = out.partition("Recent earnings surprises")
    assert "2026-08-05" not in surprises
    assert "2026-05-06" in surprises
    assert "2026-10-29" not in out


@pytest.mark.unit
def test_next_after_as_of_is_the_then_upcoming_print_not_today_s_calendar():
    """On 2026-06-01 the next print in this series is 2026-08-05, not Oct 29."""
    out = _context(RIVN_DATES, "2026-06-01")
    assert "Next earnings date: 2026-08-05" in out
    assert "Next earnings date: 2026-10-29" not in out


@pytest.mark.unit
def test_a_run_older_than_the_series_does_not_invent_a_next_date():
    """Yahoo's window is recent prints; 2020 must not be told 2026 is next."""
    out = _context(RIVN_DATES, "2020-01-15")
    assert "not available as of 2020-01-15" in out
    assert "2026-10-29" not in out
    assert "Recent earnings surprises" not in out


@pytest.mark.unit
def test_empty_calendar_is_no_data(monkeypatch):
    from tradingagents.dataflows.vendors.yahoo import common

    monkeypatch.setattr(common, "vendor_reachable", lambda url: True)
    with mock.patch.object(fundamentals, "yf_retry", lambda fn: pd.DataFrame()), \
            pytest.raises(NoMarketDataError):
        fundamentals.get_earnings_context("RIVN", TODAY)


@pytest.mark.unit
def test_the_tool_clamps_to_the_run_date():
    from tradingagents.agents import tools

    with mock.patch.object(tools, "route_to_vendor", return_value="ok") as routed:
        tools.get_earnings_context.func("NVDA", "2026-12-01", trade_date="2026-10-16")
    assert routed.call_args.args == ("get_earnings_context", "NVDA", "2026-10-16")
