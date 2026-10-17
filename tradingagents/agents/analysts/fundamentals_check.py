"""Optional check: numbers in the fundamentals report vs the packet.

Never blocks a run. Unmatched figures trigger one retry, then a footnote.
"""

from __future__ import annotations

import re

# Currency- or unit-tagged figures the report should have copied from the packet.
_FIGURE_RE = re.compile(
    r"(?:[$¥]\s*)?[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:\s*(?:USD|JPY)?\s*[MBTbn]n?)?"
    r"|(?:[$¥]\s*)[-+]?\d+(?:\.\d+)?(?:\s*(?:USD|JPY)?\s*[MBTbn]n?)?"
    r"|[-+]?\d+(?:\.\d+)?\s*(?:%|x)\b",
    re.IGNORECASE,
)

_SKIP = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{4}$|^FY\d{4}$", re.IGNORECASE)


def _parse_figure(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").replace("¥", "").strip()
    mult = 1.0
    if re.search(r"\bT\b|tn", cleaned, re.I):
        mult = 1e12
    elif re.search(r"\bbn\b|\bB\b", cleaned, re.I):
        mult = 1e9
    elif re.search(r"\bM\b", cleaned, re.I):
        mult = 1e6
    cleaned = re.sub(r"(USD|JPY|%|x|[MBTbn]+)\s*$", "", cleaned, flags=re.I).strip()
    try:
        return float(cleaned) * (1.0 if "%" in text or "x" in text.lower() else mult)
    except ValueError:
        return None


def figures_in(text: str) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    for match in _FIGURE_RE.finditer(text or ""):
        raw = match.group(0).strip()
        if _SKIP.match(raw):
            continue
        value = _parse_figure(raw)
        if value is None:
            continue
        found.append((raw, value))
    return found


def unmatched_figures(report: str, packet_md: str, packet_numbers: list[float] | None = None) -> list[str]:
    """Report figures that are not in the packet, within 0.5% or the last displayed digit."""
    allowed = list(packet_numbers or [])
    for _raw, value in figures_in(packet_md):
        allowed.append(value)
    unmatched: list[str] = []
    seen: set[str] = set()
    for raw, value in figures_in(report):
        if raw in seen:
            continue
        seen.add(raw)
        if _matches(value, allowed, raw):
            continue
        unmatched.append(raw)
    return unmatched


def _matches(value: float, allowed: list[float], raw: str) -> bool:
    if value == 0:
        return any(abs(a) < 1e-9 for a in allowed)
    for other in allowed:
        if other == 0:
            continue
        if abs(value - other) <= max(abs(other) * 0.005, _last_digit(raw)):
            return True
        # Unit-stripped packet numbers (already scaled) vs full-unit report.
        for scale in (1.0, 1e6, 1e9, 1e12, 1e-2, 100.0):
            if abs(value - other * scale) <= max(abs(other * scale) * 0.005, _last_digit(raw)):
                return True
    return False


def _last_digit(raw: str) -> float:
    if "." in raw:
        decimals = len(re.sub(r"[^\d].*$", "", raw.split(".", 1)[1]) or "0")
        return 10 ** (-decimals)
    return 0.5


def footnote(unmatched: list[str]) -> str:
    shown = ", ".join(unmatched[:12])
    extra = "" if len(unmatched) <= 12 else f" (+{len(unmatched) - 12} more)"
    return f"\n\n⚠ Unverified figures (not found in the fundamentals packet): {shown}{extra}"
