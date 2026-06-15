"""Thesis-aware exit levels — wire agent stop/target into math exits."""

from typing import Optional, Tuple


def exit_reason(
    symbol: str,
    last_price: float,
    average_price: float,
    desk,
    stop_pct: float,
    target_pct: float,
    sentinel_stop: Optional[float] = None,
) -> Optional[Tuple[str, str]]:
    """Return (reason, source) if position should exit, else None."""
    if average_price <= 0 or last_price <= 0:
        return None

    change = (last_price / average_price - 1) * 100
    thesis = desk.thesis_for(symbol) if desk else None

    # 1. Agent thesis absolute levels (highest priority)
    if thesis:
        sl = float(thesis.get("stop_loss") or 0)
        tp = float(thesis.get("take_profit") or 0)
        if sl > 0 and last_price <= sl:
            return f"thesis stop-loss hit @ Rs.{sl:.2f} ({change:+.1f}%)", "thesis"
        if tp > 0 and last_price >= tp:
            return f"thesis target hit @ Rs.{tp:.2f} ({change:+.1f}%)", "thesis"

    # 2. Sentinel-tightened stop
    if sentinel_stop and last_price <= sentinel_stop:
        return f"sentinel stop {sentinel_stop:.2f} hit ({change:+.1f}%)", "sentinel"

    # 3. Config percentage stops (fallback)
    if change <= -stop_pct:
        return f"stop-loss hit ({change:.1f}%)", "config"
    if change >= target_pct:
        return f"target hit ({change:.1f}%)", "config"
    return None
