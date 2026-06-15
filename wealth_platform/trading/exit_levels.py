"""Thesis-aware exit levels — trailing stops, partial profit, math exits."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExitSignal:
    reason: str
    source: str
    quantity: Optional[int] = None  # None = sell entire position


def exit_reason(
    symbol: str,
    last_price: float,
    average_price: float,
    desk,
    stop_pct: float,
    target_pct: float,
    sentinel_stop: Optional[float] = None,
    partial_take_pct: float = 5.0,
    partial_taken: bool = False,
    trail_activate_pct: float = 4.0,
    trail_pct: float = 2.5,
) -> Optional[ExitSignal]:
    """Return ExitSignal if position should exit (full or partial), else None."""
    if average_price <= 0 or last_price <= 0:
        return None

    change = (last_price / average_price - 1) * 100
    thesis = desk.thesis_for(symbol) if desk else None
    high_water = float(thesis.get("high_water_mark") or average_price) if thesis else average_price

    # Partial profit — bank half, let rest run with trailing stop
    if not partial_taken and change >= partial_take_pct:
        return ExitSignal(
            reason=f"partial take-profit at +{change:.1f}% (bank {partial_take_pct:.0f}% threshold)",
            source="partial",
        )

    # 1. Agent thesis absolute levels
    if thesis:
        sl = float(thesis.get("stop_loss") or 0)
        tp = float(thesis.get("take_profit") or 0)
        if sl > 0 and last_price <= sl:
            return ExitSignal(
                reason=f"thesis stop-loss hit @ Rs.{sl:.2f} ({change:+.1f}%)",
                source="thesis",
            )
        if tp > 0 and last_price >= tp:
            return ExitSignal(
                reason=f"thesis target hit @ Rs.{tp:.2f} ({change:+.1f}%)",
                source="thesis",
            )

    # 2. Trailing stop after position has worked
    peak_gain = (high_water / average_price - 1) * 100 if average_price else 0
    if peak_gain >= trail_activate_pct:
        trail_stop = high_water * (1 - trail_pct / 100)
        if last_price <= trail_stop:
            return ExitSignal(
                reason=f"trailing stop hit (peak +{peak_gain:.1f}%, trail {trail_pct:.1f}%)",
                source="trailing",
            )

    # 3. Sentinel-tightened stop
    if sentinel_stop and last_price <= sentinel_stop:
        return ExitSignal(
            reason=f"sentinel stop {sentinel_stop:.2f} hit ({change:+.1f}%)",
            source="sentinel",
        )

    # 4. Config percentage stops (fallback)
    if change <= -stop_pct:
        return ExitSignal(reason=f"stop-loss hit ({change:.1f}%)", source="config")
    if change >= target_pct:
        return ExitSignal(reason=f"target hit ({change:.1f}%)", source="config")
    return None
