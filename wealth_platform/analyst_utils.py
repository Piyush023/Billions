"""Structured analyst verdict parsing, consensus voting, and rule-based signals."""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

BULLISH = frozenset({"BULLISH", "UNDERVALUED", "POSITIVE", "BUY", "OVERWEIGHT"})
BEARISH = frozenset({"BEARISH", "OVERVALUED", "NEGATIVE", "SELL", "UNDERWEIGHT"})
NEUTRAL = frozenset({"NEUTRAL", "MIXED", "FAIRLY VALUED", "HOLD"})


def parse_verdict(report: str) -> Dict:
    """Extract verdict label and confidence from an analyst report tail line."""
    text = (report or "").upper()
    confidence = 50
    m = re.search(r"CONFIDENCE[:\s]+(\d{1,3})", text)
    if m:
        confidence = min(100, max(0, int(m.group(1))))
    verdict = "NEUTRAL"
    for label in BULLISH | BEARISH | NEUTRAL:
        if label in text:
            verdict = label
            break
    return {"verdict": verdict, "confidence": confidence}


def consensus_vote(
    technical: Dict, fundamentals: Dict, news: Dict, sentiment: Optional[Dict] = None,
) -> Dict:
    """Deterministic panel vote from parsed analyst verdicts."""
    votes = [technical, fundamentals, news]
    if sentiment:
        votes.append(sentiment)

    def bucket(v: str) -> str:
        if v in BULLISH:
            return "bull"
        if v in BEARISH:
            return "bear"
        return "neutral"

    buckets = [bucket(v["verdict"]) for v in votes]
    bull = buckets.count("bull")
    bear = buckets.count("bear")
    avg_conf = sum(v["confidence"] for v in votes) / len(votes)

    if bull >= 3 and bear == 0:
        agreement = "strong_bull"
        rating = "BUY"
    elif bull >= 2 and bear <= 1:
        agreement = "lean_bull"
        rating = "OVERWEIGHT"
    elif bear >= 3 and bull == 0:
        agreement = "strong_bear"
        rating = "SELL"
    elif bear >= 2 and bull <= 1:
        agreement = "lean_bear"
        rating = "UNDERWEIGHT"
    else:
        agreement = "split"
        rating = "HOLD"

    return {
        "agreement": agreement,
        "rating": rating,
        "confidence": round(avg_conf),
        "bull_votes": bull,
        "bear_votes": bear,
        "needs_debate": agreement == "split" or (40 <= avg_conf <= 55),
    }


def derive_sentiment_block(headlines: List[str], price_change_1w_pct: float, vol_ratio: float) -> str:
    """Rule-based sentiment proxy — no LLM call."""
    pos_words = ("surge", "gain", "beat", "upgrade", "record", "growth", "profit", "win")
    neg_words = ("fall", "drop", "miss", "downgrade", "loss", "fraud", "probe", "cut")
    pos, neg = 0, 0
    for h in headlines:
        low = h.lower()
        pos += sum(1 for w in pos_words if w in low)
        neg += sum(1 for w in neg_words if w in low)
    if price_change_1w_pct > 3 and vol_ratio > 1.2:
        pos += 1
    elif price_change_1w_pct < -3:
        neg += 1
    if pos > neg + 1:
        label, conf = "BULLISH", min(85, 55 + pos * 5)
    elif neg > pos + 1:
        label, conf = "BEARISH", min(85, 55 + neg * 5)
    else:
        label, conf = "MIXED", 45
    return (
        f"Rule-based sentiment (no LLM): {label} confidence {conf}. "
        f"Headline tone +{pos}/-{neg}; 1w price {price_change_1w_pct:+.1f}%; vol ratio {vol_ratio:.2f}."
    )


def compute_ml_signal(symbol: str) -> Optional[Dict]:
    """Lightweight momentum classifier from price data — no external ML deps."""
    sym = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    try:
        df = yf.download(sym, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"]
        r_1m = float(close.iloc[-1] / close.iloc[-22] - 1)
        r_1w = float(close.iloc[-1] / close.iloc[-6] - 1)
        sma20 = float(close.rolling(20).mean().iloc[-1])
        price = float(close.iloc[-1])
        score = r_1m * 2 + r_1w + (0.15 if price > sma20 else -0.15)
        if score > 0.08:
            return {"signal": "BUY", "confidence": min(90, int(50 + score * 200))}
        if score < -0.08:
            return {"signal": "SELL", "confidence": min(90, int(50 + abs(score) * 200))}
        return {"signal": "HOLD", "confidence": 45}
    except Exception:  # noqa: BLE001
        return None


def research_gate_allows(rating: str, confidence: int, gate: int, held_qty: int) -> Tuple[bool, str]:
    """Expanded gate: block non-actionable ratings before trader/PM spend."""
    rating = (rating or "HOLD").upper()
    if confidence < gate:
        return False, f"confidence {confidence} below gate {gate}"
    if rating in ("HOLD",):
        return False, "research rating HOLD"
    if rating in ("SELL", "UNDERWEIGHT") and held_qty <= 0:
        return False, f"{rating} with no holding (CNC — cannot short)"
    if rating in ("BUY", "OVERWEIGHT") or (rating in ("SELL", "UNDERWEIGHT") and held_qty > 0):
        return True, "actionable"
    return False, f"non-actionable rating {rating}"
