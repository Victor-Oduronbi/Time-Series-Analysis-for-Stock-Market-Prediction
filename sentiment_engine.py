"""
NGX Hybrid Pipeline — Sentiment Engine (Day 2)
================================================
VADER sentiment scoring, augmented with a custom `ngx_financial_lexicon` so
general-purpose VADER understands Nigerian financial/corporate vocabulary it
wasn't trained on (e.g. "dividend", "impairment", "delisting").

IMPORTANT — untested in this environment: vaderSentiment is not installed
here, and this sandbox has no network access to pip-install it. This module
is written against vaderSentiment's stable, long-unchanged public API
(SentimentIntensityAnalyzer().polarity_scores(text) -> dict with
'neg'/'neu'/'pos'/'compound'), but I have NOT been able to run it end-to-end
myself the way earlier pieces were verified. Install and test locally:

    pip install vaderSentiment

Then run the __main__ block at the bottom as a smoke test before trusting
this against your real tagged_df.
"""

import logging
from functools import lru_cache

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ngx_sentiment_engine")


# ---------------------------------------------------------------------------
# Custom NGX financial lexicon
# ---------------------------------------------------------------------------
# VADER's internal lexicon scores words on roughly a -4 (most negative) to
# +4 (most positive) scale, later normalized into the -1..+1 compound score.
# These additions/overrides follow that same scale. Values are a reasonable
# starting point, not empirically tuned — expect to revise these once you
# see real compound scores against real headlines (e.g. if "acquisition"
# behaves too positively on stories about hostile takeovers, adjust down).
#
# Organized by rough theme so it's easy to extend.
NGX_FINANCIAL_LEXICON: dict[str, float] = {
    # --- Corporate actions: positive ---
    "dividend": 2.2,
    "buyback": 2.0,
    "upgrade": 2.5,
    "profit": 2.3,
    "growth": 2.0,
    "expansion": 1.8,
    "partnership": 1.5,
    "acquisition": 1.2,   # mildly positive by default; context can flip this
    "merger": 1.0,
    "listing": 1.2,
    "ipo": 1.5,
    "surge": 1.8,
    "rally": 1.8,
    "outperform": 2.0,
    "bonus": 1.5,

    # --- Corporate actions: negative ---
    "fine": -2.8,
    "suspension": -3.0,
    "suspended": -3.0,
    "delisting": -3.5,
    "delisted": -3.5,
    "default": -3.0,
    "downgrade": -2.5,
    "recall": -2.0,
    "lawsuit": -2.2,
    "litigation": -2.0,
    "investigation": -2.2,
    "fraud": -3.5,
    "impairment": -2.2,
    "writedown": -2.5,
    "restructuring": -1.0,   # often neutral-to-cautious rather than sharply negative
    "bailout": -1.5,
    "loss": -2.3,
    "layoffs": -2.5,
    "resignation": -1.2,
    "sacked": -2.0,
    "probe": -2.0,
    "sanction": -2.5,
    "penalty": -2.5,
    "plunge": -2.5,
    "crash": -3.0,
    "shortfall": -2.0,

    # --- Overrides for single words VADER's general-English lexicon misreads ---
    # "gross" defaults negative in VADER (colloquial "disgusting"), but in
    # financial text it's near-neutral and often precedes good news
    # ("gross earnings", "gross profit"). Confirmed as a real false-negative
    # via a live test: a genuinely positive FirstHoldCo earnings headline
    # scored -0.477 purely because of this collision.
    "gross": 0.3,
    # Market-direction slang VADER's general lexicon has no opinion on.
    "bullish": 2.2,
    "bearish": -2.2,
    # Neutral financial mechanics terms VADER's general lexicon sometimes
    # leans negative on (e.g. "leverage" as in "leveraging/exploiting").
    "liquidity": 0.0,
    "leverage": 0.0,
    "liquid": 0.3,
}

# NOTE: vaderSentiment tokenizes text word-by-word and looks up each token
# individually — it does NOT support multi-word phrase matching. Lexicon
# entries must be single words. Phrases like "gross earnings" or "record
# profit" would silently never match anything if added here; that mistake
# was caught and removed during development of this module. If you want
# phrase-level detection (e.g. "shares suspended" vs. "trading suspended"
# meaning different things), that requires custom pre-processing before
# handing text to VADER, not lexicon entries.


@lru_cache(maxsize=1)
def get_ngx_analyzer() -> SentimentIntensityAnalyzer:
    """
    Build (once, cached) a VADER analyzer whose lexicon has been augmented
    with NGX_FINANCIAL_LEXICON. lru_cache means repeated calls reuse the
    same analyzer instance rather than rebuilding the lexicon merge each time.
    """
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(NGX_FINANCIAL_LEXICON)
    logger.info("Initialized VADER analyzer with %d custom NGX lexicon terms",
                len(NGX_FINANCIAL_LEXICON))
    return analyzer


def score_text(text: str) -> float:
    """
    Return VADER's compound sentiment score for a single string, already
    in the -1 (most negative) to +1 (most positive) range the project spec
    calls for. Empty/missing text scores as neutral (0.0).
    """
    if not text or not isinstance(text, str):
        return 0.0
    analyzer = get_ngx_analyzer()
    return analyzer.polarity_scores(text)["compound"]


def score_articles(df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
    """
    Add a `sentiment_score` column to a tagged articles DataFrame.

    Defaults to scoring the HEADLINE only, matching the project roadmap's
    Day 2 spec ("pass the scraped text headlines through VADER"). Headlines
    are also shorter and less noisy than full body text, which tends to
    produce a cleaner compound score. If you want body-text-informed scoring
    instead, pass text_col="body_text" — but note VADER's compound score can
    get diluted/washed toward neutral on very long text since it's a
    sentence-level tool at heart; consider sentence-splitting and averaging
    if you go that route, rather than scoring one giant string.
    """
    df = df.copy()
    df["sentiment_score"] = df[text_col].apply(score_text)
    logger.info(
        "Scored %d articles on '%s' — mean=%.3f, min=%.3f, max=%.3f",
        len(df), text_col,
        df["sentiment_score"].mean(), df["sentiment_score"].min(), df["sentiment_score"].max(),
    )
    return df


if __name__ == "__main__":
    # Smoke test — run this locally after `pip install vaderSentiment` to
    # sanity-check the lexicon behaves as expected before trusting it
    # against your real tagged_df.
    test_headlines = [
        "FirstHoldCo delivers N1.93trn gross earnings, N653.5bn PBT in H1 2026",
        "CAC gives companies August 1 deadline to comply with disclosure rules",
        "Company fined for regulatory breach, shares suspended",
        "Bank announces record profit and special dividend",
        "Firm faces fraud investigation after shareholder lawsuit",
        "Analysts turn bullish on banking stocks after strong H1 results",
        "Investors grow bearish as naira volatility persists",
        "Bank boosts liquidity position ahead of Q3",
    ]
    for headline in test_headlines:
        print(f"{score_text(headline):+.3f}  {headline}")
