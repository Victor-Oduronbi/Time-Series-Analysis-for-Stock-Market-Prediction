"""
NGX Hybrid Pipeline — Text Engine
==================================
Polite web scraper for Nigerian corporate announcements / financial news.

Design commitments (from project spec):
  - Check robots.txt before parsing anything.
  - Random 3-6s delay between requests (time.sleep).
  - Declared, honest User-Agent header.
  - Graceful handling of missing/blocked paths (fail closed, not silently skip).

Target source (v1): BusinessDay's "Companies" section (businessday.ng/category/companies/).

  NOTE — source history: Nairametrics was the original v1 target, but its
  robots.txt explicitly disallows the company-news category path for bots,
  and its server also 403s bare/generic requests at the network layer. Both
  signals were respected rather than worked around (no UA spoofing, no
  headless-browser evasion) — that's the whole point of the polite-scraper
  commitment. BusinessDay's robots.txt only disallows login/register/search/
  AMP paths, so /category/companies/ (and /category/markets/, /category/banking/)
  are open to a well-behaved bot.

  Swap BASE_URL / SECTION_PATH to point at yet another source later if
  needed — the polite-fetch and robots-check logic is source-agnostic;
  only `parse_listing_page()` and `parse_article_page()` are source-specific
  and will need new selectors per site.

NOTE ON SELECTORS: the CSS selectors in parse_listing_page/parse_article_page
below are a reasonable WordPress-style starting guess (article cards with
h2/h3 headline links, time or .date-published tags, and .entry-content
article bodies) — but they have NOT been verified against BusinessDay's
actual markup, since this environment only had access to a text/markdown
rendering of the page, not raw HTML. Before running this for real:
    1. Run inspect_page_structure() against one real listing page URL.
    2. Compare the printed candidate selectors against what's hardcoded
       below and adjust SELECTORS if they don't match.
This script cannot be validated against the live internet from within this
environment (no network egress here), so treat the selectors as a documented
starting point, not a guarantee.
"""

import random
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://businessday.ng"
SECTION_PATH = "/category/companies/"   # also try /category/markets/ or /category/banking/

USER_AGENT = (
    "OVA-NGX-Research-Bot/1.0 "
    "(Educational/Portfolio project; contact: victoroduronbi@gmail.com)"
)

MIN_DELAY_SECONDS = 3
MAX_DELAY_SECONDS = 6
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3

# CSS selectors — see NOTE ON SELECTORS above. Centralized here so a site
# swap only requires editing this dict, not the functions below.
SELECTORS = {
    "listing_headline_link": "h2.post-title a",  # confirmed via inspect_page_structure
    "article_date": "time, .date-published, .entry-date, .post-date",
    "article_body": ".entry-content, article .content, .post-content",
}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ngx_text_engine")


@dataclass
class NewsItem:
    date: str
    ticker: str
    headline: str
    body_text: str
    source_url: str


# ---------------------------------------------------------------------------
# Compliance layer
# ---------------------------------------------------------------------------

def is_allowed_by_robots(target_url: str, user_agent: str = USER_AGENT) -> bool:
    """
    Check robots.txt for target_url before we ever request it.
    Fails CLOSED: if robots.txt can't be fetched/parsed, we do NOT assume
    permission — we block and let the caller decide how to proceed.
    """
    parsed = urlparse(target_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception as exc:
        logger.warning(
            "Could not fetch/parse %s (%s). Failing closed — treating as disallowed.",
            robots_url, exc,
        )
        return False

    allowed = rp.can_fetch(user_agent, target_url)
    logger.info("robots.txt check for %s -> allowed=%s", target_url, allowed)
    return allowed


def polite_get(url: str, session: requests.Session) -> requests.Response | None:
    """
    Fetch a URL with retry/backoff, honest headers, and a randomized delay
    BEFORE the request (so the delay is paid whether or not this call
    succeeds — that's the point of being polite to the host).
    """
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, MAX_RETRIES + 1):
        delay = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
        logger.info("Sleeping %.1fs before requesting %s (attempt %d/%d)",
                     delay, url, attempt, MAX_RETRIES)
        time.sleep(delay)

        try:
            resp = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            logger.warning("Non-200 status %s for %s", resp.status_code, url)
            if resp.status_code in (429, 503):
                # Back off harder on explicit rate-limit / unavailable signals
                time.sleep(delay * 2)
        except requests.RequestException as exc:
            logger.warning("Request error on attempt %d for %s: %s", attempt, url, exc)

    logger.error("Giving up on %s after %d attempts", url, MAX_RETRIES)
    return None


# ---------------------------------------------------------------------------
# Parsing layer (source-specific)
# ---------------------------------------------------------------------------

def parse_listing_page(html: str, page_url: str) -> list[str]:
    """Extract article URLs from a listing/category page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for link_tag in soup.select(SELECTORS["listing_headline_link"]):
        if link_tag.get("href"):
            links.append(urljoin(page_url, link_tag["href"]))
    logger.info("Parsed %d article links from listing page %s", len(links), page_url)
    return links


def parse_article_page(html: str, article_url: str, ticker: str) -> NewsItem | None:
    """Extract headline, date, and body text from a single article page."""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1")
    headline = title_tag.get_text(strip=True) if title_tag else None

    # Date: confirmed via inspection that BusinessDay embeds this in a meta tag,
    # not visible markup. Fall back to visible tags only if the meta tag is absent.
    raw_date = None
    meta_date_tag = soup.find("meta", attrs={"property": "article:published_time"})
    if meta_date_tag and meta_date_tag.get("content"):
        raw_date = meta_date_tag["content"]
    else:
        date_tag = soup.select_one(SELECTORS["article_date"])
        if date_tag:
            raw_date = date_tag.get("datetime") or date_tag.get_text(strip=True)

    # Body: strip script/style/ad elements first — confirmed the raw .post-content
    # div mixes ad-tracking markup in with the real article text.
    body_text = ""
    body_container = soup.select_one(SELECTORS["article_body"])
    if body_container:
        cleaned = BeautifulSoup(str(body_container), "html.parser")
        for junk in cleaned.find_all(["script", "style", "iframe", "ins"]):
            junk.decompose()
        body_text = cleaned.get_text(" ", strip=True)

    if not headline:
        logger.warning("No headline found for %s — skipping (selector likely stale)", article_url)
        return None

    return NewsItem(
        date=normalize_date(raw_date),
        ticker=ticker,
        headline=headline,
        body_text=body_text,
        source_url=article_url,
    )


def normalize_date(raw_date: str | None) -> str:
    """Best-effort normalization to YYYY-MM-DD; falls back to raw string."""
    if not raw_date:
        return ""
    # Try ISO-8601 with offset first (matches the article:published_time meta tag
    # format confirmed on BusinessDay, e.g. "2026-07-22T09:48:15+00:00").
    try:
        return datetime.fromisoformat(raw_date).strftime("%Y-%m-%d")
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(raw_date[:len(fmt) + 5], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw_date  # leave as-is; downstream merge step should re-check this


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def scrape_ticker_news(ticker: str, max_articles: int = 20, max_pages: int = 1) -> pd.DataFrame:
    """
    End-to-end: check robots.txt -> fetch listing page(s) -> fetch each article ->
    return a tidy DataFrame ready for Day 2 (sentiment scoring).

    max_pages controls how many listing pages to walk (page 1, then
    /page/2/, /page/3/, ... — standard WordPress pagination pattern).
    Each page is independently robots.txt-checked and politely delayed,
    same as every article fetch.
    """
    session = requests.Session()
    items: list[NewsItem] = []
    article_urls: list[str] = []

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            listing_url = urljoin(BASE_URL, SECTION_PATH)
        else:
            # Standard WordPress pagination: /category/companies/page/2/
            listing_url = urljoin(BASE_URL, f"{SECTION_PATH.rstrip('/')}/page/{page_num}/")

        if not is_allowed_by_robots(listing_url):
            logger.warning("robots.txt disallows %s — stopping pagination here.", listing_url)
            break

        listing_resp = polite_get(listing_url, session)
        if listing_resp is None:
            logger.warning("Could not retrieve %s — stopping pagination here.", listing_url)
            break

        page_links = parse_listing_page(listing_resp.text, listing_url)
        if not page_links:
            logger.info("No links found on %s — likely past the last page. Stopping.", listing_url)
            break

        article_urls.extend(page_links)
        if len(article_urls) >= max_articles:
            break

    article_urls = article_urls[:max_articles]

    for url in article_urls:
        if not is_allowed_by_robots(url):
            logger.warning("Skipping %s — disallowed by robots.txt", url)
            continue
        resp = polite_get(url, session)
        if resp is None:
            continue
        item = parse_article_page(resp.text, url, ticker)
        if item:
            items.append(item)

    df = pd.DataFrame([asdict(i) for i in items])
    logger.info("Collected %d news items for %s across %d listing page(s)", len(df), ticker, page_num)
    return df


def scrape_company_news(max_articles: int = 50, max_pages: int = 3) -> pd.DataFrame:
    """
    Scrape the general Companies feed WITHOUT assuming a single ticker.
    This replaces the old one-ticker-per-scrape design: the category page is
    a mixed feed (any listed or unlisted company can appear), so tagging by
    ticker has to happen AFTER scraping, via tag_articles_by_ticker().
    """
    return scrape_ticker_news(ticker="UNTAGGED", max_articles=max_articles, max_pages=max_pages)


# Company name -> ticker alias map. Extend this to match your actual target
# universe (the project spec's example tickers were ZENITHBANK.LG,
# DANGCEM.LG, MTNN.LG — add those and any others you're tracking).
# Keys are the canonical ticker; values are name variants as they actually
# appear in BusinessDay prose (confirmed against the sample headline batch).
NGX_TICKER_ALIASES: dict[str, list[str]] = {
    "FIRSTHOLDCO": ["FirstHoldCo", "First HoldCo"],
    "BUAFOODS": ["BUA Foods"],
    "ACCESSCORP": ["Access Holdings", "Access Bank"],
    "TRANSPOWER": ["Transcorp Power"],
    "WEMABANK": ["Wema Bank"],
    "CSCS": ["CSCS"],
    "ZENITHBANK": ["Zenith Bank", "ZenithBank"],
    "DANGCEM": ["Dangote Cement", "DangCem"],
    "MTNN": ["MTN Nigeria", "MTNN"],
}


def tag_articles_by_ticker(
    df: pd.DataFrame,
    aliases: dict[str, list[str]] = NGX_TICKER_ALIASES,
) -> pd.DataFrame:
    """
    Tag each scraped article with the ticker(s) it actually mentions, by
    matching company-name aliases against headline + body text.

    Distinguishes WHERE the match came from:
      - mention_scope="headline": the company is named in the headline —
        strong signal the article is actually ABOUT this company.
      - mention_scope="body_only": the company is only named in the body —
        often just incidental context (e.g. "First HoldCo overtook Zenith
        Bank" mentions Zenith but isn't a Zenith Bank story). Treating this
        the same as a headline mention would attribute an article's full
        sentiment to a company it barely concerns.

    Returns a NEW DataFrame (one row per article-ticker match). Articles
    matching none of the tracked companies are dropped. Downstream (Day 2),
    you can choose to keep only mention_scope=="headline" rows for a
    cleaner, more conservative sentiment signal, or keep both scopes for
    broader coverage at the cost of some noise — that's a judgment call
    worth making deliberately rather than defaulting silently.
    """
    tagged_rows = []
    for _, row in df.iterrows():
        headline_lower = str(row["headline"]).lower()
        body_lower = str(row["body_text"]).lower()

        for ticker, names in aliases.items():
            in_headline = any(name.lower() in headline_lower for name in names)
            in_body = any(name.lower() in body_lower for name in names)
            if not (in_headline or in_body):
                continue
            new_row = row.to_dict()
            new_row["ticker"] = ticker
            new_row["mention_scope"] = "headline" if in_headline else "body_only"
            tagged_rows.append(new_row)

    result = pd.DataFrame(tagged_rows)
    if len(result):
        logger.info(
            "Tagged %d article-ticker rows out of %d scraped articles "
            "(%d headline-scope, %d body-only-scope)",
            len(result), len(df),
            (result["mention_scope"] == "headline").sum(),
            (result["mention_scope"] == "body_only").sum(),
        )
    return result


# Weighting scheme for mention_scope, used when aggregating sentiment scores
# per ticker per day (Day 2). A headline mention means the article is very
# likely genuinely ABOUT that company; a body-only mention is real but
# weaker context (e.g. "First HoldCo overtook Zenith Bank" is a FirstHoldCo
# story that merely NAMES Zenith Bank). Full weight vs. half weight reflects
# that difference without discarding body-only rows outright — with a small
# corpus, that volume still matters.
MENTION_SCOPE_WEIGHTS: dict[str, float] = {
    "headline": 1.0,
    "body_only": 0.5,
}


def add_mention_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a numeric `weight` column derived from mention_scope."""
    df = df.copy()
    df["weight"] = df["mention_scope"].map(MENTION_SCOPE_WEIGHTS).fillna(1.0)
    return df


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment_score",
) -> pd.DataFrame:
    """
    Collapse article-level (possibly multiple-per-day) sentiment scores into
    one weighted daily sentiment value per ticker per day.

    Expects df to have: date, ticker, weight, and `sentiment_col` (added in
    Day 2 once VADER + the lexicon score each article). Formula:
        daily_sentiment = sum(sentiment_i * weight_i) / sum(weight_i)
    grouped by (date, ticker). This is a placeholder for Day 2 — call it
    AFTER sentiment scoring is added, not before. Included here now so the
    weighting decision is locked in alongside the tagging logic it depends on.
    """
    if sentiment_col not in df.columns:
        raise KeyError(
            f"'{sentiment_col}' not found — run VADER/lexicon scoring "
            f"before aggregating daily sentiment."
        )

    weighted = df.assign(_weighted=df[sentiment_col] * df["weight"])
    grouped = weighted.groupby(["date", "ticker"]).agg(
        weighted_sum=("_weighted", "sum"),
        weight_sum=("weight", "sum"),
        n_articles=(sentiment_col, "count"),
    )
    grouped["daily_sentiment"] = grouped["weighted_sum"] / grouped["weight_sum"]
    return grouped[["daily_sentiment", "n_articles"]].reset_index()


def inspect_page_structure(url: str) -> None:
    """
    Diagnostic helper: fetch a single page and print candidate selectors
    (tag + class combos) so you can quickly re-derive SELECTORS if the
    live site's markup has drifted from what's hardcoded above.
    """
    if not is_allowed_by_robots(url):
        raise PermissionError(f"robots.txt disallows fetching {url}")

    session = requests.Session()
    resp = polite_get(url, session)
    if resp is None:
        logger.error("Could not fetch %s for inspection", url)
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    print("--- Headline candidates (h1/h2/h3), with hrefs where present ---")
    for tag in soup.find_all(["h1", "h2", "h3"])[:10]:
        link = tag.find("a")
        href = urljoin(url, link["href"]) if link and link.get("href") else None
        print(tag.name, tag.get("class"), tag.get_text(strip=True)[:80], "->", href)

    print("\n--- Date candidates: meta tags ---")
    for prop in ("article:published_time", "article:modified_time", "og:updated_time"):
        tag = soup.find("meta", attrs={"property": prop})
        if tag:
            print(prop, "->", tag.get("content"))
    for name in ("date", "publish-date", "sailthru.date"):
        tag = soup.find("meta", attrs={"name": name})
        if tag:
            print(f"meta[name={name}]", "->", tag.get("content"))

    print("\n--- Date candidates: visible tags (time / class contains 'date') ---")
    for tag in soup.find_all("time")[:5]:
        print(tag.attrs, tag.get_text(strip=True))
    for tag in soup.find_all(class_=lambda c: c and "date" in " ".join(c).lower())[:5]:
        print(tag.name, tag.get("class"), tag.get_text(strip=True)[:80])

    print("\n--- Body text candidates: ALL matches per class, AFTER stripping <script>/<style> ---")
    for cls in ("entry-content", "post-content", "article-content", "content", "article-body"):
        matches = soup.find_all(class_=cls)
        for i, found in enumerate(matches):
            cleaned = BeautifulSoup(str(found), "html.parser")
            for junk in cleaned.find_all(["script", "style", "iframe", "ins"]):
                junk.decompose()
            text = cleaned.get_text(" ", strip=True)
            print(f".{cls}[{i}] (len={len(text)}) ->", text[:400])


if __name__ == "__main__":
    # Example run — replace TICKER with the NGX ticker you're targeting.
    # This will make real network requests when run in an environment with
    # egress enabled; it will NOT run against the live internet from here.
    TICKER = "ZENITHBANK"
    news_df = scrape_ticker_news(TICKER, max_articles=50, max_pages=3)
    print(news_df.head())
    news_df.to_csv(f"{TICKER}_news_raw.csv", index=False)
