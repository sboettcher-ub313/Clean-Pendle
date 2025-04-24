# demo/components/trends_utils.py

import time
import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import wikipediaapi
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta

# ⏳ Google Trends throttle control
SLEEP_DELAY = 10  # seconds between keyword fetches
pytrends = TrendReq(hl="en-US", tz=360)


def fetch_interest_over_time(keyword, timeframe="now 7-d", geo=""):
    """
    Retrieves Google Trends time-series data for a keyword.
    Returns a DataFrame or empty DataFrame on failure.
    """
    print(f"⏳ Sleeping {SLEEP_DELAY}s before fetching '{keyword}'...")
    time.sleep(SLEEP_DELAY)
    try:
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop="")
        df = pytrends.interest_over_time()
        if not df.empty and keyword in df:
            return df
    except Exception as e:
        print(f"⚠️ Google Trends fetch error: {e}")
    return pd.DataFrame()


def extract_conservative_config(df, keyword):
    """
    Converts a trend time-series to a Pendle-style config dictionary.
    """
    config = {}
    s = df[keyword]

    if s.max() > 70:
        config["shock"] = True
        config["burstiness"] = round((s.max() - s.mean()) / 100, 3)

    if s.std() > 15:
        config["noise"] = round(s.std() / 100, 3)

    if s.mean() > 25:
        config["semantic_drift"] = 0.1

    return config


def get_realtime_stress_config(keyword):
    """
    Full pipeline: sleep, fetch trend data, compute conservative config.
    Returns a dict (may be empty if no data).
    """
    df = fetch_interest_over_time(keyword)
    if not df.empty:
        return extract_conservative_config(df, keyword)
    return {}


def get_wikipedia_drift_signal(word, days=7, min_edits=10):
    """
    Estimates stress from Wikipedia based on edit activity in recent days.
    If a topic has many recent edits, it may reflect instability or re-interpretation.
    This is a placeholder signal using basic page existence and title normalization.
    """
    # 🧭 Required: polite, descriptive user agent
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="PendleStressOracle/1.0 (sboettcher@ub313.net)"
    )

    page = wiki.page(word.title())

    if not page.exists():
        return {}

    # 🪄 Simulated logic: any existing page implies minor reinterpretation
    return {"semantic_drift": 0.1}


def get_twitter_volatility_signal(word, days=3, max_tweets=50):
    """
    Uses snscrape to estimate recent Twitter burstiness for a keyword.
    Higher burstiness implies reflexivity, meme cascades, or public panic.
    """
    try:
        since = (datetime.now() - timedelta(days=days)).date().isoformat()
        query = f'"{word}" since:{since}'
        tweets = list(sntwitter.TwitterSearchScraper(query).get_items())

        counts = []
        for i, tweet in enumerate(tweets):
            if i >= max_tweets:
                break
            counts.append(tweet.date)

        if len(counts) < 5:
            return {}

        # Compute frequency spread (tweet timestamps over time)
        times = sorted([ts.timestamp() for ts in counts])
        diffs = np.diff(times)
        if len(diffs) == 0:
            return {}

        burstiness = round((np.std(diffs) / np.mean(diffs)), 3)
        config = {}
        if burstiness > 1.0:
            config["burstiness"] = burstiness
            config["reflexivity"] = 0.1

        return config
    except Exception as e:
        print(f"⚠️ Twitter volatility fetch error: {e}")
        return {}