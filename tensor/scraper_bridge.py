"""Bridge between HTML article scraping and UnifiedTensor L0.

Parses HTML articles, extracts ticker mentions via regex,
scores sentiment via lexicon, and injects into MarketGraph → tensor L0.
"""
import re
import sys
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from html.parser import HTMLParser

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tensor.core import UnifiedTensor
from tensor.market_graph import MarketGraph


# Known ticker patterns (1-5 uppercase letters, optionally preceded by $)
_TICKER_RE = re.compile(r'(?<!\w)\$?([A-Z]{1,5})(?!\w)')

# Common English words to exclude from ticker matches
_STOP_WORDS = frozenset([
    'A', 'I', 'AM', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'HE',
    'IF', 'IN', 'IS', 'IT', 'ME', 'MY', 'NO', 'OF', 'ON', 'OR', 'SO',
    'TO', 'UP', 'US', 'WE', 'ARE', 'BUT', 'CAN', 'DID', 'FOR', 'GET',
    'GOT', 'HAS', 'HAD', 'HER', 'HIM', 'HIS', 'HOW', 'ITS', 'LET',
    'MAY', 'NEW', 'NOT', 'NOW', 'OLD', 'OUR', 'OUT', 'OWN', 'RAN',
    'SAY', 'SHE', 'THE', 'TOO', 'USE', 'WAS', 'WAY', 'WHO', 'WHY',
    'ALL', 'AND', 'ANY', 'BIG', 'DAY', 'END', 'FAR', 'FEW', 'GOD',
    'HIT', 'LOW', 'MAN', 'ONE', 'PUT', 'RUN', 'SET', 'TRY', 'TWO',
    'YET', 'CEO', 'CFO', 'SEC', 'FED', 'GDP', 'IPO', 'ETF', 'NYSE',
    'ALSO', 'BACK', 'BEEN', 'CALL', 'CAME', 'COME', 'EACH', 'EVEN',
    'FIND', 'FIVE', 'FROM', 'GIVE', 'GOOD', 'HAVE', 'HERE', 'HIGH',
    'INTO', 'JUST', 'KEEP', 'KNOW', 'LAST', 'LIKE', 'LONG', 'LOOK',
    'MADE', 'MAKE', 'MANY', 'MORE', 'MOST', 'MUCH', 'MUST', 'NAME',
    'NEXT', 'ONLY', 'OVER', 'PART', 'SAID', 'SAME', 'SHOW', 'SIDE',
    'SOME', 'SUCH', 'SURE', 'TAKE', 'TELL', 'THAN', 'THAT', 'THEM',
    'THEN', 'THEY', 'THIS', 'TIME', 'VERY', 'WANT', 'WELL', 'WENT',
    'WERE', 'WHAT', 'WHEN', 'WILL', 'WITH', 'WORD', 'WORK', 'YEAR',
    'YOUR', 'ZERO', 'DOWN', 'MOVE', 'REAL', 'RISE', 'SELL', 'GAIN',
    'LOSS', 'BULL', 'BEAR', 'DATA', 'OPEN', 'CLOSE',
])

# Sentiment lexicon (word → score)
_POSITIVE = {
    'buy': 0.6, 'bullish': 0.8, 'surge': 0.7, 'rally': 0.7, 'gain': 0.5,
    'profit': 0.5, 'growth': 0.5, 'strong': 0.4, 'beat': 0.5, 'upgrade': 0.6,
    'outperform': 0.6, 'positive': 0.4, 'up': 0.3, 'rise': 0.4, 'rising': 0.4,
    'high': 0.3, 'record': 0.4, 'boom': 0.6, 'soar': 0.7, 'jump': 0.5,
    'increase': 0.4, 'opportunity': 0.4, 'momentum': 0.4, 'breakout': 0.5,
    'recover': 0.4, 'recovery': 0.4, 'optimistic': 0.5, 'confident': 0.4,
    'exceed': 0.4, 'exceeded': 0.4, 'revenue': 0.2, 'earnings': 0.2,
}
_NEGATIVE = {
    'sell': -0.6, 'bearish': -0.8, 'crash': -0.8, 'decline': -0.5,
    'loss': -0.5, 'drop': -0.5, 'fall': -0.4, 'falling': -0.4,
    'weak': -0.4, 'miss': -0.5, 'downgrade': -0.6, 'negative': -0.4,
    'down': -0.3, 'low': -0.3, 'risk': -0.3, 'fear': -0.5,
    'plunge': -0.7, 'tumble': -0.6, 'slump': -0.6, 'cut': -0.3,
    'decrease': -0.4, 'concern': -0.3, 'warning': -0.4, 'recession': -0.6,
    'bankruptcy': -0.8, 'default': -0.7, 'debt': -0.2, 'layoff': -0.5,
    'layoffs': -0.5, 'lawsuit': -0.4, 'fraud': -0.7, 'investigation': -0.4,
}


class _TextExtractor(HTMLParser):
    """Extract plain text from HTML."""

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript'):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript'):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _html_to_text(html: str) -> str:
    """Strip HTML tags, return plain text."""
    parser = _TextExtractor()
    parser.feed(html)
    return ' '.join(parser.parts)


class ScraperBridge:
    """Parses HTML articles into ticker sentiments for tensor L0."""

    def __init__(self, tensor: UnifiedTensor, market_graph: MarketGraph):
        self.tensor = tensor
        self.mg = market_graph
        self._known_tickers = set(market_graph.tickers.keys())

    def parse_article(self, html: str) -> dict:
        """Parse a single HTML article.

        Returns:
            {
                'tickers': list of extracted ticker symbols,
                'sentiment': float in [-1, 1] (overall article sentiment),
                'ticker_sentiments': {ticker: float} per-ticker sentiment,
                'text_length': int,
            }
        """
        text = _html_to_text(html)
        words = text.split()
        text_lower = text.lower()

        # Extract tickers
        raw_matches = _TICKER_RE.findall(text)
        tickers = [t for t in raw_matches
                    if t not in _STOP_WORDS and len(t) >= 2]
        # Deduplicate preserving order
        seen = set()
        unique_tickers = []
        for t in tickers:
            if t not in seen:
                seen.add(t)
                unique_tickers.append(t)

        # Score sentiment via lexicon
        word_list = re.findall(r'[a-z]+', text_lower)
        pos_score = sum(_POSITIVE.get(w, 0.0) for w in word_list)
        neg_score = sum(_NEGATIVE.get(w, 0.0) for w in word_list)
        total = pos_score + neg_score
        n_words = max(len(word_list), 1)
        overall = float(np.clip(total / (n_words ** 0.5), -1.0, 1.0))

        # Per-ticker sentiment (context window around mentions)
        ticker_sentiments: Dict[str, float] = {}
        for ticker in unique_tickers:
            # Find positions of this ticker in text
            positions = [m.start() for m in re.finditer(
                r'(?<!\w)\$?' + re.escape(ticker) + r'(?!\w)', text)]
            if not positions:
                ticker_sentiments[ticker] = overall
                continue

            # Score words within ±50 chars of each mention
            local_scores = []
            for pos in positions:
                window = text_lower[max(0, pos - 50):pos + 50]
                local_words = re.findall(r'[a-z]+', window)
                local_s = sum(_POSITIVE.get(w, 0.0) for w in local_words)
                local_s += sum(_NEGATIVE.get(w, 0.0) for w in local_words)
                local_scores.append(local_s)
            ticker_sentiments[ticker] = float(
                np.clip(np.mean(local_scores), -1.0, 1.0))

        return {
            'tickers': unique_tickers,
            'sentiment': overall,
            'ticker_sentiments': ticker_sentiments,
            'text_length': len(text),
        }

    def inject(self, article_dict: dict):
        """Inject a parsed article's sentiment into the market graph.

        Updates sentiment on matching tickers in the MarketGraph.
        """
        for ticker, sent in article_dict.get('ticker_sentiments', {}).items():
            if ticker in self._known_tickers:
                # Blend with existing sentiment (exponential moving average)
                old = self.mg.tickers[ticker].sentiment
                self.mg.tickers[ticker].sentiment = float(
                    np.clip(0.7 * old + 0.3 * sent, -1.0, 1.0))

    def batch_inject(self, html_list: List[str], t: float = 0.0) -> dict:
        """Parse multiple articles, inject all, and update tensor L0 once.

        Returns:
            {
                'n_articles': int,
                'tickers_found': list of all unique tickers,
                'avg_sentiment': float,
                'articles': list of parse results,
            }
        """
        all_tickers = set()
        sentiments = []
        articles = []

        for html in html_list:
            parsed = self.parse_article(html)
            self.inject(parsed)
            all_tickers.update(parsed['tickers'])
            sentiments.append(parsed['sentiment'])
            articles.append(parsed)

        # Single tensor update after all articles processed
        mna = self.mg.to_mna()
        self.tensor.update_level(0, mna, t=t)

        return {
            'n_articles': len(html_list),
            'tickers_found': sorted(all_tickers),
            'avg_sentiment': float(np.mean(sentiments)) if sentiments else 0.0,
            'articles': articles,
        }
