from typing import List, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class QueryRanker:
    """Ranks queries by frequency to determine top-N for caching."""

    def __init__(self, usage_analytics):
        self.usage_analytics = usage_analytics

    async def get_top_queries(self, n: int = 20, language: Optional[str] = None) -> List[str]:
        """
        Returns the top N most frequent queries.
        Falls back to DEFAULT_TOP_QUERIES if analytics unavailable.
        """
        try:
            if hasattr(self.usage_analytics, "get_query_frequencies"):
                frequencies = await self.usage_analytics.get_query_frequencies(
                    days=30,
                    language=language,
                )
                if frequencies:
                    sorted_queries = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
                    return [q for q, _ in sorted_queries[:n]]
        except Exception as e:
            logger.warning(f"Could not get query frequencies: {e}")

        return None
