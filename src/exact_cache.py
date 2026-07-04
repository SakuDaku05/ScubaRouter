"""
Exact Match Cache — completely free, zero-latency caching.
If we see the exact same query twice in a single run (which can happen
if the eval loops or benchmark repeats questions), we return the previously
calculated answer instantly and cost 0 tokens.
"""
from typing import Optional


class ExactCache:
    def __init__(self):
        self._store = {}

    def get(self, query: str) -> Optional[dict]:
        """Returns the previous full record if found."""
        key = query.strip().lower()
        if key in self._store:
            # We return a copy of the record, but force tokens to 0
            # because caching means we didn't call the API this time.
            cached_record = self._store[key].copy()
            cached_record["remote_tokens_used"] = 0
            cached_record["route"] = "cache_hit"
            cached_record["route_reason"] = "exact_match_found_in_cache"
            return cached_record
        return None

    def set(self, query: str, record: dict):
        """Stores a successful query and its record."""
        key = query.strip().lower()
        self._store[key] = record
