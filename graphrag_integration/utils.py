"""
GraphRAG Performance Utilities for Palli Sahayak

Provides caching, batch processing, async optimization, and memory management
utilities for improved GraphRAG performance.

Usage:
    from graphrag_integration.utils import (
        QueryCache,
        BatchProcessor,
        AsyncThrottler,
        MemoryManager,
    )

    # Query caching
    cache = QueryCache(maxsize=100, ttl_seconds=3600)
    cache.set("query", "method", result)
    cached = cache.get("query", "method")

    # Batch processing
    processor = BatchProcessor(batch_size=10)
    results = await processor.process(items, async_func)

    # Async throttling
    throttler = AsyncThrottler(max_concurrent=5)
    async with throttler:
        result = await expensive_operation()
"""

import asyncio
import hashlib
import logging
import time
import gc
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# QUERY CACHE
# =============================================================================

@dataclass
class CacheEntry:
    """
    Cache entry with value and metadata.

    Attributes:
        value: Cached value
        created_at: When the entry was created
        access_count: Number of times accessed
        last_accessed: Last access timestamp
    """
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired."""
        if ttl_seconds <= 0:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > ttl_seconds


class QueryCache:
    """
    LRU cache for GraphRAG queries with TTL support.

    Features:
    - LRU eviction when maxsize reached
    - Optional TTL (time-to-live) for entries
    - Hit/miss statistics tracking
    - Thread-safe operations

    Attributes:
        maxsize: Maximum number of entries
        ttl_seconds: Time-to-live in seconds (0 = no expiry)

    Example:
        cache = QueryCache(maxsize=100, ttl_seconds=3600)

        # Check cache first
        result = cache.get("What is morphine?", "local")
        if result is None:
            result = await query_engine.local_search("What is morphine?")
            cache.set("What is morphine?", "local", result)

        # Get statistics
        stats = cache.get_stats()
        print(f"Hit rate: {stats['hit_rate']:.1%}")
    """

    def __init__(self, maxsize: int = 100, ttl_seconds: int = 3600):
        """
        Initialize query cache.

        Args:
            maxsize: Maximum number of entries (default 100)
            ttl_seconds: Entry TTL in seconds (default 3600, 0 = no expiry)
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._maxsize = maxsize
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()

    def _hash_query(self, query: str, method: str) -> str:
        """
        Generate hash for cache key.

        Args:
            query: Query string
            method: Search method

        Returns:
            MD5 hash string
        """
        content = f"{query.strip().lower()}:{method.lower()}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, method: str) -> Optional[Any]:
        """
        Get cached result.

        Args:
            query: Query string
            method: Search method

        Returns:
            Cached value or None if not found/expired
        """
        key = self._hash_query(query, method)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Check expiry
        if entry.is_expired(self._ttl_seconds):
            del self._cache[key]
            self._misses += 1
            return None

        # Update access metadata and move to end (most recently used)
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        self._cache.move_to_end(key)
        self._hits += 1

        return entry.value

    def set(self, query: str, method: str, result: Any) -> None:
        """
        Cache a result.

        Args:
            query: Query string
            method: Search method
            result: Result to cache
        """
        key = self._hash_query(query, method)

        # Evict oldest if at capacity
        while len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(value=result)

    async def get_async(self, query: str, method: str) -> Optional[Any]:
        """Thread-safe async get."""
        async with self._lock:
            return self.get(query, method)

    async def set_async(self, query: str, method: str, result: Any) -> None:
        """Thread-safe async set."""
        async with self._lock:
            self.set(query, method, result)

    def invalidate(self, query: str, method: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            query: Query string
            method: Search method

        Returns:
            True if entry was found and removed
        """
        key = self._hash_query(query, method)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        if self._ttl_seconds <= 0:
            return 0

        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired(self._ttl_seconds)
        ]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self._ttl_seconds,
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"QueryCache(size={stats['size']}/{stats['maxsize']}, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )


# =============================================================================
# BATCH PROCESSOR
# =============================================================================

@dataclass
class BatchResult(Generic[T]):
    """
    Result of batch processing.

    Attributes:
        results: List of successful results
        errors: List of (index, error) tuples for failures
        duration_seconds: Total processing time
    """
    results: List[T]
    errors: List[tuple[int, Exception]]
    duration_seconds: float

    @property
    def success_count(self) -> int:
        return len(self.results)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def total_count(self) -> int:
        return self.success_count + self.error_count


class BatchProcessor:
    """
    Batch processor for efficient bulk operations.

    Features:
    - Configurable batch size
    - Concurrent batch processing
    - Progress callbacks
    - Error handling with continuation

    Attributes:
        batch_size: Number of items per batch
        max_concurrent: Maximum concurrent batches

    Example:
        processor = BatchProcessor(batch_size=10, max_concurrent=3)

        async def process_document(doc):
            return await index_document(doc)

        result = await processor.process(
            documents,
            process_document,
            on_progress=lambda p: print(f"Progress: {p}%")
        )

        print(f"Processed {result.success_count} documents")
    """

    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent: int = 3,
        retry_count: int = 0
    ):
        """
        Initialize batch processor.

        Args:
            batch_size: Items per batch (default 10)
            max_concurrent: Max concurrent batches (default 3)
            retry_count: Retries for failed items (default 0)
        """
        self._batch_size = batch_size
        self._max_concurrent = max_concurrent
        self._retry_count = retry_count
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def process(
        self,
        items: List[Any],
        processor: Callable[[Any], Any],
        on_progress: Optional[Callable[[int], None]] = None,
        on_batch_complete: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """
        Process items in batches.

        Args:
            items: Items to process
            processor: Async function to process each item
            on_progress: Progress callback (0-100)
            on_batch_complete: Batch completion callback (batch_num, total_batches)

        Returns:
            BatchResult with results and errors
        """
        start_time = time.time()
        results = []
        errors = []

        # Create batches
        batches = [
            items[i:i + self._batch_size]
            for i in range(0, len(items), self._batch_size)
        ]

        total_batches = len(batches)
        completed_batches = 0

        async def process_batch(batch_idx: int, batch: List[Any]) -> List[tuple[int, Any]]:
            nonlocal completed_batches

            async with self._semaphore:
                batch_results = []

                for item_idx, item in enumerate(batch):
                    global_idx = batch_idx * self._batch_size + item_idx

                    try:
                        if asyncio.iscoroutinefunction(processor):
                            result = await processor(item)
                        else:
                            result = processor(item)
                        batch_results.append((global_idx, result, None))
                    except Exception as e:
                        batch_results.append((global_idx, None, e))

                completed_batches += 1

                if on_batch_complete:
                    on_batch_complete(completed_batches, total_batches)

                if on_progress:
                    progress = int(completed_batches / total_batches * 100)
                    on_progress(progress)

                return batch_results

        # Process all batches concurrently
        tasks = [
            process_batch(i, batch)
            for i, batch in enumerate(batches)
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                errors.append((-1, batch_result))
                continue

            for idx, result, error in batch_result:
                if error:
                    errors.append((idx, error))
                else:
                    results.append(result)

        duration = time.time() - start_time

        return BatchResult(
            results=results,
            errors=errors,
            duration_seconds=duration
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if value < 1:
            raise ValueError("Batch size must be at least 1")
        self._batch_size = value

    def __repr__(self) -> str:
        return (
            f"BatchProcessor(batch_size={self._batch_size}, "
            f"max_concurrent={self._max_concurrent})"
        )


# =============================================================================
# ASYNC THROTTLER
# =============================================================================

class AsyncThrottler:
    """
    Async throttler for rate limiting concurrent operations.

    Features:
    - Configurable concurrency limit
    - Optional rate limiting (requests per second)
    - Context manager support
    - Statistics tracking

    Example:
        throttler = AsyncThrottler(max_concurrent=5, rate_limit=10)

        async def fetch_data(url):
            async with throttler:
                return await http_client.get(url)

        # Process many URLs with controlled concurrency
        results = await asyncio.gather(*[
            fetch_data(url) for url in urls
        ])
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        rate_limit: Optional[float] = None
    ):
        """
        Initialize throttler.

        Args:
            max_concurrent: Maximum concurrent operations
            rate_limit: Max operations per second (None = no limit)
        """
        self._max_concurrent = max_concurrent
        self._rate_limit = rate_limit
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._last_request_time = 0.0
        self._request_count = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self._semaphore.acquire()

        if self._rate_limit:
            async with self._lock:
                now = time.time()
                min_interval = 1.0 / self._rate_limit
                elapsed = now - self._last_request_time

                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)

                self._last_request_time = time.time()

        self._request_count += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._semaphore.release()
        return False

    async def acquire(self):
        """Acquire throttler slot."""
        return await self.__aenter__()

    def release(self):
        """Release throttler slot."""
        self._semaphore.release()

    def get_stats(self) -> Dict[str, Any]:
        """Get throttler statistics."""
        return {
            "max_concurrent": self._max_concurrent,
            "rate_limit": self._rate_limit,
            "request_count": self._request_count,
            "available_slots": self._semaphore._value,
        }

    def __repr__(self) -> str:
        return (
            f"AsyncThrottler(max_concurrent={self._max_concurrent}, "
            f"rate_limit={self._rate_limit})"
        )


# =============================================================================
# MEMORY MANAGER
# =============================================================================

class MemoryManager:
    """
    Memory management utilities for GraphRAG operations.

    Features:
    - Memory usage tracking
    - Automatic cleanup triggers
    - DataFrame memory optimization
    - Garbage collection management

    Example:
        manager = MemoryManager(threshold_mb=500)

        # Check memory before loading
        if manager.should_cleanup():
            manager.cleanup()

        # Load large data
        df = pd.read_parquet("entities.parquet")

        # Optimize DataFrame memory
        df = manager.optimize_dataframe(df)

        # Get memory report
        report = manager.get_memory_report()
    """

    def __init__(self, threshold_mb: int = 500):
        """
        Initialize memory manager.

        Args:
            threshold_mb: Memory threshold in MB for cleanup triggers
        """
        self._threshold_mb = threshold_mb
        self._cleanup_count = 0
        self._bytes_freed = 0

    def get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback without psutil
            return sys.getsizeof(gc.get_objects()) / (1024 * 1024)

    def should_cleanup(self) -> bool:
        """
        Check if cleanup is needed based on threshold.

        Returns:
            True if memory usage exceeds threshold
        """
        return self.get_memory_usage_mb() > self._threshold_mb

    def cleanup(self, generations: int = 2) -> int:
        """
        Perform garbage collection.

        Args:
            generations: Number of generations to collect (0-2)

        Returns:
            Number of objects collected
        """
        before = self.get_memory_usage_mb()

        collected = 0
        for gen in range(generations + 1):
            collected += gc.collect(gen)

        after = self.get_memory_usage_mb()
        freed = max(0, before - after)

        self._cleanup_count += 1
        self._bytes_freed += freed * 1024 * 1024

        logger.debug(f"Memory cleanup: freed {freed:.1f}MB, collected {collected} objects")

        return collected

    def optimize_dataframe(self, df: Any) -> Any:
        """
        Optimize DataFrame memory usage.

        Downcasts numeric types and converts object columns to category
        where appropriate.

        Args:
            df: pandas DataFrame

        Returns:
            Optimized DataFrame
        """
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            return df

        if not isinstance(df, pd.DataFrame):
            return df

        for col in df.columns:
            col_type = df[col].dtype

            if col_type == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif col_type == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif col_type == 'object':
                num_unique = df[col].nunique()
                num_total = len(df[col])
                # Convert to category if low cardinality
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')

        return df

    def get_memory_report(self) -> Dict[str, Any]:
        """
        Get memory usage report.

        Returns:
            Dictionary with memory statistics
        """
        current_mb = self.get_memory_usage_mb()

        return {
            "current_mb": round(current_mb, 2),
            "threshold_mb": self._threshold_mb,
            "above_threshold": current_mb > self._threshold_mb,
            "cleanup_count": self._cleanup_count,
            "total_freed_mb": round(self._bytes_freed / (1024 * 1024), 2),
            "gc_stats": gc.get_stats(),
        }

    def __repr__(self) -> str:
        return (
            f"MemoryManager(threshold_mb={self._threshold_mb}, "
            f"current_mb={self.get_memory_usage_mb():.1f})"
        )


# =============================================================================
# CACHING DECORATOR
# =============================================================================

def cached_query(cache: QueryCache, method: str):
    """
    Decorator to cache query results.

    Args:
        cache: QueryCache instance
        method: Search method name

    Example:
        cache = QueryCache(maxsize=100)

        @cached_query(cache, "local")
        async def local_search(query: str):
            return await expensive_search(query)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(query: str, *args, **kwargs):
            # Check cache
            cached = cache.get(query, method)
            if cached is not None:
                logger.debug(f"Cache hit for {method} query: {query[:50]}...")
                return cached

            # Execute and cache
            result = await func(query, *args, **kwargs)
            cache.set(query, method, result)

            return result

        return wrapper
    return decorator


# =============================================================================
# TIMING UTILITIES
# =============================================================================

class Timer:
    """
    Simple timer for performance measurement.

    Example:
        with Timer() as t:
            result = await expensive_operation()

        print(f"Operation took {t.elapsed:.2f}s")
    """

    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._start is None:
            return 0.0
        end = self._end or time.perf_counter()
        return end - self._start

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000


async def timed_operation(
    name: str,
    operation: Callable,
    *args,
    **kwargs
) -> tuple[Any, float]:
    """
    Execute operation and return result with timing.

    Args:
        name: Operation name for logging
        operation: Async callable to execute
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Tuple of (result, elapsed_seconds)
    """
    start = time.perf_counter()

    try:
        if asyncio.iscoroutinefunction(operation):
            result = await operation(*args, **kwargs)
        else:
            result = operation(*args, **kwargs)

        elapsed = time.perf_counter() - start
        logger.debug(f"{name} completed in {elapsed:.3f}s")

        return result, elapsed

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"{name} failed after {elapsed:.3f}s: {e}")
        raise
