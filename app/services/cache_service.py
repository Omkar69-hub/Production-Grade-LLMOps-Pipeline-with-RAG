"""
app/services/cache_service.py — Redis-backed response cache with graceful fallback.
"""

import hashlib
import json
import logging

logger = logging.getLogger(__name__)

_redis_client = None


def _get_redis():
    """Lazily create the redis.asyncio client."""
    global _redis_client
    if _redis_client is None:
        from app.config import get_settings

        cfg = get_settings()
        if not cfg.redis_enabled:
            return None
        try:
            import redis.asyncio as aioredis

            _redis_client = aioredis.from_url(
                cfg.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
            )
        except Exception as exc:
            logger.warning("Redis connection failed — caching disabled: %s", exc)
            return None
    return _redis_client


def _cache_key(query: str, top_k: int, metadata_filter: dict | None) -> str:
    """Stable hash key that incorporates all query parameters."""
    raw = json.dumps(
        {"q": query.strip().lower(), "k": top_k, "f": metadata_filter},
        sort_keys=True,
    )
    return "rag:v1:" + hashlib.sha256(raw.encode()).hexdigest()


async def get_cached(
    query: str, top_k: int = 4, metadata_filter: dict | None = None
) -> dict | None:
    """Return cached response dict or None."""
    client = _get_redis()
    if client is None:
        return None
    try:
        key = _cache_key(query, top_k, metadata_filter)
        value = await client.get(key)
        if value:
            logger.debug("Cache HIT for key %s", key[:16])
            return json.loads(value)
    except Exception as exc:
        logger.warning("Cache GET error: %s", exc)
    return None


async def set_cached(
    query: str,
    response: dict,
    top_k: int = 4,
    metadata_filter: dict | None = None,
    ttl: int | None = None,
) -> None:
    """Store response dict in Redis with TTL."""
    client = _get_redis()
    if client is None:
        return
    try:
        from app.config import get_settings

        cfg = get_settings()
        key = _cache_key(query, top_k, metadata_filter)
        await client.setex(key, ttl or cfg.cache_ttl_seconds, json.dumps(response))
        logger.debug("Cache SET for key %s (ttl=%ds)", key[:16], ttl or cfg.cache_ttl_seconds)
    except Exception as exc:
        logger.warning("Cache SET error: %s", exc)


async def invalidate_all() -> int:
    """Delete all RAG cache keys. Returns count deleted."""
    client = _get_redis()
    if client is None:
        return 0
    try:
        keys = await client.keys("rag:v1:*")
        if keys:
            return await client.delete(*keys)
    except Exception as exc:
        logger.warning("Cache invalidation error: %s", exc)
    return 0


async def close_cache() -> None:
    """Close the Redis connection pool."""
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            pass
        _redis_client = None
