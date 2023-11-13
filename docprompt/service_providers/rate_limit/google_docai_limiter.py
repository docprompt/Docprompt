from typing import ContextManager


class GoogleDocumentAIRedisRateLimiter(ContextManager):
    """
    A rate limiter that uses Redis as a backend
    """

    def __init__(self, redis_client, max_calls, period):
        self.redis_client = redis_client
        self.max_calls = max_calls
        self.period = period

    def __enter__(self):
        pass
