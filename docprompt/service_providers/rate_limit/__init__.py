from typing import ContextManager


class NoOpRateLimiter(ContextManager):
    """
    A no-op rate limiter that doesn't do anything
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
