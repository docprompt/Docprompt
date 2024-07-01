import asyncio
import sys
from functools import wraps


def get_to_thread():
    if sys.version_info >= (3, 9):
        return asyncio.to_thread
    else:

        def to_thread(func, /, *args, **kwargs):
            @wraps(func)
            async def wrapper():
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # If there's no running event loop, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return await loop.run_in_executor(None, func, *args, **kwargs)

            return wrapper()

        return to_thread


to_thread = get_to_thread()
