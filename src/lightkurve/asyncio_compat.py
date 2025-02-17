"""Provide Python 3.7/3.8 compatibility for background I/O tasks with ``asyncio``."""
import asyncio
import functools


def create_task(coro, *, name=None):
    """create_task that supports Python 3.6."""
    if hasattr(asyncio, "create_task"):
        task = asyncio.create_task(coro)
        if name is not None and hasattr(task, "set_name"):
            task.set_name(name)
        return task
    else:
        return asyncio.ensure_future(coro)


def get_running_loop():
    """get_running_loop that supports Python 3.6."""
    if hasattr(asyncio, "get_running_loop"):
        return asyncio.get_running_loop()
    else:
        return asyncio.get_event_loop()


# Adapted from python 3.9's version:
# https://github.com/python/cpython/blob/v3.9.0/Lib/asyncio/threads.py
async def to_thread(func, *args, **kwargs):
    """Asynchronously run function *func* in a separate thread.
    Any *args and **kwargs supplied for this function are directly passed
    to *func*. Also, the current :class:`contextvars.Context` is propagated,
    allowing context variables from the main thread to be accessed in the
    separate thread.
    Return a coroutine that can be awaited to get the eventual result of *func*.

    This is a backport that supports Python 3.6.
    """
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func, *args, **kwargs)
    # Otherwise fallback to our approximation
    loop = get_running_loop()
    import contextvars

    ctx = contextvars.copy_context()  # TODO: contextvars only available in Python 3.7
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)
