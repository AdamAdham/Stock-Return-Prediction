from src.config.settings import ENABLE_TIMING
import time


def time_call(func, *args, label=None, **kwargs):
    if ENABLE_TIMING:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        name = label if label else func.__name__
        print(f"{name:<40} took {end - start:.6f} seconds")
        return result
    else:
        return func(*args, **kwargs)
