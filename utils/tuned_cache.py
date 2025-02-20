import functools
import inspect
import random
import time
import warnings
from copy import deepcopy
from typing import Callable, Optional

from joblib import hashing

print('INFO: Loaded tuned variant of joblib cache.')

# noinspection PyProtectedMember,PyPep8
import joblib
# noinspection PyProtectedMember,PyPep8
from joblib.func_inspect import _clean_win_chars
# noinspection PyProtectedMember,PyPep8
from joblib.memory import MemorizedFunc, _FUNCTION_HASHES, NotMemorizedFunc, Memory

_FUNC_NAMES = {}


# noinspection SpellCheckingInspection
class TunedMemory(Memory):
    def cache(self, func=None, ignore=None, verbose=None, mmap_mode=False, identifier_cache_maxsize=0, cache_key: Optional[Callable] = None):
        if func is None:
            # Partial application, to be able to specify extra keyword arguments in decorators
            return functools.partial(self.cache, ignore=ignore, identifier_cache_maxsize=identifier_cache_maxsize,
                                     verbose=verbose, mmap_mode=mmap_mode, cache_key=cache_key)
        if self.store_backend is None:
            return NotMemorizedFunc(func)
        if verbose is None:
            verbose = self._verbose
        if mmap_mode is False:
            mmap_mode = self.mmap_mode
        if cache_key is not None:
            assert ignore is None
        if isinstance(func, TunedMemorizedFunc):
            func = func.func
        return TunedMemorizedFunc(func,
                                  location=self.store_backend,
                                  backend=self.backend,
                                  ignore=ignore,
                                  mmap_mode=mmap_mode,
                                  compress=self.compress,
                                  verbose=verbose,
                                  cache_key=cache_key,
                                  timestamp=self.timestamp)


class TunedMemorizedFunc(MemorizedFunc):
    identifier_cache = {}
    identifier_cache_requests = 0
    identifier_cache_misses = 0

    def __init__(self, *args, cache_key: Optional[Callable] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_key = cache_key

    def __call__(self, *args, **kwargs):
        # Also store in the in-memory store of function hashes
        if self.func not in _FUNCTION_HASHES:
            is_named_callable = (hasattr(self.func, '__name__') and
                                 self.func.__name__ != '<lambda>')
            if is_named_callable:
                # Don't do this for lambda functions or strange callable
                # objects, as it ends up being too fragile
                func_hash = self._hash_func()
                try:
                    _FUNCTION_HASHES[self.func] = func_hash
                except TypeError:
                    # Some callable are not hashable
                    pass

        # return same result as before
        return MemorizedFunc.__call__(self, *args, **kwargs)

    def _persist_input(self, duration, args, kwargs, this_duration_limit=0.5):
        """ Save a small summary of the call using json format in the
            output directory.

            output_dir: string
                directory where to write metadata.

            duration: float
                time taken by hashing input arguments, calling the wrapped
                function and persisting its output.

            args, kwargs: list and dict
                input arguments for wrapped function

            this_duration_limit: float
                Max execution time for this function before issuing a warning.
        """
        start_time = time.time()

        if isinstance(this_duration_limit, dict):
            # probably an older version of joblib with a different signature
            duration, call_id, args, kwargs, this_duration_limit = duration, args, kwargs, this_duration_limit, 0.5
        else:
            call_id =  self._get_output_identifiers(*args, **kwargs)

        # This can fail due to race-conditions with multiple
        # concurrent joblibs removing the file or the directory
        metadata = {
            "duration": duration, "time": start_time,
        }

        func_id, args_id = call_id
        self.store_backend.store_metadata([func_id, args_id], metadata)

        this_duration = time.time() - start_time
        if this_duration > this_duration_limit:
            # This persistence should be fast. It will not be if repr() takes
            # time and its output is large, because json.dump will have to
            # write a large file. This should not be an issue with numpy arrays
            # for which repr() always output a short representation, but can
            # be with complex dictionaries. Fixing the problem should be a
            # matter of replacing repr() above by something smarter.
            warnings.warn("Persisting input arguments took %.2fs to run."
                          "If this happens often in your code, it can cause "
                          "performance problems "
                          "(results will be correct in all cases). "
                          "The reason for this is probably some large input "
                          "arguments for a wrapped function."
                          % this_duration, stacklevel=5)
        return metadata

    def _get_output_identifiers(self, *args, **kwargs):
        TunedMemorizedFunc.identifier_cache_requests += 1
        cache = TunedMemorizedFunc.identifier_cache
        identifier_cache_key = None
        try:
            if self.cache_key is None:
                identifier_cache_key = (self.func, *args, frozenset(kwargs.items()))
            else:
                identifier_cache_key = (self.func, self.cache_key(*args, **kwargs))
            return cache[identifier_cache_key]
        except TypeError as e:
            if 'unhashable' in str(e):
                assert self.cache_key is None
                # return same result as before
                return MemorizedFunc._get_output_identifiers(self, *args, **kwargs)
            else:
                raise
        except KeyError:
            TunedMemorizedFunc.identifier_cache_misses += 1
            if len(cache) > 10000:  # keep cache small
                requests = TunedMemorizedFunc.identifier_cache_requests
                misses = TunedMemorizedFunc.identifier_cache_misses
                print(f'Clearing randomly half of the cache, {requests - misses} hits total, {misses} misses total')
                for k in random.choices(list(cache.keys()), k=len(cache) // 2):
                    if k not in cache:
                        continue
                    del cache[k]
            assert identifier_cache_key is not None
            cache[identifier_cache_key] = MemorizedFunc._get_output_identifiers(self, *args, **kwargs)
            return cache[identifier_cache_key]

    def _get_argument_hash(self, *args, **kwargs):
        if self.cache_key is not None:
            return hashing.hash(self.cache_key(*args, **kwargs), coerce_mmap=(self.mmap_mode is not None))
        else:
            return super()._get_argument_hash(*args, **kwargs)

    def __get__(self, instance, owner=None):
        if instance is None:  # something like type(obj).cached_method
            return self
        else:
            return functools.partial(self, instance)


old_get_func_name = joblib.func_inspect.get_func_name


def tuned_get_func_name(func, resolv_alias=True, win_characters=True):
    if (func, resolv_alias, win_characters) not in _FUNC_NAMES:
        _FUNC_NAMES[(func, resolv_alias, win_characters)] = old_get_func_name(func, resolv_alias, win_characters)

        if len(_FUNC_NAMES) > 1000:
            # keep cache small and fast
            for idx, k in enumerate(_FUNC_NAMES.keys()):
                if idx % 2:
                    del _FUNC_NAMES[k]
        # print('cache size ', len(_FUNC_NAMES))

    return deepcopy(_FUNC_NAMES[(func, resolv_alias, win_characters)])


joblib.func_inspect.get_func_name = tuned_get_func_name
joblib.memory.get_func_name = tuned_get_func_name
