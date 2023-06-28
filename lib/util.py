import datetime
import faulthandler
import functools
import gc
import inspect
import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
import typing
from bisect import bisect_left
from enum import Enum
from itertools import chain, combinations
from math import log, isnan, nan, floor, log10, gcd
from numbers import Number
from shutil import copyfile
from subprocess import check_output, CalledProcessError, PIPE
from threading import RLock
from types import FunctionType
from typing import Union, Tuple, List, Optional, Dict, Any, Type, Callable
# noinspection PyUnresolvedReferences
from unittest import TestCase, mock

import cachetools
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
import scipy.optimize
import scipy.stats
import tabulate
from scipy.ndimage import zoom

from lib import stack_tracer, print_exc_plus
from lib.my_logger import logging

X = Y = Z = float


class KnownIssue(Exception):
    """
    This means the code is not working and should not be used but still too valuable to be deleted
    """
    pass


def powerset(iterable, largest_first=False):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    sizes = range(len(s) + 1)
    if largest_first:
        sizes = reversed(sizes)
    return chain.from_iterable(combinations(s, r) for r in sizes)


def plot_with_conf(x, y_mean, y_conf, alpha=0.5, **kwargs):
    ax = kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(x, y_mean, **kwargs)
    y_mean = np.array(y_mean)
    y_conf = np.array(y_conf)
    lb = y_mean - y_conf
    ub = y_mean + y_conf

    ax.fill_between(x, lb, ub, facecolor=base_line.get_color(), alpha=alpha)


def choice(sequence, probabilities):
    # if sum(probabilities) != 1:
    #     raise AssertionError('Probabilities must sum to 1')
    r = random.random()
    for idx, c in enumerate(sequence):
        r -= probabilities[idx]
        if r < 0:
            return c
    raise AssertionError('Probabilities must sum to 1')


def local_timezone():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(0))).astimezone().tzinfo


def print_attributes(obj, include_methods=False, ignore=None):
    if ignore is None:
        ignore = []
    for attr in dir(obj):
        if attr in ignore:
            continue
        if attr.startswith('_'):
            continue
        if not include_methods and callable(obj.__getattr__(attr)):
            continue
        print(attr, ':', obj.__getattr__(attr).__class__.__name__, ':', obj.__getattr__(attr))


def attr_dir(obj, include_methods=False, ignore=None):
    if ignore is None:
        ignore = []
    return {attr: obj.__getattr__(attr)
            for attr in dir(obj)
            if not attr.startswith('_') and (
                    include_methods or not callable(obj.__getattr__(attr))) and attr not in ignore}


def zoom_to_shape(a: np.ndarray, shape: Tuple, mode: str = 'smooth', verbose=1):
    from keras import backend
    a = np.array(a, dtype=backend.floatx())  # also does a copy
    shape_dim = len(a.shape)
    if len(a.shape) != len(shape):
        raise ValueError('The shapes must have the same dimension but were len({0}) = {1} (original) '
                         'and len({2}) = {3} desired.'.format(a.shape, len(a.shape), shape, len(shape)))
    if len(shape) == 0:
        return a
    zoom_factors = tuple(shape[idx] / a.shape[idx] for idx in range(shape_dim))

    def _current_index_in_old_array():
        return tuple(slice(0, length) if axis != current_axis else slice(current_pixel_index, current_pixel_index + 1)
                     for axis, length in enumerate(a.shape))

    def _current_pixel_shape():
        return tuple(length if axis != current_axis else 1
                     for axis, length in enumerate(a.shape))

    def _current_result_index():
        return tuple(
            slice(0, length) if axis != current_axis else slice(pixel_index_in_result, pixel_index_in_result + 1)
            for axis, length in enumerate(a.shape))

    def _current_result_shape():
        return tuple(orig_length if axis != current_axis else shape[axis]
                     for axis, orig_length in enumerate(a.shape))

    if mode == 'constant':
        result = zoom(a, zoom_factors)
        assert result.shape == shape
        return result
    elif mode == 'smooth':
        result = a
        for current_axis, zoom_factor in sorted(enumerate(zoom_factors), key=lambda x: x[1]):
            result = np.zeros(_current_result_shape(), dtype=backend.floatx())
            # current_length = a.shape[current_axis]
            desired_length = shape[current_axis]
            current_pixel_index = 0
            current_pixel_part = 0  # how much of the current pixel is already read
            for pixel_index_in_result in range(desired_length):
                pixels_remaining = 1 / zoom_factor
                pixel_sum = np.zeros(_current_pixel_shape())
                while pixels_remaining + current_pixel_part > 1:
                    pixel_sum += (1 - current_pixel_part) * a[_current_index_in_old_array()]
                    current_pixel_index += 1
                    pixels_remaining -= (1 - current_pixel_part)
                    current_pixel_part = 0

                # the remaining pixel_part
                try:
                    pixel_sum += pixels_remaining * a[_current_index_in_old_array()]
                except (IndexError, ValueError):
                    if verbose:
                        print('WARNING: Skipping {0} pixels because of numerical imprecision.'.format(pixels_remaining))
                else:
                    current_pixel_part += pixels_remaining

                # insert to result
                pixel_sum *= zoom_factor

                result[_current_result_index()] = pixel_sum
            a = result

        assert result.shape == shape
        return result
    else:
        return NotImplementedError('Mode not available.')


def profile_wall_time_instead_if_profiling():
    try:
        import yappi
    except ModuleNotFoundError:
        return
    currently_profiling = len(yappi.get_func_stats())
    if currently_profiling and yappi.get_clock_type() != 'wall':
        yappi.stop()
        print('Profiling wall time instead of cpu time.')
        yappi.clear_stats()
        yappi.set_clock_type("wall")
        yappi.start()


def dummy_computation(*_args, **_kwargs):
    pass


def backup_file(filename):
    copyfile(filename, backup_file_path(filename))


def backup_file_path(filename):
    return filename + time.strftime("%Y%m%d") + '.bak'


# noinspection SpellCheckingInspection
def my_tabulate(data, tablefmt='pipe', **params):
    if data == [] and 'headers' in params:
        data = [(None for _ in params['headers'])]
    tabulate.MIN_PADDING = 0
    return tabulate.tabulate(data, tablefmt=tablefmt, **params)


def ce_loss(y_true, y_predicted):
    return -(y_true * log(y_predicted) + (1 - y_true) * log(1 - y_predicted))


class DontSaveResultsError(Exception):
    pass


def multinomial(n, bins):
    if bins == 0:
        if n > 0:
            raise ValueError('Cannot distribute to 0 bins.')
        return []
    remaining = n
    results = []
    for i in range(bins - 1):
        from numpy.random.mtrand import binomial
        x = binomial(remaining, 1 / (bins - i))
        results.append(x)
        remaining -= x

    results.append(remaining)
    return results


class UnknownTypeError(Exception):
    pass


# def shape_analysis(xs):
#     composed_dtypes = [list, tuple, np.ndarray, dict, set]
#     base_dtypes = [str, int, float, type, object]  # TODO add class and superclass of xs first element
#     all_dtypes = composed_dtypes + base_dtypes
#     if isinstance(xs, np.ndarray):
#         outer_brackets = ('[', ']')
#         shape = xs.shape
#         dtype = xs.dtype
#     elif isinstance(xs, tuple):
#         outer_brackets = ('(', ')')
#         shape = len(xs)
#         dtype = [t for t in all_dtypes if all(isinstance(x, t) for x in xs)][0]
#     elif isinstance(xs, list):
#         outer_brackets = ('[', ']')
#         shape = len(xs)
#         dtype = [t for t in all_dtypes if all(isinstance(x, t) for x in xs)][0]
#     elif isinstance(xs, dict) or isinstance(xs, set):
#         outer_brackets = ('{', '}')
#         shape = len(xs)
#         dtype = [t for t in all_dtypes if all(isinstance(x, t) for x in xs)][0]
#     elif any(isinstance(xs, t) for t in base_dtypes):
#         for t in base_dtypes:
#             if isinstance(xs, t):
#                 return str(t.__name__)
#         raise AssertionError('This should be unreachable.')
#     else:
#         raise UnknownTypeError('Unknown type:' + type(xs).__name__)
#
#     if shape and shape != '?':
#         return outer_brackets[0] + str(xs.shape) + ' * ' + str(dtype) + outer_brackets[1]
#     else:
#         return outer_brackets[0] + outer_brackets[1]


def beta_conf_interval_mle(data, conf=0.95):
    if len(data) <= 1:
        return 0, 1  # overestimates the interval
    if any(d < 0 or d > 1 or isnan(d) for d in data):
        return nan, nan
    if numpy.var(data) == 0:
        return numpy.mean(data), numpy.mean(data)
    epsilon = 1e-3
    # adjusted_data = data.copy()
    # for idx in range(len(adjusted_data)):
    #     adjusted_data[idx] *= (1 - 2 * epsilon)
    #     adjusted_data[idx] += epsilon
    alpha, beta, _, _ = scipy.stats.beta.fit(data, floc=-epsilon, fscale=1 + 2 * epsilon)

    lower, upper = scipy.stats.beta.interval(alpha=conf, a=alpha, b=beta)
    if lower < 0:
        lower = 0
    if upper < 0:
        upper = 0
    if lower > 1:
        lower = 1
    if upper > 1:
        upper = 1
    return lower, upper


def gamma_conf_interval_mle(data, conf=0.95) -> Tuple[float, float]:
    if len(data) == 0:
        return nan, nan
    if len(data) == 1:
        return nan, nan
    if any(d < 0 or isnan(d) for d in data):
        return nan, nan
    if numpy.var(data) == 0:
        return numpy.mean(data).item(), 0
    alpha, _, scale = scipy.stats.gamma.fit(data, floc=0)

    lower, upper = scipy.stats.gamma.interval(alpha=conf, a=alpha, scale=scale)
    if lower < 0:
        lower = 0
    if upper < 0:
        upper = 0
    return lower, upper


beta_quantile_cache = cachetools.LRUCache(maxsize=10)


@cachetools.cached(cache=beta_quantile_cache, key=lambda x1, p1, x2, p2, guess: (x1, x2, p1, p2))
def beta_parameters_quantiles(x1, p1, x2, p2, guess=(3, 3)):
    "Find parameters for a beta random variable X; so; that; P(X > x1) = p1 and P(X > x2) = p2.; "

    def square(x):
        return x * x

    def objective(v):
        (a, b) = v
        temp = square(scipy.stats.beta.cdf(x1, a, b) - p1)
        temp += square(scipy.stats.beta.cdf(x2, a, b) - p2)
        return temp

    xopt = scipy.optimize.fmin(objective, guess, disp=False)
    return (xopt[0], xopt[1])


def beta_conf_interval_quantile(data, conf=0.95, quantiles=(0.25, 0.75)):
    if len(data) <= 1:
        return 0, 1  # overestimates the interval
    mu = numpy.mean(data)
    v = numpy.var(data)
    data = numpy.array(data)
    if v == 0:
        return mu, mu
    lower = numpy.quantile(data, quantiles[0])
    upper = numpy.quantile(data, quantiles[1])

    alpha_guess = mu ** 2 * ((1 - mu) / v - 1 / mu)
    beta_guess = alpha_guess * (1 / mu - 1)

    alpha, beta = beta_parameters_quantiles(lower, quantiles[0], upper, quantiles[1], (alpha_guess, beta_guess))
    return scipy.stats.beta.interval(alpha=conf, a=alpha, b=beta)


def beta_stats_quantile(data, quantiles=(0.25, 0.75)):
    if len(data) <= 1:
        return 0, 1  # overestimates the interval
    data = numpy.array(data)
    mu = numpy.mean(data)
    v = numpy.var(data)
    if v == 0:
        return mu, mu
    lower = numpy.quantile(data, quantiles[0])
    upper = numpy.quantile(data, quantiles[1])

    alpha_guess = mu ** 2 * ((1 - mu) / v - 1 / mu)
    beta_guess = alpha_guess * (1 / mu - 1)

    alpha, beta = beta_parameters_quantiles(lower, quantiles[0], upper, quantiles[1], (alpha_guess, beta_guess))
    return scipy.stats.beta.stats(a=alpha, b=beta)


def beta_stats_mle(data):
    if len(data) == 0:
        return nan, nan
    if len(data) == 1:
        return nan, nan
    if any(d < 0 or d > 1 or isnan(d) for d in data):
        return nan, nan
    if numpy.var(data) == 0:
        return numpy.mean(data), 0
    epsilon = 1e-4
    # adjusted_data = data.copy()
    # for idx in range(len(adjusted_data)):
    #     adjusted_data[idx] *= (1 - 2 * epsilon)
    #     adjusted_data[idx] += epsilon
    alpha, beta, _, _ = scipy.stats.beta.fit(data, floc=-epsilon, fscale=1 + 2 * epsilon)

    return scipy.stats.beta.stats(a=alpha, b=beta)


def gamma_stats_mle(data):
    if len(data) == 0:
        return nan, nan
    if len(data) == 1:
        return nan, nan
    if any(d < 0 or isnan(d) for d in data):
        return nan, nan
    if numpy.var(data) == 0:
        return numpy.mean(data), 0
    alpha, _, scale = scipy.stats.gamma.fit(data, floc=0)

    return scipy.stats.gamma.stats(a=alpha, scale=scale)


beta_stats = beta_stats_quantile
beta_conf_interval = beta_conf_interval_quantile
gamma_stats = gamma_stats_mle
gamma_conf_interval = gamma_conf_interval_mle


def split_df_list(df, target_column):
    """
    df = data frame to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a data frame with each entry for the target column separated, with each element moved into a new row.
    The values in the other columns are duplicated across the newly divided rows.

    SOURCE: https://gist.github.com/jlln/338b4b0b55bd6984f883
    """

    def split_list_to_rows(row, row_accumulator):
        split_row = json.loads(row[target_column])
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)

    new_rows = []
    df.apply(split_list_to_rows, axis=1, args=(new_rows,))
    new_df = pandas.DataFrame(new_rows)
    return new_df


try:
    import winsound as win_sound


    def beep(*args, **kwargs):
        win_sound.Beep(*args, **kwargs)
except ImportError:
    win_sound = None


    def beep(*_args, **_kwargs):
        pass


def round_to_digits(x, d):
    if x == 0:
        return 0
    if isnan(x):
        return nan
    try:
        return round(x, d - 1 - int(floor(log10(abs(x)))))
    except OverflowError:
        return x


def gc_if_memory_error(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except MemoryError:
        print('Starting garbage collector')
        gc.collect()
        return f(*args, **kwargs)


def assert_not_empty(x):
    assert len(x)
    return x


def validation_steps(validation_dataset_size, maximum_batch_size):
    batch_size = gcd(validation_dataset_size, maximum_batch_size)
    steps = validation_dataset_size // batch_size
    assert batch_size * steps == validation_dataset_size
    return batch_size, steps


def functional_dependency_trigger(connection: sqlite3.Connection,
                                  table_name: str,
                                  determining_columns: List[str],
                                  determined_columns: List[str],
                                  exist_ok: bool, ):
    cursor = connection.cursor()
    # possible_performance_improvements
    determined_columns = [c for c in determined_columns if c not in determining_columns]
    trigger_base_name = '_'.join([table_name] + determining_columns + ['determine'] + determined_columns)

    error_message = ','.join(determining_columns) + ' must uniquely identify ' + ','.join(determined_columns)

    # when inserting check if there is already an entry with these values
    cursor.execute(f'''
    CREATE TRIGGER {'IF NOT EXISTS' if exist_ok else ''} {trigger_base_name}_after_insert
    BEFORE INSERT ON {table_name}
    WHEN EXISTS(SELECT * FROM {table_name}
         WHERE ({' AND '.join(f'NEW.{c} IS NOT NULL AND {c} = NEW.{c}' for c in determining_columns)})
         AND ({' OR '.join(f'{c} != NEW.{c}' for c in determined_columns)}))
    BEGIN SELECT RAISE(ROLLBACK, '{error_message}'); END
    ''')

    # when updating check if there is already an entry with these values (only if changed)
    cursor.execute(f'''
    CREATE TRIGGER {'IF NOT EXISTS' if exist_ok else ''} {trigger_base_name}_after_update
    BEFORE UPDATE ON {table_name}
    WHEN EXISTS(SELECT * FROM {table_name}
         WHERE ({' AND '.join(f'NEW.{c} IS NOT NULL AND {c} = NEW.{c}' for c in determining_columns)})
         AND ({' OR '.join(f'{c} != NEW.{c}' for c in determined_columns)}))
    BEGIN SELECT RAISE(ROLLBACK, '{error_message}'); END
    ''')


def heatmap_from_points(x, y,
                        x_lim: Optional[Union[int, Tuple[int, int]]] = None,
                        y_lim: Optional[Union[int, Tuple[int, int]]] = None,
                        gridsize=30):
    if isinstance(x_lim, Number):
        x_lim = (x_lim, x_lim)
    if isinstance(y_lim, Number):
        y_lim = (y_lim, y_lim)

    plt.hexbin(x, y, gridsize=gridsize, cmap=matplotlib.cm.jet, bins=None)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    cb = plt.colorbar()
    cb.set_label('mean value')


def strptime(date_string, fmt):
    return datetime.datetime(*(time.strptime(date_string, fmt)[0:6]))


class PrintLineRLock(RLock().__class__):
    def __init__(self, *args, name='', **kwargs):
        # noinspection PyArgumentList
        super().__init__(*args, **kwargs)
        self.name = name

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        print(f'Trying to acquire Lock {self.name}')
        result = RLock.acquire(self, blocking, timeout)
        print(f'Acquired Lock {self.name}')
        return result

    def release(self) -> None:
        print(f'Trying to release Lock {self.name}')
        # noinspection PyNoneFunctionAssignment
        result = RLock.release(self)
        print(f'Released Lock {self.name}')
        return result

    def __enter__(self, *args, **kwargs):
        print('Trying to enter Lock')
        # noinspection PyArgumentList
        super().__enter__(*args, **kwargs)
        print('Entered Lock')

    def __exit__(self, *args, **kwargs):
        print('Trying to exit Lock')
        super().__exit__(*args, **kwargs)
        print('Exited Lock')


class CallCounter():
    def __init__(self, f):
        self.f = f
        self.calls = 0
        self.__name__ = f.__name__

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.f(*args, **kwargs)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__class__.__name__ + repr(self.__dict__)


def test_with_timeout(timeout=2):
    def wrapper(f):
        from lib.threading_timer_decorator import exit_after
        f = exit_after(timeout)(f)

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            try:
                print(f'Running this test with timeout: {timeout}')
                return f(*args, **kwargs)
            except KeyboardInterrupt:
                raise AssertionError(f'Test took longer than {timeout} seconds')

        return wrapped

    return wrapper


def lru_cache_by_id(maxsize):
    return cachetools.cached(cachetools.LRUCache(maxsize=maxsize), key=id)


def iff_patch(patch: mock._patch):
    def decorator(f):
        def wrapped(*args, **kwargs):
            with patch:
                f(*args, **kwargs)
            try:
                f(*args, **kwargs)
            except:
                pass
            else:
                raise AssertionError('Test did not fail without patch')

        return wrapped

    return decorator


def iff_not_patch(patch: mock._patch):
    def decorator(f):
        def wrapped(*args, **kwargs):
            f(*args, **kwargs)
            try:
                with patch:
                    f(*args, **kwargs)
            except Exception as e:
                pass
            else:
                raise AssertionError('Test did not fail with patch')

        return wrapped

    return decorator


EMAIL_CRASHES_TO = []


def list_logger(base_logging_function, store_in_list: list):
    def print_and_store(*args, **kwargs):
        base_logging_function(*args, **kwargs)
        store_in_list.extend(args)

    return print_and_store


def main_wrapper(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        # import lib.stack_tracer
        import __main__
        # does not help much
        # monitoring_thread = hanging_threads.start_monitoring(seconds_frozen=180, test_interval=1000)
        os.makedirs('logs', exist_ok=True)
        stack_tracer.trace_start('logs/' + os.path.split(__main__.__file__)[-1] + '.html', interval=5)
        faulthandler.enable()
        profile_wall_time_instead_if_profiling()

        # noinspection PyBroadException
        try:
            return f(*args, **kwargs)
        except KeyboardInterrupt:
            error_messages = []
            print_exc_plus.print_exc_plus(print=list_logger(logging.error, error_messages),
                                          serialize_to='logs/' + os.path.split(__main__.__file__)[-1] + '.dill')
        except:
            error_messages = []
            print_exc_plus.print_exc_plus(print=list_logger(logging.error, error_messages),
                                          serialize_to='logs/' + os.path.split(__main__.__file__)[-1] + '.dill')
            for recipient in EMAIL_CRASHES_TO:
                from jobs.sending_emails import send_mail
                send_mail.create_simple_mail_via_gmail(body='\n'.join(error_messages), filepath=None, excel_name=None, to_mail=recipient, subject='Crash report')
        finally:
            logging.info('Terminated.')
            total_time = time.perf_counter() - start
            faulthandler.disable()
            stack_tracer.trace_stop()
            frequency = 2000
            duration = 500
            beep(frequency, duration)
            print('Total time', total_time)
            try:
                from algorithm_development.metatrader import ZeroMQ_Connector
                ZeroMQ_Connector.DWX_ZeroMQ_Connector.deactivate_all()
            except ImportError:
                pass

    return wrapper


def required_size_for_safe_rotation(base: Tuple[X, Y, Z], rotate_range_deg) -> Tuple[X, Y, Z]:
    if abs(rotate_range_deg) > 45:
        raise NotImplementedError
    if abs(rotate_range_deg) > 0:
        x_length = base[2] * math.sin(rotate_range_deg / 180 * math.pi) + base[1] * math.cos(
            rotate_range_deg / 180 * math.pi)
        y_length = base[2] * math.cos(rotate_range_deg / 180 * math.pi) + base[1] * math.sin(
            rotate_range_deg / 180 * math.pi)
        result = (base[0],
                  x_length,
                  y_length,)
    else:
        result = base
    return result


def round_to_closest_value(x, values, assume_sorted=False):
    if not assume_sorted:
        values = sorted(values)
    next_largest = bisect_left(values, x)  # binary search
    if next_largest == 0:
        return values[0]
    if next_largest == len(values):
        return values[-1]
    next_smallest = next_largest - 1
    smaller = values[next_smallest]
    larger = values[next_largest]
    if abs(smaller - x) < abs(larger - x):
        return smaller
    else:
        return larger


def binary_search(a, x, lo=0, hi=None):
    hi = hi if hi is not None else len(a)  # hi defaults to len(a)

    pos = bisect_left(a, x, lo, hi)  # find insertion position

    return pos if pos != hi and a[pos] == x else -1  # don't walk off the end


def ceil_to_closest_value(x, values):
    values = sorted(values)
    next_largest = bisect_left(values, x)  # binary search
    if next_largest < len(values):
        return values[next_largest]
    else:
        return values[-1]  # if there is no larger value use the largest one


def print_progress_bar(iteration, total, prefix='Progress:', suffix='', decimals=1, length=50, fill='█',
                       print_eta=True):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:" + str(4 + decimals) + "." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    if getattr(print_progress_bar, 'last_printed_value', None) == (prefix, bar, percent, suffix):
        return
    print_progress_bar.last_printed_value = (prefix, bar, percent, suffix)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
    # Print New Line on Complete
    if iteration == total:
        print()


def get_all_subclasses(klass):
    all_subclasses = []

    for subclass in klass.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def latin1_json(data):
    return json.dumps(data, ensure_ascii=False).encode('latin-1')


def utf8_json(data):
    return json.dumps(data, ensure_ascii=False).encode('utf-8')


def l2_norm(v1, v2):
    if len(v1) != len(v2):
        raise ValueError('Both vectors must be of the same size')
    return math.sqrt(sum([(x1 - x2) * (x1 - x2) for x1, x2 in zip(v1, v2)]))


def allow_additional_unused_keyword_arguments(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        allowed_kwargs = [param.name for param in inspect.signature(func).parameters.values()]
        allowed_kwargs = {a: kwargs[a] for a in kwargs if a in allowed_kwargs}
        return func(*args, **allowed_kwargs)

    return wrapper


def copy_and_rename_method(func, new_name):
    funcdetails = [
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__
    ]
    old_name = func.__name__
    # copy
    # new_func = dill.loads(dill.dumps(func))
    new_func = FunctionType(*funcdetails)
    assert new_func is not funcdetails
    # rename
    new_func.__name__ = new_name
    assert func.__name__ is old_name
    return new_func


def rename(new_name):
    def decorator(f):
        f.__name__ = new_name
        return f

    return decorator


class LogicError(Exception):
    pass


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def shorten_name(name):
    name = re.sub(r'\s+', r' ', str(name))
    name = name.replace(', ', ',')
    name = name.replace(', ', ',')
    name = name.replace(' ', '_')
    return re.sub(r'([A-Za-z])[a-z]*_?', r'\1', str(name))


def array_analysis(a: numpy.ndarray):
    print(f'  Shape: {a.shape}')
    mean = a.mean()
    print(f'  Mean: {mean}')
    print(f'  Std: {a.std()}')
    print(f'  Min, Max: {a.min(), a.max()}')
    print(f'  Mean absolute: {numpy.abs(a).mean()}')
    print(f'  Mean square: {numpy.square(a).mean()}')
    print(f'  Mean absolute difference from mean: {numpy.abs(a - mean).mean()}')
    print(f'  Mean squared difference from mean: {numpy.square(a - mean).mean()}')
    nonzero = numpy.count_nonzero(a)
    print(f'  Number of non-zeros: {nonzero}')
    print(f'  Number of zeros: {numpy.prod(a.shape) - nonzero}')
    if a.shape[-1] > 1 and a.shape[-1] <= 1000:
        # last dim is probably the number of classes
        print(f'  Class counts: {numpy.count_nonzero(a, axis=tuple(range(len(a.shape) - 1)))}')


def current_year_begin():
    return datetime.datetime(datetime.datetime.today().year, 1, 1).timestamp()


def current_day_begin():
    return datetime.datetime.today().timestamp() // (3600 * 24) * (3600 * 24)


def current_second_begin():
    return floor(datetime.datetime.today().timestamp())


def running_workers(executor):
    print(next(iter(executor._threads)).__dict__)
    return sum(1 for t in executor._threads
               if t == 1)


class Bunch(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__.update(kwargs)

    def add_method(self, m):
        setattr(self, m.__name__, functools.partial(m, self))


def queued_calls(executor):
    return len(executor._work_queue.queue)


def single_use_function(f: Callable):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not wrapper.already_called:
            wrapper.already_called = True
            return f(*args, **kwargs)
        else:
            raise RuntimeError(f'{f} is supposed to be called only once.')

    wrapper.already_called = False
    return wrapper


def single_use_method(f: Callable):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, f'_already_called_{id(wrapper)}') or not getattr(self, f'_already_called_{id(wrapper)}'):
            setattr(self, f'_already_called_{id(wrapper)}', True)
            return f(self, *args, **kwargs)
        else:
            raise RuntimeError(f'{f} is supposed to be called only once per object.')

    return wrapper


class JSONConvertible:
    SUBCLASSES_BY_NAME: Dict[str, Type['JSONConvertible']] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        JSONConvertible.SUBCLASSES_BY_NAME[cls.__name__] = cls

    @classmethod
    def subclasses(cls, strict=False):
        return [t for t in cls.SUBCLASSES_BY_NAME.values() if issubclass(t, cls) and (not strict or t != cls)]

    def to_json(self) -> Dict[str, Any]:
        raise NotImplementedError('Abstract method')

    @staticmethod
    def from_json(data: Dict[str, Any]):
        cls = JSONConvertible.SUBCLASSES_BY_NAME[data['type']]
        if cls.from_json is JSONConvertible.from_json:
            raise NotImplementedError(f'Abstract method from_json of class {cls.__name__}')
        return cls.from_json(data)

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        fields = cls.field_types()
        result: Dict[str, Any] = {
            "type": "object",
            "properties": {}
        }
        for field, field_type in fields.items():
            result['properties'][field] = schema_by_python_type(field_type)

        subclasses = cls.subclasses(strict=True)
        if len(subclasses) > 0:
            result = {'oneOf': [result] + [{'$ref': f'#/components/schemas/{s.__name__}'} for s in subclasses]}
        return result

    @classmethod
    def schema_for_referencing(cls):
        return {'$ref': f'#/components/schemas/{cls.__name__}'}

    @classmethod
    def schema_python_types(cls) -> Dict[str, Type]:
        fields = typing.get_type_hints(cls.__init__)
        result: Dict[str, Type] = {}
        for field, field_type in fields.items():
            result[field] = field_type
        return result

    @classmethod
    def field_types(cls) -> Dict[str, Type]:
        return typing.get_type_hints(cls.__init__)


class EBC(JSONConvertible):
    def __eq__(self, other):
        return type(other) == type(self) and self.__dict__ == other.__dict__

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return f'{type(self).__name__}(**' + str(self.__dict__) + ')'

    def to_json(self) -> Dict[str, Any]:
        def obj_to_json(o):
            if isinstance(o, JSONConvertible):
                return o.to_json()
            elif isinstance(o, numpy.ndarray):
                return array_to_json(o)
            elif isinstance(o, list):
                return list_to_json(o)
            elif isinstance(o, dict):
                return dict_to_json(o)
            else:
                return o  # probably a primitive type

        def dict_to_json(d: Dict[str, Any]):
            return {k: obj_to_json(v) for k, v in d.items()}

        def list_to_json(l: list):
            return [obj_to_json(v) for v in l]

        def array_to_json(o: numpy.ndarray):
            return o.tolist()

        result: Dict[str, Any] = {
            'type': type(self).__name__,
            **self.__dict__,
        }
        for k in result:
            result[k] = obj_to_json(result[k])
        return result

    @staticmethod
    def from_json(data: Dict[str, Any]):
        cls = EBC.SUBCLASSES_BY_NAME[data['type']]
        return class_from_json(cls, data)

    @classmethod
    def from_json_file(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_json(data)


class _Container(EBC):
    def __init__(self, item):
        self.item = item


def container_to_json(obj: Union[list, dict, numpy.ndarray]) -> Dict[str, Any]:
    # Useful when converting list or dicts of json-convertible objects to json (recursively)
    e = _Container(obj)
    return e.to_json()['item']


def schema_by_python_type(t: Union[Type, dict]):
    if t == 'TODO':
        return t
    if isinstance(t, dict):
        return {
            "type": "object",
            "properties": {n: schema_by_python_type(t2) for n, t2 in t.items()},
            'required': [n for n in t],
        }
    elif t is int:
        return {'type': 'integer'}
    elif t is Any:
        return {}
    elif t is float:
        return {'type': 'number'}
    elif t is str:
        return {'type': 'string'}
    elif t is bool:
        return {'type': 'boolean'}
    elif t is type(None) or t is None:
        return {'type': 'null'}
    elif hasattr(t, '__origin__'):  # probably a generic typing-type
        if t.__origin__ is list:
            return {'type': 'array', 'items': schema_by_python_type(t.__args__[0])}
        elif t.__origin__ is dict:
            return {'type': 'object', 'additionalProperties': schema_by_python_type(t.__args__[1])}
        elif t.__origin__ is tuple and len(set(t.__args__)) == 1:
            return {'type': 'array', 'items': schema_by_python_type(t.__args__[0]), 'minItems': len(t.__args__), 'maxItems': len(t.__args__)}
        elif t.__origin__ is Union:
            return {'oneOf': [schema_by_python_type(t2) for t2 in t.__args__]}
        else:
            raise ValueError(f'Unknown type {t}')
    elif issubclass(t, list):
        return {'type': 'array'}
    elif issubclass(t, type) and len(set(t.__args__)) == 1:
        return {'type': 'array', 'items': schema_by_python_type(t.__args__[0]), 'minItems': len(t.__args__), 'maxItems': len(t.__args__)}
    elif hasattr(t, 'schema_for_referencing'):
        return t.schema_for_referencing()
    else:
        raise ValueError(f'Unknown type {t}')


class Synchronizable:
    def __init__(self):
        self._lock = threading.RLock()

    @staticmethod
    def synchronized(f):
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            with self._lock:
                return f(self, *args, **kwargs)

        return wrapper


def class_from_json(cls, data: Dict[str, Any]):
    if isinstance(data, str):
        data = json.loads(data)
    # noinspection PyArgumentList
    try:
        return cls(**data)
    except TypeError as e:
        if "__init__() got an unexpected keyword argument 'type'" in str(e):
            # probably this was from a to_json method
            if data['type'] != cls.__name__:
                t = data['type']
                logging.warning(f'Reconstructing a {cls.__name__} from a dict with type={t}')
            data = data.copy()
            del data['type']
            for k, v in data.items():
                if probably_serialized_from_json_convertible(v):
                    data[k] = JSONConvertible.SUBCLASSES_BY_NAME[v['type']].from_json(v)
                elif isinstance(v, list):
                    data[k] = [JSONConvertible.SUBCLASSES_BY_NAME[x['type']].from_json(x)
                               if probably_serialized_from_json_convertible(x)
                               else x
                               for x in v]
                elif isinstance(v, dict):
                    data[k] = {
                        k2: JSONConvertible.SUBCLASSES_BY_NAME[x['type']].from_json(x)
                        if probably_serialized_from_json_convertible(x)
                        else x
                        for k2, x in v.items()
                    }
            try:
                return cls(**data)
            except TypeError:
                return allow_additional_unused_keyword_arguments(cls)(**data)


def probably_serialized_from_json_convertible(data):
    return isinstance(data, dict) and 'type' in data and data['type'] in JSONConvertible.SUBCLASSES_BY_NAME


class EBE(JSONConvertible, Enum):
    def __int__(self):
        return self.value

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        return list(type(self)).index(self) < list(type(self)).index(other)

    def __ge__(self, other):
        return list(type(self)).index(self) >= list(type(self)).index(other)

    @classmethod
    def from_name(cls, variable_name) -> 'EBE':
        return cls.__dict__[variable_name]

    def to_json(self) -> Dict[str, Any]:
        return {
            'type': type(self).__name__,
            'name': self.name,
        }

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {'type': schema_by_python_type(str), 'name': schema_by_python_type(str)}
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> 'JSONConvertible':
        cls = EBE.SUBCLASSES_BY_NAME[data['type']]
        return cls.from_name(data['name'])


def floor_to_multiple_of(x, multiple_of):
    return math.floor(x / multiple_of) * multiple_of


def ceil_to_multiple_of(x, multiple_of):
    return math.ceil(x / multiple_of) * multiple_of


def call_tool(command, cwd=None):
    try:
        print(f'Calling `{" ".join(command)}`...')
        sub_env = os.environ.copy()
        output: bytes = check_output(command, stderr=PIPE, env=sub_env, cwd=cwd)
        output: str = output.decode('utf-8', errors='ignore')
        return output
    except CalledProcessError as e:
        stdout = e.stdout.decode('utf-8', errors='ignore')
        stderr = e.stderr.decode('utf-8', errors='ignore')
        if len(stdout) == 0:
            print('stdout was empty.')
        else:
            print('stdout was: ')
            print(stdout)
        if len(stderr) == 0:
            print('stderr was empty.')
        else:
            print('stderr was: ')
            print(stderr)
        raise
