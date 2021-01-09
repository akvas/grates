# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Convenience functions for Python datetime objects
"""

import datetime as dt
import calendar as cal


def __mjd_fepoch():

    return dt.datetime(1858, 11, 17)


def __gps_fepoch():

    return dt.datetime(1980, 1, 6)


def mjd(dtime):
    """
    Convert datetime.datetime object to MJD.

    Parameters
    ----------
    dtime : datetime.datetime
        datetime.datetime object

    Returns
    -------
    mjd : float
        datetime.datetime object expressed in modified julian date
    """

    delta = (dtime - __mjd_fepoch())
    return delta.days + delta.seconds/86400.0


def datetime(mjd):
    """
    Convert MJD to datetime.datetime object

    Parameters
    ----------
    mjd : float
        datetime.datetime object expressed in modified julian date

    Returns
    -------
    dtime : datetime.datetime
        datetime.datetime object
    """
    return __mjd_fepoch() + dt.timedelta(days=mjd)


def date_iterator(start, stop, step):
    """
    Generator for a sequence of datetime objects.

    To be consistent with Python ranges, the last epoch generated will be strictly less than `stop`.

    Parameters
    ----------
    start : datetime object
        first epoch
    stop : datetime object
        upper bound of datetime sequence, last epoch will be strictly less than stop
    step : timedelta object
        step size of sequence (negative steps are allowed)

    Returns
    -------
    g : Generator object
        Generator for datetime objects

    """
    if step.total_seconds() == 0.0:
        raise ValueError('Step size must not be zero')

    op = dt.datetime.__gt__ if step.total_seconds() < 0 else dt.datetime.__lt__

    current = start
    while op(current, stop):
        yield current
        current += step


def month_iterator(start, stop, use_middle=False):
    """
    Generator for a monthly sequence of datetime objects.

    To be consistent with Python ranges, the last epoch generated will be strictly less than `stop`.

    Parameters
    ----------
    start : datetime object
        epoch from which the first month will be generated
    stop : datetime object
        epoch from which the last month will be generated (last month will be strictly less than stop)
    use_middle : bool
        If True, the midpoint of each month will be returned, otherwise the first of each month is used (default: False)

    Returns
    -------
    g : Generator object
        Generator for monthly datetime objects

    """
    current = dt.datetime(start.year, start.month,
                          round(cal.monthrange(start.year, start.month)[1]*0.5) if use_middle else 1)
    while current < stop:
        yield current
        roll_over = (current.month == 12)

        next_year = current.year+1 if roll_over else current.year
        next_month = 1 if roll_over else current.month+1
        next_day = round(cal.monthrange(next_year, next_month)[1]*0.5) if use_middle else current.day

        current = dt.datetime(next_year, next_month, next_day)


def day_iterator(start, stop, use_middle=False):
    """
    Generator for a daily sequence of datetime objects.

    To be consistent with Python ranges, the last epoch generated will be strictly less than `stop`.

    Parameters
    ----------
    start : datetime object
        epoch from which the first day will be generated
    stop : datetime object
        epoch from which the last day will be generated (last month will be strictly less than stop)
    use_middle : bool
        If True, the midpoint of each day (12:00) will be returned, otherwise 00:00 used (default: False)

    Returns
    -------
    g : Generator object
        Generator for monthly datetime objects

    """
    current = dt.datetime(start.year, start.month, start.day, 12 if use_middle else 0)
    while current < stop:
        yield current
        current += dt.timedelta(days=1)


def decyear2mjd(dy):
    """
    Convert decimal year to MJD.

    Parameters
    ----------
    dy : float
        epoch as decimal year

    Returns
    -------
    mjd : float
        epoch as MJD
    """
    y0 = mjd(dt.datetime(int(dy), 1, 1))
    y1 = mjd(dt.datetime(int(dy)+1, 1, 1))

    return (dy-int(dy))*(y1-y0)+y0


def mjd2decyear(t_mjd):
    """
    Convert MJD to decimal year.

    Parameters
    ----------
    mjd : float
        epoch as MJD

    Returns
    -------
    dy : float
        epoch as decimal year
    """
    t = datetime(t_mjd)
    length = 366.0 if cal.isleap(t.year) else 365.0
    days = (t - dt.datetime(t.year, 1, 1)).days
    return float(t.year) + days/length


def gpsweekday(dt):
    """
    Compute GPS week and day in week from datetime object

    Parameters
    ----------
    dt : datetime.datetime
        epoch as datetime object

    Returns
    -------
    week : int
        GPS week
    days : int
        day in GPS week
    """
    delta = dt - __gps_fepoch()

    week = delta.days//7
    days = delta.days - week*7

    return week, days


def gpsweekseconds(dt):
    """
    Compute GPS week and seconds in week from datetime object

    Parameters
    ----------
    dt : datetime.datetime
        epoch as datetime object

    Returns
    -------
    week : int
        GPS week
    seconds : float
        seconds in GPS week
    """
    delta = dt - __gps_fepoch()

    week = delta.days//7
    days = delta.total_seconds() - week*7*86400

    return week, days


def gpsweekday2datetime(week, day):
    """
    Convert GPS week and day in week to datetime object.

    Parameters
    ----------
    week : int
        GPS week
    days : int
        day in GPS week

    Returns
    -------
    dt : datetime.datetime
        epoch as datetime objec
    """
    delta = dt.timedelta(days=week*7+day)

    return __gps_fepoch() + delta
