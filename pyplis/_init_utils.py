# -*- coding: utf-8 -*-
import logging
from os.path import abspath, dirname
from pkg_resources import get_distribution
from matplotlib import rcParams

def _init_supplemental():
    
    rcParams["mathtext.default"] = u"regular"

    return (get_distribution('pyplis').version, abspath(dirname(__file__)))

def _init_logger():
    ### LOGGING
    # Note: configuration will be propagated to all child modules of
    # pyplis, for details see
    # http://eric.themoritzfamily.com/learning-python-logging.html
    logger = logging.getLogger('pyplis')

    fmt = "%(filename)s(l%(lineno)s,%(funcName)s()): %(message)s"
    #fmt = "%(funcName)s():%(lineno)i: %(message)s"
    default_formatter = logging.Formatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(default_formatter)

    logger.addHandler(console_handler)

    logger.setLevel(logging.WARNING)

    print_log = logging.getLogger('pyplis_print')

    print_handler = logging.StreamHandler()
    print_handler.setFormatter(logging.Formatter("%(message)s"))

    print_log.addHandler(print_handler)

    print_log.setLevel(logging.INFO)
    return (logger, print_log)

def _get_loglevels():
    
    return dict(critical=logging.CRITICAL,
                exception=logging.ERROR,
                error=logging.ERROR,
                warn=logging.WARNING,
                warning=logging.WARNING,
                info=logging.INFO,
                debug=logging.DEBUG)

def get_loglevel(logger):
    return logger.getEffectiveLevel()

def change_loglevel(logger, level, update_fmt=False, fmt_debug=True):
    LOG_LEVELS = _get_loglevels()
    if level in LOG_LEVELS:
        logger.setLevel(LOG_LEVELS[level])
    else:
        try:
            logger.setLevel(level)
        except Exception as e:
            raise ValueError('Could not update loglevel, invalid input. Error: {}'.format(repr(e)))
    if update_fmt:
        import logging
        if fmt_debug:
            fmt = logging.Formatter("%(filename)s(l%(lineno)s,%(funcName)s()): %(message)s")
        else:
            fmt = logging.Formatter("%(message)s")
        for handler in logger.handlers:
            handler.setFormatter(fmt)

def check_requirements():
    try:
        import pydoas
        PYDOASAVAILABLE = True
    except BaseException:
        PYDOASAVAILABLE = False
    try:
        import geonum
        GEONUMAVAILABLE = True
    except BaseException:
        GEONUMAVAILABLE = False
    try:
        from PIL.Image import open as open_pil
        PILAVAILABLE = True
    except BaseException:
        PILAVAILABLE = False
    return (PYDOASAVAILABLE, GEONUMAVAILABLE, PILAVAILABLE)