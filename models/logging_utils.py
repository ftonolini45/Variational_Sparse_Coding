

import logging

_loggers = {}



def get_logger(name):
    global _loggers
    try:
        log = _loggers[name]
    except KeyError:
        # create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # create console handler and file and set level to debug
        channel_stream = logging.StreamHandler()
        channel_stream.setLevel(logging.DEBUG)

        # create formatter
        formatter_stream = logging.Formatter('{}: %(asctime)s - %(levelname)s - %(message)s'.format(name))

        # add formatter to ch
        channel_stream.setFormatter(formatter_stream)

        # add ch to logger
        logger.addHandler(channel_stream)
        _loggers[name] = logger
        log = logger
    return log


