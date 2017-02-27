# -*- coding: utf-8 -*-

from cStringIO import StringIO
import sys

import logging as _l

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


class StopRecursion(Exception):
    pass


# %(levelname)s:\t%(message)s
def setup_logger(logger_name, log_file, format='%(message)s', level=_l.INFO):
    logger = _l.getLogger(logger_name)
    formatter = _l.Formatter(format)
    fileHandler = _l.FileHandler(log_file)
    fileHandler.setFormatter(formatter)
    streamHandler = _l.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)
    # l.addHandler(streamHandler)
    return logger


def reset_logger(logger_name):
    logger = _l.getLogger(logger_name)
    logger.setLevel(_l.INFO)
    logger.handlers = []
    return logger
