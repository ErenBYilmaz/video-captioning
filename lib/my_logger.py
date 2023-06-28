import os
import logging
import sys

import __main__
from typing import List, Dict

FORMAT = "%(asctime)-14s %(levelname)-8s %(module)s:%(lineno)s %(message)s"


class ListLogger(logging.Handler):  # Inherit from logging.Handler
    def __init__(self):
        logging.Handler.__init__(self)
        self.running = False
        self.log_lists: Dict[str, List[str]] = {}

    def start(self):
        self.running = True
        self.log_lists.clear()

    def stop(self):
        self.running = False
        self.log_lists.clear()

    def emit(self, record):
        if self.running:
            level_name = record.levelname
            if hasattr(record, 'end_user') and record.end_user:
                level_name = 'END_USER_' + level_name
            self.log_lists.setdefault(level_name, []).append(record.msg)


def end_user_warning(msg):
    logging.warning(msg, extra={'end_user': True})


def end_user_info(msg):
    logging.info(msg, extra={'end_user': True})


list_handler = ListLogger()

logging = logging
try:
    logfile = 'logs/' + os.path.normpath(__main__.__file__).replace(os.path.abspath('.'), '') + '.log'
except AttributeError:
    print('WARNING: unable to set log file path.')
else:
    try:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        logging.basicConfig(filename=logfile, filemode="a+", format=FORMAT)
    except OSError:
        print('WARNING: unable to set log file path.')
stdout_logger = logging.StreamHandler(sys.stdout)
stdout_logger.setFormatter(logging.Formatter(FORMAT))
logging.getLogger().addHandler(stdout_logger)
logging.getLogger().addHandler(list_handler)
logging.getLogger().setLevel(logging.INFO)
