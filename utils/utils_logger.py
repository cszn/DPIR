import os
import sys
import datetime
import logging


'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
https://github.com/xinntao/BasicSR
'''


def log(*args, **kwargs):
    """
    Log a message to the console.

    Args:
    """
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


'''
# ===============================
# logger
# logger_name = None = 'base' ???
# ===============================
'''


def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exists!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


'''
# ===============================
# print to file and std_out simultaneously
# ===============================
'''


class logger_print(object):
    def __init__(self, log_path="default.log"):
        """
        Initialize log_path

        Args:
            self: (todo): write your description
            log_path: (str): write your description
        """
        self.terminal = sys.stdout
        self.log = open(log_path, 'a')

    def write(self, message):
        """
        Write a message to the log.

        Args:
            self: (todo): write your description
            message: (str): write your description
        """
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        """
        Flush the cache entries.

        Args:
            self: (todo): write your description
        """
        pass
