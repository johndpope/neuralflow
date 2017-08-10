import datetime
import os

import logging


def start_logger(log_dir, log_file, log_to_console=True):
    os.makedirs(log_dir, exist_ok=True)

    complete_filename = "{}/{}".format(log_dir, log_file)
    formatter = logging.Formatter('%(levelname)s:%(message)s')

    logger = logging.getLogger('logger_{}'.format(complete_filename))
    logger.setLevel(logging.INFO)

    for hdlr in logger.handlers:  # remove all old handlers
        logger.removeHandler(hdlr)

    file_handler = logging.FileHandler(filename=complete_filename, mode='a')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)  # set file handler

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)  # set console handler

    now = datetime.datetime.now()

    logger.info('starting logging activity in date {}'.format(now.strftime("%d-%m-%Y %H:%M")))
    return logger
