from datetime import datetime
import numpy as np
from empath import Empath
import re
import os
import errno
import ast
import logging
import sys

lexicon = Empath()


def date_parser(data):
    if data == 'nan':
        return np.nan
    return datetime.utcfromtimestamp(int(float(data))).date()


def get_tweet_rep(tweet):
    tokens = re.split('[^A-Za-z0-9]', tweet)
    rep = lexicon.analyze(tokens, normalize=True)
    return rep


def get_representation(tweets):
    keys = []
    all_rep = []
    for tweet in tweets:
        rep = get_tweet_rep(tweet)
        if rep is not None:
            all_rep.append(list(rep.values()))
            if len(keys) == 0:
                keys = list(rep.keys())
    if len(all_rep) == 0:
        return None, None
    all_rep = np.array(all_rep)
    return keys, np.mean(all_rep, axis=0)


def check_dir(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def make_logger(name=__name__, logname=None, level=logging.INFO,
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                filemode="a"):
    logger = logging.getLogger(name)
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    if logname is not None:
        if "logs" not in logname:
            logname = "logs/" + logname
        if ".log" not in logname:
            logname = logname + ".log"
        check_dir(logname)
        formatter = logging.Formatter(fmt, datefmt=datefmt)
        file_handler = logging.FileHandler(logname, mode=filemode)
        file_handler.setFormatter(formatter)
        # stream_handler = logging.StreamHandler()
        for old_handler in logger.handlers:
            logger.removeHandler(old_handler)
        logger.addHandler(file_handler)
        # logger.addHandler(stream_handler)
    return logging.getLogger(name)


def jaccard_sim():
    pass


def get_home():
    from pathlib import Path
    home = str(Path.home())
    return home
