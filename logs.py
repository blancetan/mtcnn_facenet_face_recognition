#!/usr/bin/env python3
#_*_ coding: utf-8 _*_

"""
 @DateTime: 18/4/2020 11:33
 @Author:   balanceTan 
 @File:     logs.py
 @Software: PyCharm
 
"""
import logging
import sys

def log():

    # create logger object
    logger = logging.getLogger()

    # create fileHandler object
    fh = logging.FileHandler('log/test.log')

    # create  consHandle object
    ch = logging.StreamHandler()

    # logging formatter
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(filename)s [line:%(lineno)d] %(message)s')

    # set fileHandel logging format
    fh.setFormatter(formatter)

    # set consHandel logging format
    ch.setFormatter(formatter)

    # add fh and ch to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # set logger lever (default warning)
    # 5 levers: DEBUG, INFO,WARNING,ERROR, CRITICAL
    logger.setLevel(logging.INFO)
    return logger
    # logger.removeHandler(fh)
    # logger.removeHandler(ch)

if __name__ == '__main__':
    logger = log()