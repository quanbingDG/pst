# -*- coding: utf-8 -*-
# @Time : 2020/10/12 4:03 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : utils.py


import os


def check_is_dir(_path):
    return os.path.isdir(_path)


def rescaling(x):
    minimum = x.min()
    maximum = x.max()
    return x.apply(lambda y: (y - minimum) / (maximum - minimum))

