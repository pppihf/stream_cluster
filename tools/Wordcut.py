#!/usr/bin/python
# -*- encoding:utf-8 -*-
# author ZhangLiang
import re
from ltp import LTP


class ChineseCut:
    def __init__(self, stop_path):
        # 加载停用词
        self.stopwords = [line.strip() for line in open(stop_path, encoding='utf-8').readlines()]
        self.SPECIAL_SYMBOL_RE = re.compile(r'[^\w\s\u4e00-\u9fa5]+')  # 删除特殊符号
        self.ltp = LTP()  # 默认加载small模型

    def
