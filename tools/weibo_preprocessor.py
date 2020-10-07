#!/usr/bin/python
# -*- encoding:utf-8 -*-
# author ZhangLiang
import re
from ltp import LTP


class ChinesePreprocessor:
    def __init__(self, stopwords_path, user_dict):
        """
        :param stopwords_path: 停用词文件路径
        :param user_dict: 用户自定义词典路径
        """
        self.stopwords = [line.strip() for line in open(stopwords_path, encoding='utf-8').readlines()]
        self.SPECIAL_SYMBOL_RE = re.compile(r'[^\w\s\u4e00-\u9fa5]+')  # 删除特殊符号
        self.ltp = LTP()  # 默认加载small模型
        self.ltp.init_dict(path=user_dict, max_window=4)

    def cleaner(self, text: list):
        """
        将传入的list进行清洗和分词
        :param text: 待分词字符串,传入的应该是个list
        :return: 分词后的文档字符串
        """
        # 去除回车符和换行符
        text = text.replace('\n', '').replace('\r', '')
        # 去除特殊字符
        text = re.sub(pattern=self.SPECIAL_SYMBOL_RE, repl=' ', string=text)
        # 分词
        segments, _ = self.ltp.seg(text)
        # 去停用词
        words = []
        for w in segments[0]:
            if len(w) < 2:  # 去除单个字符
                continue
            if w.isdigit():  # 去除完全为数字的字符串
                continue
            if w not in self.stopwords:  # 去除停用词
                words.append(w)
        return words

    def entity_recognition(self, text: list):
        """
        命名实体识别
        :param text: 原始文本
        :return: 从原始文本中抽取的命名实体
        """
        seg, hidden = self.ltp.seg(text)   # 分词
        ner = self.ltp.ner(hidden)
        entity = []
        for tag, start, end in ner[0]:
            entity.append(seg[0][start:end+1][0])
        return ner










