#!/usr/bin/python
# -*- encoding:utf-8 -*-
# author ZhangLiang
import re
from ltp import LTP
import jieba
from jieba import analyse
# 设置用户自定停用词集合
analyse.set_stop_words("../static_data/baidu_stopwords.txt")


class ChinesePreprocessor:
    def __init__(self, stopwords_path):
        """
        :param stopwords_path: 停用词文件路径
        """
        self.stopwords = [line.strip() for line in open(stopwords_path, encoding='utf-8').readlines()]
        self.SPECIAL_SYMBOL_RE = re.compile(r'[^\w\s\u4e00-\u9fa5]+')  # 删除特殊符号
        self.extract = analyse.extract_tags

    def word_cut(self, text: str):
        """
        将传入的list进行清洗和分词
        :param text: 待分词字符串
        :return: 分词后的文档字符串
        """
        # 去除回车符和换行符
        text = text.replace('\n', '').replace('\r', '')
        # 去除特殊字符
        text = re.sub(pattern=self.SPECIAL_SYMBOL_RE, repl=' ', string=text)
        # 分词
        segments = jieba.cut(text)  # 默认精确模式切割词语
        # 去停用词
        words = []
        for w in segments:
            if len(w) < 2:  # 去除单个字符
                continue
            if w.isdigit():  # 去除完全为数字的字符串
                continue
            if w not in self.stopwords:  # 去除停用词
                words.append(w)
        return words

    def get_keywords(self, content: str, topK=5):
        """
        使用TFIDF算法获取关键词，最多获取5个TFIDF值超过0.5的关键词
        :param content: 原始文本
        :param topK: 关键词数量
        :return: 关键词列表
        """
        keywords = []
        try:
            tags = self.extract(content, topK, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
            i = 0
            for v, n in tags:
                if i > 4:
                    break
                if n < 0.5:
                    break
                keywords.append(v)
                i = i+1
        except Exception as e:
            pass
        return keywords


class NamedEntity:
    def __init__(self, user_dict):
        self.ltp = LTP() # 默认加载 Small 模型
        # user_dict.txt 是词典文件， max_window是最大前向分词窗口
        self.ltp.init_dict(path=user_dict, max_window=4)

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
        return entity














