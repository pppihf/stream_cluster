#!/usr/bin/python
# -*- encoding:utf-8 -*-
import os
import _pickle as cPickle
import gzip
from gensim import corpora, models
import time


class LDAGetTopic:
    def __init__(self):  # LDA参数设置
        self.num_topics = 100

    def load_data(self, file):
        """
        读取文本并转化成LDA的输入形式
        :param file:  要读取的文件名
        :return: 词表, id化了的文本（LDA的输入）
        """
        content = []
        with open(file, 'r', encoding='utf-8') as F:
            for line in F.readlines():
                line = [word.strip() for word in line.split(' ')]
                content.append(line)
        dictionary = corpora.Dictionary(content)
        texts = [dictionary.doc2bow(text) for text in content]
        return dictionary, texts

    def train_lda(self, texts, dictionary):
        """
        训练HLDA模型
        :param texts:  用来训练的文本（已经id化）
        :param dictionary:  词表
        :return:训练得到的lda模型
        """
        lda = models.ldamulticore.LdaMulticore(corpus=texts, id2word=dictionary, num_topics=self.num_topics)
        return lda

    def get_topic(self, texts, lda):
        """
        提取主题
        :param texts:  待分析的文本（已经id化）
        :param lda:  训练好的LDA模型
        :return: 从文本中获得的topic相关信息
        """
        # 存储结果
        topic_id = []

        topic2id = set()
        instance2topic = []
        topic_list = lda.print_topics(num_topics=-1, num_words=5)
        for n, topic in enumerate(topic_list):
            topic2id_str = ''
            topic_word_list = topic[1].split('"')
            for i in range(len(topic_word_list)):
                if i % 2:
                    topic2id_str += topic_word_list[i] + ', '
            topic2id_str += '\tt' + str(topic[0])
            topic_id.append(n)
            topic2id.add(topic2id_str)
        for d in range(len(texts)):
            instance_id = 'i' + str(d)
            topic = lda.get_document_topics(texts[d])
            try:
                max_prob = 0
                topicno = topic[0][0]
                for t in topic:
                    if t[1] > max_prob:
                        max_prob = t[1]
                        topicno = t[0]
                instance2topic_str = instance_id + '\tt' + str(topicno)
                instance2topic.append(instance2topic_str)
            except IndexError:
                continue
        print(instance2topic)
        # 整理topic的序号
        tid_dict = {}
        tid_new = 0
        tmp_list = []
        for line in topic2id:
            tmp = line.split('\t')
            tid_old = tmp[1]
            tmp_list.append(tid_old)
        tmp_list.sort()
        for tmp in tmp_list:
            tid_dict[tmp] = "t" + str(tid_new)
            tid_new += 1
        return tid_dict, topic2id, instance2topic


def save_topic_result(data_dir, tid_dict, topic2id, instance2topic):
    with open(os.path.join(data_dir, 'topic2id.txt'), 'w', encoding='utf-8') as f1:
        for line in topic2id:
            text = line.split('\t')
            f1.write(text[0] + '\t' + tid_dict[text[1]] + '\n')
    with open(os.path.join(data_dir, 'instance2topic.txt'), 'w', encoding='utf-8') as f2:
        for line in instance2topic:
            text = line.split('\t')
            f2.write(text[0] + '\t' + tid_dict[text[1]] + '\n')


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object


if __name__ == "__main__":
    start = time.perf_counter()
    oridata_dir = "../data/oridata"
    data_dir = "../data/HINdata"
    lda_dir = os.path.join(data_dir, 'topic_lda.model')

    L = LDAGetTopic()
    dictionary, texts = L.load_data(os.path.join(oridata_dir, "wordcut_text.txt"))
    lda = L.train_lda(texts, dictionary)
    tid_dict, topic2id, instance2topic = L.get_topic(texts, lda)
    save_topic_result(data_dir, tid_dict, topic2id,instance2topic)  # 保存结果
    lda.save(lda_dir)  # 保存模型
    # 读取模型并更新
    # _, other_corpus = L.load_data('xxx')
    # lda = models.ldamulticore.LdaModel.load(lda_dir)
    # lda.update(other_corpus)

    print('s3_getTopicLDA done', flush=True)
    end = time.perf_counter()
    duration = end - start
    print('用时:', duration, flush=True)
