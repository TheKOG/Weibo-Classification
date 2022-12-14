#!/usr/bin/python3
# -*- coding=utf-8 -*-

"""自动训练文本信息
@author: 张伟
@time: 2018/2/28 10:58
"""

import math
import re
from threading import Thread
import queue as queues

ALL_WORDS = dict()
All_LENS = 0


class Neologism(Thread):
    def __init__(self, queue, split_num=4):
        Thread.__init__(self)
        self.queue = queue
        self.split_num = split_num

    def run(self):
        while True:
            try:
                line = self.queue.get_nowait()
                self.read_string(line)
                self.queue.task_done()
            except queues.Empty:
                return

    def read_string(self, st, split_seq='[，。！？：]'):
        """
        讲字符按照split_seq格式来分割
        :param st: 字符串
        :param split_seq: 字符分割
        :return: None
        """
        ls = re.split(split_seq, st)
        self.read_ls(ls=ls)

    def read_ls(self, ls):
        """数据类型[str]
        :param ls: 表示链表
        """
        global All_LENS
        for word in ls:
            All_LENS += len(word)
            clean_data = clean(data=word)
            if clean_data:
                self.split(clean_data)

    def split(self, words):
        """
        拆分字符，最大匹配num个字符，并也字典的形式返回，
        [出现次数,出现频率,凝固程度,自由程度,关键字的左邻,关键字的右邻](作为信息熵的衡量)
        """
        global ALL_WORDS
        lens = len(words)
        for i in range(0, lens):
            for j in range(1, self.split_num + 1):
                if i + j < lens:
                    key = words[i:i + j]
                    word = ALL_WORDS.get(key)
                    if word:
                        word[0] += 1
                        word[4].append(words[i - 1])
                        word[5].append(words[i + j])
                    else:
                        ALL_WORDS[key] = [1, 0.0, 1, 0, [words[i - 1]], [words[i + j]]]


def statistics(key_list):
    for key in key_list:
        ALL_WORDS[key][1] = ALL_WORDS[key][0] / All_LENS


def handle(key_list):
    """
    处理数据
    计算左邻字集合和右邻字集合有多随机，左邻字信息熵和右邻字信息熵中的较小值
    计算凝固程度,自由程度
    """
    for key in key_list:
        key_list_words = ALL_WORDS[key]
        if len(key) == 1:
            continue
        end_all = front_all = 0.0
        left = key_list_words[1] / (ALL_WORDS[key[0]][1] * ALL_WORDS[key[1:]][1])
        right = key_list_words[1] / (ALL_WORDS[key[-1]][1] * ALL_WORDS[key[:-1]][1])

        for front in key_list_words[4]:
            if ALL_WORDS.get(front):
                front_all -= math.log(ALL_WORDS[front][1]) * ALL_WORDS[front][1]

        for end in key_list_words[5]:
            if ALL_WORDS.get(end):
                end_all -= math.log(ALL_WORDS[end][1]) * ALL_WORDS[end][1]

        key_list_words[2] = left if left < right else right
        key_list_words[3] = front_all if front_all < end_all else end_all


def filter_words(frequency, cond, free, flag):
    """
    过滤一些不重要的数据
    [出现次数,出现频率,凝固程度,自由程度]
    :param frequency: 过滤的频率
    :param cond:过滤凝聚度
    :param free:过滤自由度
    :param flag: 是否是并且还是或者,默认是或者，满足一个就过滤
    :return:过滤后的数据字典
    """
    key_words = dict()
    for key in ALL_WORDS.keys():
        if len(key) <= 1:
            continue
        one_word = ALL_WORDS[key]
        if flag:
            if one_word[1] > frequency and one_word[2] > cond and one_word[3] > free:
                key_words[key] = [one_word[0], one_word[1], one_word[2], one_word[3]]
        else:
            if one_word[1] > frequency or one_word[2] > cond or one_word[3] > free:
                key_words[key] = [one_word[0], one_word[1], one_word[2], one_word[3]]
    return key_words


def read_file(file, file_encoding='utf-8'):
    """
    读取文件内容，注意文件是UTF-8的格式且不是BOM格式
    :param file: 读取的文件
    :param file_encoding: 文本编码
    """
    queue = queues.Queue(maxsize=0)
    with open(file, encoding=file_encoding) as fp:
        for line in fp:
            queue.put(line)
    return queue


def clean(data):
    # 去除非中文字符
    words = [work for work in data if 19968 < ord(work) < 40959]
    if len(words) > 2:
        return ''.join(words)
    return None


def analysis(file, thread_num=10, split_num=4, frequency=0.0001, cond=10, free=0.1, flag=False):
    """
    :param file: 训练的文本
    :param thread_num: 线程数
    :param split_num: 匹配个数
    :param frequency: 频率
    :param cond: 凝聚度
    :param free: 自由度
    :param flag:是否是并且还是或者,默认是或者，满足一个就过滤
    :return: 分析完毕的字典
    """
    queue = read_file(file)
    neologisms = [Neologism(split_num=split_num, queue=queue) for _ in range(thread_num)]
    for neologism in neologisms:
        neologism.start()
    queue.join()
    print("开始统计频率.........")
    keys_list = list(ALL_WORDS.keys())
    size = len(keys_list) // split_num + 1
    threads = []
    for i in range(split_num):
        t = Thread(target=statistics, args=(keys_list[i * size:(i + 1) * size],))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    print("开始处理数据.........")
    for i in range(split_num):
        t = Thread(target=handle, args=(keys_list[i * size:(i + 1) * size],))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    print("开始过滤数据.........")
    neologism_word = filter_words(frequency, cond, free, flag)
    print("分析完毕！..........")
    return neologism_word


if __name__ == '__main__':
    neologism_words = analysis(file='小时代.txt', thread_num=10, frequency=0.00001, split_num=8, cond=100, flag=True)
    for k, v in neologism_words.items():
        print('key:{0} count:{1} frequency:{2} cond:{3} free:{4}'.format(k, v[0], v[1], v[2], v[3]))
