# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2020  Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
"""
本文件定义了分词及其实现
"""
import io
import re
import sys
import time
import logging
from math import log

from .prefix_tree import TriedTree


re_eng = re.compile('[a-zA-Z0-9]', re.U)

def load_seg_dict(dict_path):
    """
    Load profile dict from file and calculate word frequency
    """
    result_dict = TriedTree()
    result_total = 0

    with io.open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, count = line.strip().split(' ')
            result_dict.add_word(word)
            result_total += int(count)

    return result_dict.tree, log(result_total)

class Segment(object):
    def __init__(self, dict_path):
        super(Segment, self).__init__()
        """
        Args:
            self.dict_path: 字典地址
            self.f_dict   : 前缀字典
            self.logtotal : 词频总数取log
            self.length   : 句子长度
            self.dag      : DAG
        """

        self.dict_path = dict_path
        self.dag = {} 
        self.length = 0
     
        start_time = time.time()

        self.f_dict, self.logtotal = load_seg_dict(self.dict_path)

        init_dict_time = time.time()
        logging.info("Init Prefix Trie used {}s".format(init_dict_time - start_time))
    
    def fast_get_DAG(self, text):  
        """生成DAG"""
        self.length = len(text)
        self.dag = {_:[_] for _ in range(self.length)}

        for head_word in range(self.length):
            end_word = head_word + 1
            word = text[head_word:end_word]

            while end_word < self.length and word in self.f_dict:
                if self.f_dict[word]:
                    self.dag[head_word].append(end_word-1)
                end_word += 1
                word = text[head_word:end_word]
    
    def fast_cut(self, text):
        """
        分词
        Args:
            route   : 最大路径字典
            buf     : 临时分词结果
        Return:
            segment : 分词结果
        """
        self.fast_get_DAG(text)
        route = dict()

        route[self.length] = (0, 0)
        
        for idx in range(self.length-1, -1, -1):
            # 取log防止向下溢出,取过log后除法变为减法
            route[idx] = max((log(self.f_dict.get(text[idx: _+1]) or 1) -
                            self.logtotal + route[_+1][0], _) for _ in self.dag[idx])

        incept_idx = 0
        buf = ""
        segment = []
        while incept_idx < self.length:
            end_idx = route[incept_idx][1] + 1
            l_word = text[incept_idx:end_idx]
            
            if re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                incept_idx = end_idx
            else:
                if buf:
                    segment.append(buf)
                    buf = ""
                segment.append(l_word)
                incept_idx = end_idx
        if buf:
            segment.append(buf)
            buf = ""
        return segment