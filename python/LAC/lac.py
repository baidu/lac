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
本文件定义了LAC类，实现其调用分词，词性标注，训练模型的接口。
"""

import os
import shutil
import logging

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

from . import reader
from . import utils
from . import nets

from ._compat import *
from .custom import Customization
from .models import Model, SegModel, LacModel, RankModel

def _get_abs_path(path): return os.path.normpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__), path))

DEFAULT_LAC = _get_abs_path('lac_model')
DEFAULT_SEG = _get_abs_path('seg_model')
DEFAULT_RANK = _get_abs_path('rank_model')

PATH_DICT = {
                "lac": DEFAULT_LAC,
                "seg": DEFAULT_SEG,
                "rank":DEFAULT_RANK
             }


class LAC(object):
    """Docstring for LAC"""
    def __init__(self, model_path=None, mode='lac', use_cuda=False):
        super(LAC, self).__init__()
        utils.check_cuda(use_cuda)

        model_path = model_path if model_path else PATH_DICT[mode]

        if mode == 'seg':
            model = SegModel(model_path, mode, use_cuda)
        elif mode == 'lac':
            model = LacModel(model_path, mode, use_cuda)
        elif mode == 'rank':
            model = RankModel(model_path, mode, use_cuda)

        self.model = model

    def run(self, texts):
        """执行模型预测过程
        Args:
            texts: 模型输入的文本，一个Unicode编码的字符串或者
                   由Unicode编码字符串组成的List
        Returns:
            if mode=='seg',  返回分词结果
            if mode=='lac',  返回分词,词性结果
            if mode=='rank', 返回分词,词性,词语重要性结果
        """
        return self.model.run(texts)
    
    def train(self, model_save_dir, train_data, test_data=None, iter_num=10, thread_num=10):
        """执行模型增量训练
        Args:
            model_save_dir: 训练结束后模型保存的路径
            train_data: 训练数据路径
            test_data: 测试数据路径，若为None则不进行测试
            iter_num: 训练数据的迭代次数
            thread_num: 执行训练的线程数
        """
        self.model.train(model_save_dir, train_data, test_data, iter_num, thread_num)
    
    def load_customization(self, customization_file, sep=None):
        """装载用户词典

        Args:
            texts: 用户词典路径
            sep: 表示词典中，短语片段的分隔符，默认为空格' '或制表符'\t'
        """
        self.model.custom = Customization()
        self.model.custom.load_customization(customization_file, sep)
    
    def add_word(self, word, sep=None):
        """添加单词，格式与用户词典一致
        Args:
            texts: 用户定义词典，如："春天"、"花 开"、"春天/SEASON"、"花/n 开/v"、
            sep: 表示词典中，短语片段的分隔符，默认为空格' '或制表符'\t'
        """
        if self.model.custom is None:
            self.model.custom = Customization()
        self.model.custom.add_word(word, sep)

if __name__ == "__main__":
    print('######### mode = lac ##############')
    lac = LAC('lac_model')

    test_data = [u'百度是一家高科技公司', u'LAC是一个优秀的分词工具', '']

    print('######### run:list ##############')
    result = lac.run(test_data)
    for res in result:
        print(' '.join(res))

    print('######### run:str ##############')
    result = lac.run(test_data[0])
    print(' '.join(result))

    print('######### run:tag ##############')
    result = lac.run(test_data)
    for i, (sent, tags) in enumerate(result):
        result_list = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)]
        print(''.join(result_list))

    print('######### mode = rank ##############')
    lac = LAC('lac_model', mode='rank')

    print('######### run:list ##############')
    result = lac.run(test_data)
    for res in result:
        print(' '.join(res))

    print('######### run:str ##############')
    result = lac.run(test_data[0])
    print(' '.join(result))

    print('######### run:tag ##############')
    result = lac.run(test_data)
    for i, (sent, tags, word_rank) in enumerate(result):
        result_list = ['(%s, %s, %s)' % (ch, tag, rank) for ch, tag, rank in zip(sent, tags, word_rank)]
        print(''.join(result_list))


    # 重训模型
    lac.train(model_save_dir='models_test',
              train_data='./data/train.tsv', test_data='./data/test.tsv')

    print('######### run:list ##############')
    result = lac.run(test_data)
    for res in result:
        print(' '.join(res))
    print('######### run:str ##############')
    result = lac.run(test_data[0])
    print(' '.join(result))
    print('######### run:tag ##############')
    result = lac.run(test_data, mode='lac')
    for i, (sent, tags) in enumerate(result):
        result_list = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)]
        print(''.join(result_list))
