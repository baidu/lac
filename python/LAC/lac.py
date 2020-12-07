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
from .custom import Customization
from ._compat import *


def _get_abs_path(path): return os.path.normpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__), path))


DEFAULT_LAC = _get_abs_path('lac_model')
DEFAULT_SEG = _get_abs_path('seg_model')
DEFAULT_RANK = _get_abs_path('rank_model')


class LAC(object):
    """docstring for LAC"""

    def __init__(self, model_path=None, mode='lac', use_cuda=False):
        super(LAC, self).__init__()
        utils.check_cuda(use_cuda)
        
        if mode == 'seg':
            model_path = DEFAULT_SEG if model_path is None else model_path
        elif mode == 'lac' :
            model_path = DEFAULT_LAC if model_path is None else model_path
        elif mode == 'rank':
            rank_path = DEFAULT_RANK if model_path is None else model_path

            parent_path = os.path.split(rank_path)[0]
            model_path = DEFAULT_LAC if model_path is None else os.path.join(parent_path, 'lac_model')

            # init rank predictor
            self.rank_path = rank_path
            self.rank_args = utils.DefaultArgs(self.rank_path)
            rank_config = AnalysisConfig(self.rank_args.init_checkpoint)
            rank_config.disable_glog_info()
            self.rank_predictor = create_paddle_predictor(rank_config)

        self.mode = mode

        self.args = utils.DefaultArgs(model_path)
        self.args.use_cuda = use_cuda
        
        self.model_path = model_path

        config = AnalysisConfig(self.args.init_checkpoint)
        config.disable_glog_info()

        if use_cuda:
            self.place = fluid.CUDAPlace(
                int(os.getenv('FLAGS_selected_gpus', '0')))
            config.enable_use_gpu(memory_pool_init_size_mb=500,
                                  device_id=int(
                                      os.getenv('FLAGS_selected_gpus', '0')),
                                  )
        else:
            self.place = fluid.CPUPlace()

        # init executor
        self.exe = fluid.Executor(self.place)

        self.dataset = reader.Dataset(self.args)

        self.predictor = create_paddle_predictor(config)

        self.custom = None
        self.batch = False

    def run(self, texts):
        """执行模型预测过程
        Args:
            texts: 模型输入的文本，一个Unicode编码的字符串或者
                   由Unicode编码字符串组成的List
        Returns:
            if mode=='seg',  返回分词结果
            if mode=='lac',  返回分词,词性结果
            if mode=='rank', 返回分词,词性,重要性结果
        """
        if isinstance(texts, list) or isinstance(texts, tuple):
            self.batch = True
        else:
            if len(texts.strip()) == 0:
                if self.mode == 'seg':
                    return []
                elif self.mode == 'lac':
                    return ([], []) 
                elif self.mode == 'rank':
                    return ([], [], []) 
            texts = [texts]
            self.batch = False

        tensor_words, words_length = self.texts2tensor(texts)
        crf_decode = self.predictor.run([tensor_words])
        crf_result = self.parse_result(texts, crf_decode[0], self.dataset, words_length)

        if self.mode == 'seg':
            result = [word for word, tag, tag_for_rank in crf_result] if self.batch else crf_result[0][0]
            return result

        else:
            result = [[word, tag] for word, tag, tag_for_rank in crf_result] if self.batch else crf_result[0][:-1]

            if self.mode == 'lac':
                return result 

            elif self.mode == 'rank':
                tags_for_rank = [tag_for_rank for word, tag, tag_for_rank in crf_result] if self.batch else crf_result[0][-1]
                rank_decode = self.rank_predictor.run([tensor_words, crf_decode[0]])
                weight = self.parse_rank(tags_for_rank, rank_decode[0]) 
                result = [result[_] + [weight[_]] for _ in range(len(result))]

                return result if self.batch else result[0]
    
    def parse_rank(self, tags_for_rank, result):
        """将rank模型输出的Tensor转为明文"""
        offset_list = result.lod[0]
        rank_weight = result.data.int64_data()
        batch_size = len(offset_list) - 1

        batch_out = []
        for sent_index in range(batch_size):
            begin, end = offset_list[sent_index], offset_list[sent_index + 1]

            tags = tags_for_rank[sent_index]
            weight = rank_weight[begin:end]
            weight_out = []
            for ind, tag in enumerate(tags):
                if tag.endswith("B") or tag.endswith("S"):
                    weight_out.append(weight[ind])
                    continue
                weight_out[-1] = max(weight_out[-1], weight[ind])

            batch_out.append(weight_out)
        return batch_out

    def parse_result(self, lines, crf_decode, dataset, words_length):
        """将LAC模型输出的Tensor转为明文"""
        offset_list = crf_decode.lod[0]
        crf_decode = crf_decode.data.int64_data()
        batch_size = len(offset_list) - 1

        batch_out = []
        for sent_index in range(batch_size):
            begin, end = offset_list[sent_index], offset_list[sent_index + 1]

            sent = lines[sent_index]
            tags = [dataset.id2label_dict[str(id)]
                    for id in crf_decode[begin:end]]
            tags_for_rank = tags[:]

            # 重新填充被省略的单词的char部分
            if len(words_length) != 0:
                word_length = words_length[sent_index]

                for current in range(len(word_length)-1, -1, -1):  
                    if word_length[current] > 1:
                        for offset in range(1, word_length[current]):
                            tags.insert(current + offset, tags[current][:-2] + '-I')

            if self.custom:
                self.custom.parse_customization(sent, tags)

            sent_out, tags_out = [], []
            for ind, tag in enumerate(tags):
                # for the first char
                if len(sent_out) == 0 or tag.endswith("B") or tag.endswith("S"):
                    sent_out.append(sent[ind])
                    tags_out.append(tag[:-2])
                    continue
                sent_out[-1] += sent[ind]
                # 取最后一个tag作为标签	
                tags_out[-1] = tag[:-2]

            batch_out.append([sent_out, tags_out, tags_for_rank])
        return batch_out

    def train(self, model_save_dir, train_data, test_data=None, iter_num=10, thread_num=10):
        """执行模型增量训练

        Args:
            model_save_dir: 训练结束后模型保存的路径
            train_data: 训练数据路径
            test_data: 测试数据路径，若为None则不进行测试
            iter_num: 训练数据的迭代次数
            thread_num: 执行训练的线程数
        """
        self.args.model = self.mode
        self.args.train_data = train_data
        self.args.test_data = test_data
        self.args.epoch = iter_num
        self.args.cpu_num = thread_num
        logging.info("Start Training!")

        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            test_program, fetch_list = nets.do_train(self.args)

            fluid.io.save_inference_model(os.path.join(model_save_dir, 'model'),
                                          ['words'],
                                          fetch_list,
                                          self.exe,
                                          main_program=test_program,
                                          )
        # 拷贝配置文件
        if os.path.exists(os.path.join(model_save_dir, 'conf')):
            shutil.rmtree(os.path.join(model_save_dir, 'conf'))
        shutil.copytree(os.path.join(self.model_path, 'conf'),
                        os.path.join(model_save_dir, 'conf'))

        self.load_model(model_save_dir)
        logging.info("Finish Training!")


    def load_model(self, model_dir):
        """装载预训练的模型"""
        use_cuda = self.args.use_cuda
        self.args = utils.DefaultArgs(model_dir)
        self.args.use_cuda = use_cuda
        self.dataset = reader.Dataset(self.args)
        self.model = self.args.model

        self.model_path = model_dir
        config = AnalysisConfig(os.path.join(model_dir, 'model'))
        config.disable_glog_info()
        if self.args.use_cuda:
            config.enable_use_gpu(memory_pool_init_size_mb=500,
                                  device_id=int(
                                      os.getenv('FLAGS_selected_gpus', '0')),
                                  )
        self.predictor = create_paddle_predictor(config)

    def load_customization(self, customization_file, sep=None):
        """装载用户词典

        Args:
            texts: 用户词典路径
            sep: 表示词典中，短语片段的分隔符，默认为空格' '或制表符'\t'
        """
        self.custom = Customization()
        self.custom.load_customization(customization_file, sep)
    
    def add_word(self, word, sep=None):
        """添加单词，格式与用户词典一致
        Args:
            texts: 用户定义词典，如："春天"、"花 开"、"春天/SEASON"、"花/n 开/v"、
            sep: 表示词典中，短语片段的分隔符，默认为空格' '或制表符'\t'
        """
        if self.custom is None:
            self.custom = Customization()
        self.custom.add_word(word, sep)

    def texts2tensor(self, texts):
        """将文本输入转为Paddle输入的Tensor
        Args:
            texts: 由string组成的list，模型输入的文本     
        Returns:
            tensor: Paddle模型输入用的文本Tensor
            words_length: 记录送入模型的每一个单词的长度
        """
        lod, data, words_length = [0], [], []
        for i, text in enumerate(texts):
            if self.mode == 'seg':
                text_inds, word_length = self.dataset.text_to_ids(text, grade='char')
            else:  
                text_inds, word_length = self.dataset.text_to_ids(text)
                words_length.append(word_length) 

            data += text_inds
            lod.append(len(text_inds) + lod[i])

        data_np = np.array(data, dtype="int64")
        tensor = fluid.core.PaddleTensor(data_np)
        tensor.lod = [lod]
        tensor.shape = [lod[-1], 1]

        return tensor, words_length

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
