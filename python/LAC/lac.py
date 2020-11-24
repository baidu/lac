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
DEFAULT_KEY = _get_abs_path('key_model')


class LAC(object):
    """docstring for LAC"""

    def __init__(self, model_path=None, mode='lac', use_cuda=False):
        super(LAC, self).__init__()
        utils.check_cuda(use_cuda)
        if model_path is None:
            if mode == 'seg':
                model_path = DEFAULT_SEG
            else:
                model_path = DEFAULT_LAC
        self.mode = mode

        self.args = utils.DefaultArgs(model_path)
        self.args.use_cuda = use_cuda
        self.model_path = model_path
        ################
        config = AnalysisConfig(self.args.init_checkpoint)
        config.disable_glog_info()
        ################

        if use_cuda:
            self.place = fluid.CUDAPlace(
                int(os.getenv('FLAGS_selected_gpus', '0')))
            ################
            config.enable_use_gpu(memory_pool_init_size_mb=500,
                                  device_id=int(
                                      os.getenv('FLAGS_selected_gpus', '0')),
                                  )
            ################
        else:
            self.place = fluid.CPUPlace()

        # init executor
        self.exe = fluid.Executor(self.place)

        self.dataset = reader.Dataset(self.args)

        self.predictor = create_paddle_predictor(config)

        self.custom = None
        self.batch = False
        self.return_tag = self.args.tag_type != 'seg'
    
    def reload(self, second_mode):
        """重初始化部分类定义参数"""
        if second_mode == 'key':
            model_path = '/'.join(self.model_path.split('/')[:-1]) + '/key_model'
        else:
            model_path = '/'.join(self.model_path.split('/')[:-1]) + '/lac_model'

        self.args = utils.DefaultArgs(model_path)
        self.model_path = model_path
        ################
        config = AnalysisConfig(self.args.init_checkpoint)
        config.disable_glog_info()
        ################

        # init executor
        self.exe = fluid.Executor(self.place)

        self.predictor = create_paddle_predictor(config)

        # return True

    def run(self, texts):
        """执行模型预测过程

        Args:
            texts: 模型输入的文本，一个Unicode编码的字符串或者
                   由Unicode编码字符串组成的List

        Returns:
            返回LAC处理结果
            如果mode=='seg', 则只返回分词结果
            如果mode=='lac', 则同时返回分词与标签
            如果mode=='key', 则同时返回分词与关键程度结果
        """
        if isinstance(texts, list) or isinstance(texts, tuple):
            self.batch = True
        else:
            if len(texts.strip()) == 0:
                return ([], []) if self.return_tag else []
            texts = [texts]
            self.batch = False

        tensor_words, mix_data = self.texts2tensor(texts)
        crf_decode = self.predictor.run([tensor_words])

        result = self.parse_result(texts, crf_decode[0], self.dataset, mix_data)

        if self.return_tag and self.mode == 'lac':
            return result if self.batch else result[0]

        elif self.mode == 'seg':
            if not self.batch:
                return result[0][0]
            return [word for word, _ in result]

        else:
            self.reload(second_mode='key')
            
            tensor_words, tensor_tags, segs = self.texts2tensor(result, key=True)
            key_decode = self.predictor.run([tensor_words, tensor_tags])
            tags = self.parse_key(texts, key_decode[0])

            key_result = []
            for data, tag, seg in zip(result, tags, segs):  #  标签与单词对齐
                if len(seg) != 0:
                    for start, end in seg:
                        tag = tag[:start] + [max(tag[start:end])] + tag[end:]
                key_result.append([data[0], tag])

            self.reload(second_mode='lac')

            return key_result if self.batch else key_result[0]
    
    def parse_key(self, lines, result):
        """将key模型输出的Tensor转为明文"""
        offset_list = result.lod[0]
        all_tags = result.data.int64_data()
        batch_size = len(offset_list) - 1

        batch_out = []
        for sent_index in range(batch_size):
            begin, end = offset_list[sent_index], offset_list[sent_index + 1]

            sent = lines[sent_index]
            tags = all_tags[begin:end]

            batch_out.append(tags)
        return batch_out

    def parse_result(self, lines, crf_decode, dataset, mix_data=None):
        """将模型输出的Tensor转为明文"""
        offset_list = crf_decode.lod[0]
        crf_decode = crf_decode.data.int64_data()
        batch_size = len(offset_list) - 1

        batch_out = []
        for sent_index in range(batch_size):
            begin, end = offset_list[sent_index], offset_list[sent_index + 1]

            sent = lines[sent_index]
            tags = [dataset.id2label_dict[str(id)]
                    for id in crf_decode[begin:end]]

            if len(mix_data) != 0:
                sent_mix = mix_data[sent_index]
                tags_mix = []
                for word, tag in zip(sent_mix, tags):
                    if len(word) == 1:
                        tags_mix.append(tag)
                    else:
                        for _ in range(len(word)):  # 按照char长度补全字词混合模型的tag
                            if _ == 0 :
                                tags_mix.append(tag)
                            else:
                                tags_mix.append(tag[:-2] + '-I')
                tags = tags_mix

            if self.custom:
                self.custom.parse_customization(sent, tags)

            sent_out, tags_out = [], []
            for ind, tag in enumerate(tags):
                # for the first char
                if len(sent_out) == 0 or tag.endswith("B") or tag.endswith("S"):
                    sent_out.append(sent[ind])
                    if self.mode == 'key':
                        tags_out.append(tag)
                    else:
                        tags_out.append(tag[:-2])
                    continue
                sent_out[-1] += sent[ind]
                # lac取最后一个tag作为标签，key跳过
                if self.mode != 'key':
                    tags_out[-1] = tag[:-2]

            batch_out.append([sent_out, tags_out])
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
        self.return_tag = self.args.tag_type != 'seg'

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
    
    def to_tensor(self, data, lod):
        """
        Args:
            data: 文档或者词性标签的idx
            lod: 句子长度及顺序信息
        """
        data_np = np.array(data, dtype="int64")
        tensor = fluid.core.PaddleTensor(data_np)
        tensor.lod = [lod]
        tensor.shape = [lod[-1], 1]
        return tensor

    def texts2tensor(self, texts, key=False):
        """将文本输入转为Paddle输入的Tensor

        Args:
            texts: 由string组成的list，模型输入的文本
            lac和seg版本，texts = [line]，只要送文本就可以了
            key版本，texts=[[line], [tag]]，都要送进去
        
        Returns:
            Paddle模型输入用的Tensor
        
            LAC: 
                mix_data: 表示字词混合拆分的文本，后续会将词以及对应的label以char形式拆分，经过词典干预后再合并

            KEY: 
                tag_tensor: LAC词性标签转成tensor
                seg: 表示经过LAC后，重新word embedding时记录把词语拆分成char的绝对位置，用于后续合并
        """
        lod = [0]
        data, mix_data, tag, seg = [], [], [], []
        for i, text in enumerate(texts):
            if self.mode == 'seg':
                text_inds = self.dataset.word_to_ids(text)

            elif self.mode == 'lac' or key is not True:  
                text_inds, mix_words = self.dataset.mix_word_to_ids(text)
                mix_data.append(mix_words)

            else:  # key
                text_inds, tag_inds, seg_local = self.dataset.mix_word_to_ids(text, key=True)
                tag += tag_inds
                seg.append(seg_local)
            
            data += text_inds
            lod.append(len(text_inds) + lod[i])

        tensor = self.to_tensor(data, lod)

        if key is not True:
            return tensor, mix_data

        tag_tensor = self.to_tensor(tag, lod)
        return tensor, tag_tensor, seg

if __name__ == "__main__":
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
    result = lac.run(test_data, return_tag=True)
    for i, (sent, tags) in enumerate(result):
        result_list = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)]
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
    result = lac.run(test_data, return_tag=True)
    for i, (sent, tags) in enumerate(result):
        result_list = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)]
        print(''.join(result_list))
