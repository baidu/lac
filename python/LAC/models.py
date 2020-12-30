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
本文件定义了Model基类以及它的子类:LacModel, SegModel, RankModel
""" 
import os
import shutil
import logging

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

from . import nets
from . import utils
from . import reader
from .segment import Segment
from .custom import Customization

class Model(object):
    """Docstring for Model"""
    def __init__(self, model_path, mode, use_cuda):
        super(Model, self).__init__()

        self.mode = mode
        self.model_path = model_path

        self.args = utils.DefaultArgs(self.model_path)
        self.args.use_cuda = use_cuda

        utils.check_cuda(self.args.use_cuda)

        config = AnalysisConfig(self.args.init_checkpoint)
        config.disable_glog_info()

        if self.args.use_cuda:
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
        self.segment_tool = None
        self.custom = None
        self.batch = False

    def run(self, texts):
        """文本输入经过模型转为运行结果Tensor"""
        tensor_words, words_length = self.texts2tensor(texts)

        if tensor_words is None:
            return {
                    "crf_result":  [[[], [], []]] * len(texts)
                    }

        crf_decode = self.predictor.run([tensor_words])
        crf_result = self.parse_result(texts, crf_decode[0], self.dataset, words_length)

        return {
                "crf_decode": crf_decode,
                "crf_result": crf_result,
                "tensor_words": tensor_words,
                "words_length": words_length
                }   

    def to_tensor(self, data, lod, dtype="int64"):
        """Ids to Tensor"""
        data_np = np.array(data, dtype)
        tensor = fluid.core.PaddleTensor(data_np)
        tensor.lod = [lod]
        tensor.shape = [lod[-1], 1]
        return tensor

    def texts2tensor(self, texts):
        """文本输入转为Paddle输入的Tensor,适用于lac与rank
        Args:
            texts: 由string组成的list，模型输入的文本     
        Returns:
            tensor: Paddle模型输入用的文本Tensor
            words_length: 记录送入模型的每一个单词的长度
        """

        lod, data, words_length = [0], [], []
        for i, text in enumerate(texts):      
            text = self.segment_tool.fast_cut(text)
            text_inds, word_length = self.dataset.text_to_ids(text)
            words_length.append(word_length)

            data += text_inds
            lod.append(len(text_inds) + lod[i])

        tensor = self.to_tensor(data, lod) if len(data) != 0 else None

        return tensor, words_length

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

            # 重新填充被省略的单词的char部分
            word_length = words_length[sent_index]
            for current in range(len(word_length)-1, -1, -1):
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

            batch_out.append([sent_out, tags_out, tags])
        return batch_out
    
    def train(self, model_save_dir, train_data, test_data, iter_num, thread_num):
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
            test_program, fetch_list = nets.do_train(self.args, self.dataset, self.segment_tool)

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

class LacModel(Model):
    """Docstring for LAC Model"""
    def __init__(self, model_path, mode, use_cuda):
        super(LacModel, self).__init__(model_path, mode, use_cuda) 

        seg_dict_path = os.path.join(model_path, "conf", "small_seg.dic")
        self.segment_tool = Segment(dict_path=seg_dict_path)

    def run(self, texts):
        if isinstance(texts, list) or isinstance(texts, tuple):
            self.batch = True
        else:
            if len(texts.strip()) == 0:
                return ([], [])
            texts = [texts]
            self.batch = False
        
        crf_result = super(LacModel, self).run(texts)['crf_result']

        result = [[word, tag] for word, tag, tag_for_rank in crf_result] if self.batch else crf_result[0][:-1]

        return result

    def call_run(self, texts):
        """lac被rank模型调用时返回的结果"""
        lac_result = super(LacModel, self).run(texts)
        return lac_result

class SegModel(Model):
    """Docstring for Seg Model"""
    def __init__(self, model_path, mode, use_cuda):
        super(SegModel, self).__init__(model_path, mode, use_cuda) 
        self.dataset = reader.SegDataset(self.args)
    
    def run(self, texts):
        if isinstance(texts, list) or isinstance(texts, tuple):
            self.batch = True
        else:
            if len(texts.strip()) == 0:
                return []
            texts = [texts]
            self.batch = False

        crf_result = super(SegModel, self).run(texts)["crf_result"]

        result = [word for word, tag, tag_for_rank in crf_result] if self.batch else crf_result[0][0]
        return result
    
    def texts2tensor(self, texts):
        """文本输入转为Paddle输入的Tensor"""
        lod, data, words_length = [0], [], []
        for i, text in enumerate(texts):
            text_inds = self.dataset.word_to_ids(text)
            data += text_inds
            lod.append(len(text_inds) + lod[i])

        tensor = self.to_tensor(data, lod) if len(data) != 0 else None

        return tensor, words_length
    
    def parse_result(self, lines, crf_decode, dataset, words_length):
        """将SEG模型输出的Tensor转为明文"""
        offset_list = crf_decode.lod[0]
        crf_decode = crf_decode.data.int64_data()
        batch_size = len(offset_list) - 1

        batch_out = []
        for sent_index in range(batch_size):
            begin, end = offset_list[sent_index], offset_list[sent_index + 1]

            sent = lines[sent_index]
            tags = [dataset.id2label_dict[str(id)]
                    for id in crf_decode[begin:end]]
            tags_for_rank = []

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

            sent_out = [''] if len(sent_out) == 0 else sent_out
            batch_out.append([sent_out, tags_out, tags_for_rank])
        return batch_out

class RankModel(Model):
    """Docstring for Rank Model"""
    def __init__(self, model_path, mode, use_cuda):
        # init rank model
        super(RankModel, self).__init__(model_path, mode, use_cuda) 

        # parsing the lac model address
        parent_path = os.path.split(model_path)[0]
        lac_path = os.path.join(parent_path, 'lac_model')

        # init lac model
        self.lac = LacModel(model_path=lac_path, mode='lac', use_cuda=use_cuda) 

    def run(self, texts):
        if isinstance(texts, list) or isinstance(texts, tuple):
            self.batch = True
        else:
            if len(texts.strip()) == 0:
                return ([], [], [])
            texts = [texts]
            self.batch = False

        if self.custom is not None:
            self.lac.custom = self.custom
            
        lac_result = self.lac.call_run(texts)

        if len(lac_result) == 1:
            return [[[], [], []]] * len(texts)
        
        crf_decode = lac_result["crf_decode"]
        crf_result = lac_result["crf_result"]
        tensor_words = lac_result["tensor_words"]
        words_length = lac_result["words_length"]

        result = [[word, tag] for word, tag, tag_for_rank in crf_result]
        tags_for_rank = [tag_for_rank for word, tag, tag_for_rank in crf_result]

        rank_decode = self.predictor.run([tensor_words, crf_decode[0]])
        weight = self.parse_result(tags_for_rank, rank_decode[0], words_length) 
        result = [result[_] + [weight[_]] for _ in range(len(result))]

        return result if self.batch else result[0]

    def parse_result(self, tags_for_rank, result, words_length):
        """将RANK模型输出的Tensor转为明文"""
        offset_list = result.lod[0]
        rank_weight = result.data.int64_data()
        batch_size = len(offset_list) - 1

        batch_out = []
        for sent_index in range(batch_size):
            begin, end = offset_list[sent_index], offset_list[sent_index + 1]

            tags = tags_for_rank[sent_index]
            word_length = words_length[sent_index]
            weight = rank_weight[begin:end]

            # 重新填充被省略的单词的char部分
            for current in range(len(word_length)-1, -1, -1):
                for offset in range(1, word_length[current]):
                    weight.insert(current + offset, weight[current])

            weight_out = []
            for ind, tag in enumerate(tags):
                if tag.endswith("B") or tag.endswith("S"):
                    weight_out.append(weight[ind])
                    continue
                weight_out[-1] = max(weight_out[-1], weight[ind])

            batch_out.append(weight_out)
        return batch_out

    def train(self, model_save_dir, train_data, test_data, iter_num, thread_num):
        logging.info("To be continued...")
        return
