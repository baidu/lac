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


class LAC(object):
    """docstring for LAC"""

    def __init__(self, model_path=None, mode='lac', use_cuda=False):
        super(LAC, self).__init__()
        utils.check_cuda(use_cuda)
        if model_path is None:
            model_path = DEFAULT_SEG if mode == 'seg' else DEFAULT_LAC

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
        self.return_tag = mode != 'seg'

    def run(self, texts):
        """执行模型预测过程

        Args:
            texts: 模型输入的文本，一个Unicode编码的字符串或者
                   由Unicode编码字符串组成的List

        Returns:
            返回LAC处理结果
            如果mode=='seg', 则只返回分词结果
            如果mode=='lac', 则同时返回分词与标签
        """
        if isinstance(texts, list) or isinstance(texts, tuple):
            self.batch = True
        else:
            if len(texts.strip()) == 0:
                return ([], []) if self.return_tag else []
            texts = [texts]
            self.batch = False

        tensor_words = self.texts2tensor(texts)
        crf_decode = self.predictor.run([tensor_words])

        result = self.parse_result(texts, crf_decode[0], self.dataset)

        if self.return_tag:
            return result if self.batch else result[0]
        else:
            if not self.batch:
                return result[0][0]
            return [word for word, _ in result]

    def parse_result(self, lines, crf_decode, dataset):
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
            if self.custom:
                self.custom.parse_customization(sent, tags)

            sent_out = []
            tags_out = []
            for ind, tag in enumerate(tags):
                # for the first char
                if len(sent_out) == 0 or tag.endswith("B") or tag.endswith("S"):
                    sent_out.append(sent[ind])
                    tags_out.append(tag[:-2])
                    continue
                sent_out[-1] += sent[ind]
                # 取最后一个tag作为标签
                tags_out[-1] = tag[:-2]

            batch_out.append([sent_out, tags_out])
        return batch_out

    def train(self, model_save_dir, train_data, test_data=None):
        """执行模型增量训练

        Args:
            model_save_dir: 训练结束后模型保存的路径
            train_data: 训练数据路径
            test_data: 测试数据路径，若为None则不进行测试
        """
        self.args.train_data = train_data
        if test_data:
            self.args.test_data = test_data
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

    def load_customization(self, customization_file):
        """装载用户词典"""
        self.custom = Customization()
        self.custom.load_customization(customization_file)

    def texts2tensor(self, texts):
        """将文本输入转为Paddle输入的Tensor

        Args:
            texts: 由string组成的list，模型输入的文本

        Returns:
            Paddle模型输入用的Tensor
        """
        lod = [0]
        data = []
        for i, text in enumerate(texts):
            text_inds = self.dataset.word_to_ids(text)
            data += text_inds
            lod.append(len(text_inds) + lod[i])
        data_np = np.array(data, dtype="int64")
        tensor = fluid.core.PaddleTensor(data_np)
        tensor.lod = [lod]
        tensor.shape = [lod[-1], 1]

        return tensor


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
