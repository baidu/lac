# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2020  Baidu.com, Inc. All Rights Reserved.
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
本模块定义了LAC中使用到的工具类函数
"""

from __future__ import print_function
import os
import sys
import numpy as np
import paddle.fluid as fluid

try:
    import configparser
except ImportError:
    import ConfigParser as configparser


def abs_path(path): return os.path.join(
    os.path.dirname(sys._getframe().f_code.co_filename), path)


def check_cuda(use_cuda):
    """check environment for cuda's using"""

    err = ("\nYou can not set use_cuda = True in the model "
           "because you are using paddlepaddle-cpu.\n"
           "Please: 1. Install paddlepaddle-gpu to run your models on GPU"
           "or 2. Set use_cuda = False to run models on CPU.\n")
    try:
        if use_cuda == True and fluid.is_compiled_with_cuda() == False:
            print(err)
            sys.exit(1)
    except Exception as e:
        pass


class DefaultArgs(object):
    """the default args of model"""

    def __init__(self, model_path):
        """load config file and init args"""
        config = configparser.ConfigParser()
        conf_path = os.path.join(model_path, "conf", "args.ini")
        config.read(conf_path)

        # model's parameter
        self.word_emb_dim = config.getint("NETWORK_CONFIG", "word_emb_dim")
        self.grnn_hidden_dim = config.getint(
            "NETWORK_CONFIG", "grnn_hidden_dim")
        self.bigru_num = config.getint("NETWORK_CONFIG", "bigru_num")

        # train's parameter
        self.tag_type = config.get("TRAIN_CONFIG", "tag_type")
        self.random_seed = config.getint("TRAIN_CONFIG", "random_seed")
        self.batch_size = config.getint("TRAIN_CONFIG", "batch_size")
        self.epoch = config.getint("TRAIN_CONFIG", "epoch")
        self.use_cuda = config.getboolean("TRAIN_CONFIG", "use_cuda")
        self.traindata_shuffle_buffer = config.getint(
            "TRAIN_CONFIG", "traindata_shuffle_buffer")
        self.base_learning_rate = config.getfloat(
            "TRAIN_CONFIG", "base_learning_rate")
        self.emb_learning_rate = config.getfloat(
            "TRAIN_CONFIG", "emb_learning_rate")
        self.crf_learning_rate = config.getfloat(
            "TRAIN_CONFIG", "crf_learning_rate")
        self.cpu_num = config.getint("TRAIN_CONFIG", "cpu_num")
        self.init_checkpoint = os.path.join(
            model_path, config.get("TRAIN_CONFIG", "init_checkpoint"))
        self.model_save_dir = os.path.join(model_path, abs_path(
            config.get("TRAIN_CONFIG", "model_save_dir")))

        # data path
        self.word_dict_path = os.path.join(
            model_path, config.get("DICT_FILE", "word_dict_path"))
        self.label_dict_path = os.path.join(
            model_path, config.get("DICT_FILE", "label_dict_path"))
        self.word_rep_dict_path = os.path.join(
            model_path, config.get("DICT_FILE", "word_rep_dict_path"))


def print_arguments(args):
    """none"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def to_str(string, encoding="utf-8"):
    """convert to str for print"""
    if sys.version_info.major == 3:
        if isinstance(string, bytes):
            return string.decode(encoding)
    elif sys.version_info.major == 2:
        if isinstance(string, unicode):
            if os.name == 'nt':
                return string
            else:
                return string.encode(encoding)
    return string


def to_lodtensor(data, place):
    """Convert data in list into lodtensor."""
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.Tensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def init_checkpoint(exe, init_checkpoint_path, main_program):
    """Init CheckPoint"""
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        """If existed presitabels"""
        if not fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)
    print("Load model from {}".format(init_checkpoint_path))


def init_pretraining_params(exe,
                            pretraining_params_path,
                            main_program):
    """load params of pretrained model, NOT including moment, learning_rate"""
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def _existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(pretraining_params_path, var.name))

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=_existed_params)
    print("Load pretraining parameters from {}.".format(
        pretraining_params_path))
