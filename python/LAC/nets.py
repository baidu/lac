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
本模块定义了模型的网络结构
"""

import os
import math

import paddle.fluid as fluid
import paddle
import multiprocessing

from .reader import Dataset
from . import utils


def lex_net(word, args, vocab_size, num_labels, target=None):
    """定义LAC的网络结构

    Args:
        word: 模型输入的tensor
        args: 模型参数
        vocab_size: 词表大小
        num_labels: 标签的数量
        target: 标签结果，如果为None则只返回decode不返回loss

    Returns:
        loss: 模型的loss，如果target为None则不返回
        decode: 模型的输出结果
    """

    word_emb_dim = args.word_emb_dim
    grnn_hidden_dim = args.grnn_hidden_dim
    emb_lr = args.emb_learning_rate if 'emb_learning_rate' in dir(
        args) else 1.0
    crf_lr = args.emb_learning_rate if 'crf_learning_rate' in dir(
        args) else 1.0
    bigru_num = args.bigru_num
    init_bound = 0.1
    IS_SPARSE = True

    def _bigru_layer(input_feature):
        """定义Bi-GRU层"""

        pre_gru = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru = fluid.layers.dynamic_gru(
            input=pre_gru,
            size=grnn_hidden_dim,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        pre_gru_r = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru_r = fluid.layers.dynamic_gru(
            input=pre_gru_r,
            size=grnn_hidden_dim,
            is_reverse=True,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
        return bi_merge

    def _net_conf(word, target=None):
        """设置网络参数和结构"""

        word_embedding = fluid.layers.embedding(
            input=word,
            size=[vocab_size, word_emb_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(
                learning_rate=emb_lr,
                name="word_emb",
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound)))

        input_feature = word_embedding
        for i in range(bigru_num):
            bigru_output = _bigru_layer(input_feature)
            input_feature = bigru_output

        emission = fluid.layers.fc(
            size=num_labels,
            input=bigru_output,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        if target is not None:
            crf_cost = fluid.layers.linear_chain_crf(
                input=emission,
                label=target,
                param_attr=fluid.ParamAttr(
                    name='crfw',
                    learning_rate=crf_lr))
            avg_cost = fluid.layers.mean(x=crf_cost)
            crf_decode = fluid.layers.crf_decoding(
                input=emission, param_attr=fluid.ParamAttr(name='crfw'))
            return avg_cost, crf_decode

        else:
            size = emission.shape[1]
            fluid.layers.create_parameter(shape=[size + 2, size],
                                          dtype=emission.dtype,
                                          name='crfw')
            crf_decode = fluid.layers.crf_decoding(
                input=emission, param_attr=fluid.ParamAttr(name='crfw'))

        return crf_decode

    return _net_conf(word, target)


def create_model(args, vocab_size, num_labels, mode='train'):
    """创建LAC的模型"""

    # 模型输入定义
    words = fluid.layers.data(
        name='words', shape=[-1, 1], dtype='int64', lod_level=1)
    targets = fluid.layers.data(
        name='targets', shape=[-1, 1], dtype='int64', lod_level=1)

    # 生成预测用的网络
    if mode == 'infer':
        crf_decode = lex_net(words, args, vocab_size,
                             num_labels, target=None)
        return {"feed_list": [words],
                "words": words,
                "crf_decode": crf_decode, }

    # 生成测试和训练用网络
    avg_cost, crf_decode = lex_net(
        words, args, vocab_size, num_labels, target=targets)

    (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
     num_correct_chunks) = fluid.layers.chunk_eval(
        input=crf_decode,
        label=targets,
        chunk_scheme="IOB",
        num_chunk_types=int(math.ceil((num_labels - 1) / 2.0)))
    chunk_evaluator = fluid.metrics.ChunkEvaluator()
    chunk_evaluator.reset()

    ret = {
        "feed_list": [words, targets],
        "words": words,
        "targets": targets,
        "avg_cost": avg_cost,
        "crf_decode": crf_decode,
        "chunk_evaluator": chunk_evaluator,
        "num_infer_chunks": num_infer_chunks,
        "num_label_chunks": num_label_chunks,
        "num_correct_chunks": num_correct_chunks
    }
    return ret


def create_pyreader(args, file_name, feed_list, place,
                    reader=None, iterable=True, for_test=False):
    """创建PyReader用于Paddle读取数据

    Args:
        args: 模型参数，定义于utils.DefaultArgs
        file_name: string类型，数据文件路径
        feed_list: list类型，模型输入的列表
        place: Paddle执行的空间，即GPU和CPU
        reader: 读取数据用的类，定义与reader.py
        iterable: 是否返回可迭代的PyReader
        for_test: 是否用于测试，如果测试则不shuffle

    Returns:
        PyReader对象，用于迭代读取数据
    """
    # init reader
    pyreader = fluid.io.PyReader(
        feed_list=feed_list,
        capacity=50,
        use_double_buffer=True,
        iterable=iterable
    )
    if reader is None:
        reader = Dataset(args)

    if for_test:
        pyreader.decorate_sample_list_generator(
            paddle.batch(
                reader.file_reader(file_name, mode='test'),
                batch_size=args.batch_size
            ),
            places=place
        )
    else:
        pyreader.decorate_sample_list_generator(
            paddle.batch(
                paddle.reader.shuffle(
                    reader.file_reader(file_name),
                    buf_size=args.traindata_shuffle_buffer
                ),
                batch_size=args.batch_size
            ),
            places=place
        )

    return pyreader


def test_process(exe, program, reader, test_ret):
    """执行测试过程

    Args:
        exe: 执行空间，即CPU和GPU
        program: 用于测试用的program
        reader: PyReader类型，读取数据

    Returns:

    """
    test_ret["chunk_evaluator"].reset()
    for data in reader():
        nums_infer, nums_label, nums_correct = exe.run(
            program,
            fetch_list=[
                test_ret["num_infer_chunks"],
                test_ret["num_label_chunks"],
                test_ret["num_correct_chunks"],
            ],
            feed=data,
        )

        test_ret["chunk_evaluator"].update(
            nums_infer, nums_label, nums_correct)
    precision, recall, f1 = test_ret["chunk_evaluator"].eval()
    print("[test] P: %.5f, R: %.5f, F1: %.5f"
          % (precision, recall, f1))


def do_train(args):
    """执行训练过程

    Args:
        args: DefaultArgs对象，在utils.py中定义，
             存储模型训练的所有参数,

    Returns:
        训练产出的program及模型输出变量
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()

    # init executor
    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        dev_count = min(multiprocessing.cpu_count(), args.cpu_num)
        os.environ['CPU_NUM'] = str(dev_count)
        place = fluid.CPUPlace()

    dataset = Dataset(args, dev_count)

    with fluid.program_guard(train_program, startup_program):
        train_program.random_seed = args.random_seed
        startup_program.random_seed = args.random_seed

        with fluid.unique_name.guard():
            train_ret = create_model(
                args, dataset.vocab_size, dataset.num_labels, mode='train')
            test_program = train_program.clone(for_test=True)

            optimizer = fluid.optimizer.Adam(
                learning_rate=args.base_learning_rate)
            optimizer.minimize(train_ret["avg_cost"])


    train_reader = create_pyreader(args, file_name=args.train_data,
                                   feed_list=train_ret['feed_list'],
                                   place=place,
                                   reader=dataset)
    if args.test_data:
        test_reader = create_pyreader(args, file_name=args.test_data,
                                  feed_list=train_ret['feed_list'],
                                  place=place,
                                  reader=dataset,
                                  iterable=True)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    if args.init_checkpoint:
        utils.init_pretraining_params(exe, args.init_checkpoint, train_program)
    if args.test_data:
        test_process(exe, test_program, test_reader, train_ret)
    if dev_count > 1:
        # multi cpu/gpu config
        exec_strategy = fluid.ExecutionStrategy()
        build_strategy = fluid.compiler.BuildStrategy()

        compiled_prog = fluid.compiler.CompiledProgram(train_program).with_data_parallel(
            loss_name=train_ret['avg_cost'].name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy
        )
    else:
        compiled_prog = fluid.compiler.CompiledProgram(train_program)

    step = 0
    fetch_list = []
    for epoch_id in range(args.epoch):
        for data in train_reader():
            outputs = exe.run(
                compiled_prog,
                fetch_list=fetch_list,
                feed=data[0],
            )
            step += 1
    if args.test_data:
        test_process(exe, test_program, test_reader, train_ret)
    return test_program, train_ret['crf_decode']
