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
The file_reader converts raw corpus to input.
"""
import io
import logging
import argparse
import __future__



def load_kv_dict(dict_path,
                 reverse=False, delimiter="\t", key_func=None, value_func=None):
    """
    Load key-value dict from file
    """
    result_dict = {}
    with io.open(dict_path, "r", encoding='utf8') as file:
        for line in file:
            terms = line.strip("\n").split(delimiter)
            if len(terms) != 2:
                continue
            if reverse:
                value, key = terms
            else:
                key, value = terms
            # if key in result_dict:
            #     raise KeyError("key duplicated with [%s]" % (key))
            if key_func:
                key = key_func(key)
            if value_func:
                value = value_func(value)
            result_dict[key] = value
    return result_dict

class Dataset(object):
    """data reader"""

    def __init__(self, args, dev_count=10):
        # read dict
        self.word2id_dict = load_kv_dict(
            args.word_dict_path, reverse=True, value_func=int)
        self.id2word_dict = load_kv_dict(args.word_dict_path)
        self.label2id_dict = load_kv_dict(
            args.label_dict_path, reverse=True, value_func=int)
        self.id2label_dict = load_kv_dict(args.label_dict_path)
        self.word_replace_dict = load_kv_dict(args.word_rep_dict_path)
        self.oov_id = self.word2id_dict['OOV']
        self.model = args.model

        self.args = args
        self.dev_count = dev_count
        self.segment_tool = None

    @property
    def vocab_size(self):
        """vocabuary size"""
        return max(self.word2id_dict.values()) + 1

    @property
    def num_labels(self):
        """num_labels"""
        return max(self.label2id_dict.values()) + 1

    def get_num_examples(self, filename):
        """num of line of file"""
        return sum(1 for line in open(filename, "rb"))

    def parse_tag(self, line):
        """convert tagging data to lac data format"""
        tags = []
        words = []

        items = line.strip().split()
        for item in items:
            word = item[:item.rfind('/')]
            tag = item[item.rfind('/') + 1:]
            if '/' not in item or len(word) == 0 or len(tag) == 0:
                logging.warning("Data type error: %s" % line.strip())
                return [], []
            tags += [tag + '-B'] + [tag + '-I'] * (len(word) - 1)
            words.append(word)

        return self.segment_tool.fast_cut("".join(words)), tags
    
    def word_to_ids(self, words):
        """convert words to word index"""

        word_ids = []
        for word in words:
            word = self.word_replace_dict.get(word, word)
            word_id = self.word2id_dict.get(word, self.oov_id)
            word_ids.append(word_id)
        return word_ids

    def text_to_ids(self, text):
        """convert text to word index 
           lac/rank using mix char and word granularity
           seg using char granularity
        """
        word_ids, word_length = [], []

        for word in text: 
            if word in self.word2id_dict:
                word_ids += self.word_to_ids([word])
                word_length.append(len(word))
            else:
                word_ids += self.word_to_ids(word)
                word_length += [1] * len(word)

        return word_ids, word_length

    def label_to_ids(self, labels):
        """convert label to label index"""
        label_ids = []
        for label in labels:
            if label not in self.label2id_dict:
                label = "O"
            label_id = self.label2id_dict[label]
            label_ids.append(label_id)
        return label_ids

    def file_reader(self, filename, mode="train"):
        """
        yield (word_idx, target_idx) one by one from file,
            or yield (word_idx, ) in `infer` mode
        """
        def wrapper():
            """the wrapper of data generator"""
            fread = io.open(filename, "r", encoding="utf-8")
            if mode == "infer":
                for line in fread:
                    words = line.strip()
                    word_ids = self.word_to_ids(words)
                    yield (word_ids,)
            else:
                cnt = 0
                for line in fread:
                    if (len(line.strip()) == 0):
                        continue
                        
                    texts, labels = self.parse_tag(line)
                    word_ids, word_length = self.text_to_ids(texts)

                    # 删掉以词粒度处理的多余的词性标签
                    if len(word_length) != 0:
                        for current in range(len(word_ids)):
                            if word_length[current] > 1:
                                for _ in range(1, word_length[current]):
                                    labels.pop(current)
                                labels[current] = labels[current][:-1] + 'B'
                    
                    label_ids = self.label_to_ids(labels)

                    assert len(word_ids) == len(label_ids)
                    yield word_ids, label_ids
                    cnt += 1

                if mode == 'train':
                    pad_num = self.dev_count - \
                        (cnt % self.args.batch_size) % self.dev_count
                    for i in range(pad_num):
                        if self.model == 'seg':
                            yield [self.oov_id], [self.label2id_dict['-S']]
                        elif self.model == 'lac':
                            yield [self.oov_id], [self.label2id_dict['O']]
            fread.close()

        return wrapper

class SegDataset(Dataset):
    """seg model data reader"""
    def __init__(self, args, dev_count=10):
        Dataset.__init__(self, args, dev_count)
    
    def parse_tag(self, line):
        """convert segment data to lac data format"""
        tags = []
        words = line.strip().split()

        for word in words:
            if len(word) == 1:
                tags.append('-S')
            else:
                tags += ['-B'] + ['-I'] * (len(word) - 2) + ['-E']

        return "".join(words), tags

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--word_dict_path", type=str,
                        default="./conf/word.dic", help="word dict")
    parser.add_argument("--label_dict_path", type=str,
                        default="./conf/tag.dic", help="label dict")
    parser.add_argument("--word_rep_dict_path", type=str,
                        default="./conf/q2b.dic", help="word replace dict")
    args = parser.parse_args()
    dataset = Dataset(args)
    data_generator = dataset.file_reader("data/train.tsv")
    for word_idx, target_idx in data_generator():
        print(word_idx, target_idx)
        print(len(word_idx), len(target_idx))
        break
