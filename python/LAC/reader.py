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
import argparse
import logging
import __future__
import io
import jieba_fast as jieba


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
        self.tag_type = args.tag_type

        self.args = args
        self.dev_count = dev_count

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

    def parse_seg(self, line):
        """convert segment data to lac data format"""
        tags = []
        words = line.strip().split()

        for word in words:
            if len(word) == 1:
                tags.append('-S')
            else:
                tags += ['-B'] + ['-I'] * (len(word) - 2) + ['-E']

        return "".join(words), tags

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

        return words, tags
        # return "".join(words), tags
    
    def mix_word_to_ids(self, text, key=False):
        """convert mix (word and char) to word index
        Args:
            text: 输入文本，key模型下是经过lac得到的分词结果以及词性标签
            key : 模型是否是关键词模型

        Return:
            LAC:
                word_ids:  字词混合粒度的文本idx
                del_index: 输入文本中，存在于词典中，所以以词粒度进行计算的单词，其除首位剩下的其他字符绝对位置，
                           用于后续在计算出词性标签后，将全部文本变成char粒度时，重新补充，来进行人工字典干预。
            KEY:
                word_ids:  字词混合粒度的文本idx
                tag_ids:   字词混合粒度的词性标签idx
                add_index: 输入文本中，不存在于词典中，所以将词拆分成char粒度，由被拆分单词的绝对位置以及单词长度组成，
                           用于在后续计算一个单词的关键度，合并其被拆分成多个char的标签。
        """
        word_ids = []

        if key:
            text, tag = text
            tag_ids= []
        else:
            text = jieba.lcut(text, HMM=False)
        
        word_ids, del_index, add_index = self.word_to_ids(text)

        if key:
            for local, length in add_index:  # 由于部分单词被拆分成char，所以tag也要对应增加
                for _ in range(length-1):
                    tag.insert(local+1, tag[local].split('-')[0] + '-I')
            tag_ids = self.label_to_ids(tag)
        """
        for i, word in enumerate(text):
            if len(word) == 1 or word in self.word2id_dict.keys():
                mix_text.append(word)
                word = self.word_replace_dict.get(word, word)
                word_id = self.word2id_dict.get(word, self.oov_id)
                word_ids.append(word_id)
                
                if key:
                    if tag[i] not in self.label2id_dict:
                        tag[i] = "O"
                    tag_id = self.label2id_dict[tag[i]]
                    tag_ids.append(tag_id)
                    start += 1
            else:
                for a, w in enumerate(word):
                    mix_text.append(w)
                    w = self.word_replace_dict.get(w, w)
                    word_id = self.word2id_dict.get(w, self.oov_id)
                    word_ids.append(word_id)

                    if key:
                        if tag[i] not in self.label2id_dict:
                            tag[i] = "O"
                        if a == 0:
                            tag_id = self.label2id_dict[tag[i]]
                        else:
                            tag_id = self.label2id_dict[tag[i].split('-')[0] + '-I']
                        tag_ids.append(tag_id)
                if key:
                    end = start + len(word)
                    seg_local.insert(0, [start, end])
                    start = end
            """
        if key:
            return word_ids, tag_ids, add_index
        else:
            return word_ids, del_index

    def word_to_ids(self, words, gradind='mix'):
        """convert word to word index"""
        word_ids = []
        if gradind == 'char':
            for word in words:
                word = self.word_replace_dict.get(word, word)
                word_id = self.word2id_dict.get(word, self.oov_id)
                word_ids.append(word_id)
            return word_ids

        else:
            del_index, add_index= [], []
            start = 0
            for i, word in enumerate(words):
                end = start + len(word)   
                if word in self.word2id_dict.keys() or len(word) == 1:
                    word = self.word_replace_dict.get(word, word)
                    word_id = self.word2id_dict.get(word, self.oov_id)
                    word_ids.append(word_id)
                    if len(word) > 1:
                        del_index += [x for x in range(start+1, end)]  # 定位存在字典里的单词，除首位剩下的其他字符绝对位置
                else:
                    for w in word:
                        w = self.word_replace_dict.get(w, w)
                        word_id = self.word2id_dict.get(w, self.oov_id)
                        word_ids.append(word_id)
                    add_index.insert(0, [i, len(word)])  # 定位被拆分单词的绝对位置以及单词长度
                start = end
            return word_ids, del_index, add_index

    def label_to_ids(self, labels):
        """convert label to label index"""
        label_ids = []
        for label in labels:
            if label not in self.label2id_dict:
                label = "O"
            label_id = self.label2id_dict[label]
            label_ids.append(label_id)
        return label_ids

    def word_label_to_ids(self, words, labels):
        
        if self.tag_type == 'seg':
            word_ids = self.word_to_ids(words, gradind='char')
            label_ids = self.label_to_ids(labels)
            return word_ids, label_ids
        else:
            # words = "".join(words) # 可选用是否利用jieba分词粒度
            # words = jieba.lcut(words, HMM=False) 

            word_ids, del_index, add_index = self.word_to_ids(words, gradind='mix')
            label_ids = self.label_to_ids(labels)
            
            label_ids = [label_ids[i] for i in range(len(label_ids)) if (i not in del_index)]  # 删除lab

        return word_ids, label_ids


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
                for a, line in enumerate(fread):
                    if (len(line.strip()) == 0):
                        continue
                    if self.tag_type == 'seg':
                        words, labels = self.parse_seg(line)
                    elif self.tag_type == 'tag':
                        words, labels = self.parse_tag(line)
                    else:
                        line = line.strip('\n').split('\t')
                        if len(line) != 2:
                            continue
                        words, labels = line  # 训练数据处理过程
                        words = [word for i, word in enumerate(words) if i%2==0]
                        labels = [x for x in labels.split('\002')]
                        if len(words) != len(labels):
                            continue
                        words = ''.join(words)

                        # words, labels = line.strip("\n").split("\t")
                        # words = words.split("\002")
                        # labels = labels.split("\002")

                    # word_ids = self.word_to_ids(words)  # 原始char输入
                    # label_ids = self.label_to_ids(labels)
                    word_ids, label_ids = self.word_label_to_ids(words, labels)  # 修改字词混合并进行标签对齐操作

                    assert len(word_ids) == len(label_ids)
                    yield word_ids, label_ids
                    cnt += 1

                if mode == 'train':
                    pad_num = self.dev_count - \
                        (cnt % self.args.batch_size) % self.dev_count
                    for i in range(pad_num):
                        if self.tag_type == 'seg':
                            yield [self.oov_id], [self.label2id_dict['-S']]
                        else:
                            yield [self.oov_id], [self.label2id_dict['O']]
            fread.close()

        return wrapper


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
