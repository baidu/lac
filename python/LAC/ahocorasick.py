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
本模块实现AC自动机封装为Ahocorasick类，用于进行词典的多模匹配。
"""

import logging


class Node(object):
    """AC自动机的树结点.

    Attributes:
        next: dict类型，指向子结点
        fail: Node类型，AC自动机的fail指针
        length: int类型，判断节点是否为单词
    """
    __slots__ = ['next', 'fail', 'length']

    def __init__(self):
        """初始化空节点."""
        self.next = {}
        self.fail = None  # fail指针默认为None
        self.length = -1


class Ahocorasick(object):
    """实现AC自动机的类

    Attributes:
        __root: Node类型，AC自动机根节点
    """

    def __init__(self):
        """初始化Ahocorasick的根节点__root"""
        self.__root = Node()

    def add_word(self, word):
        """添加单词word到Trie树中"""
        current = self.__root
        for char in word:
            current = current.next.setdefault(char, Node())
        current.length = len(word)

    def make(self):
        """构建fail指针路径"""

        queue = list()
        for key in self.__root.next:
            self.__root.next[key].fail = self.__root
            queue.append(self.__root.next[key])

        # 广度优先算法遍历设置fail指针
        while len(queue) > 0:
            # 基于当前节点的fail指针设置其子结点的fail指针
            current = queue.pop(0)

            for k in current.next:
                current_fail = current.fail

                # 若当前节点有fail指针，尝试设置其子结点的fail指针
                while current_fail is not None:
                    if k in current_fail.next:
                        current.next[k].fail = current_fail.next[k]
                        break
                    current_fail = current_fail.fail

                # 若当前节点的fail指针不存在该子结点，令子结点fail指向根节点
                if current_fail is None:
                    current.next[k].fail = self.__root

                queue.append(current.next[k])

    def search(self, content):
        """后向最大匹配.

        对content的文本进行多模匹配，返回后向最大匹配的结果.

        Args:
            content: string类型, 用于多模匹配的字符串

        Returns:
            list类型, 最大匹配单词列表，每个元素为匹配的模式串在句中的起止位置，比如：
            [(0, 2), [4, 7]]

        """
        result = []
        p = self.__root
        for current_position in range(len(content)):
            word = content[current_position]

            #
            while word not in p.next:
                if p == self.__root:
                    break
                p = p.fail
            else:
                p = p.next[word]
                if p.length > 0:
                    result.append(
                        (current_position - p.length + 1, current_position))

        return result

    def search_all(self, content):
        """多模匹配的完全匹配.

        对content的文本进行多模匹配，返回所有匹配结果

        Args:
            content: string类型, 用于多模匹配的字符串

        Returns:
            list类型, 所有匹配单词列表，每个元素为匹配的模式串在句中的起止位置，比如：
            [(0, 2), [4, 7]]

        """
        result = []
        p = self.__root
        for current_position in range(len(content)):
            word = content[current_position]

            while word not in p.next:
                if p == self.__root:
                    break
                p = p.fail
            else:
                p = p.next[word]

                # 回溯查看是否存在以当前字符结尾的单词
                tmp = p
                while tmp != self.__root:
                    if tmp.length > 0:
                        result.append(
                            (current_position - tmp.length + 1, current_position))
                    tmp = tmp.fail

        return result


if __name__ == '__main__':

    ah = Ahocorasick()
    x = ["百度", "家", "高科技", "科技", "科技公司"]
    for i in x:
        ah.add_word(i)
    ah.make()
    string = '百度是家高科技公司'
    for begin, end in ah.search_all(string):
        print('all:', string[begin:end + 1])

    for begin, end in ah.search(string):
        print('search:', string[begin:end + 1])
