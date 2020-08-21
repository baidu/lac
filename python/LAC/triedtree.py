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
该模块实现Tried树，用于进行词典的多模匹配
"""

class Node(object):
    """Trie树的结点.

    Attributes:
        next: dict类型，指向子结点
        length: int类型，判断节点是否为单词
    """
    __slots__ = ['next', 'length']

    def __init__(self):
        """初始化空节点."""
        self.next = {}
        self.length = -1


class TriedTree(object):
    """实现Tried树的类

    Attributes:
        __root: Node类型，Tried树根节点
    """

    def __init__(self):
        """初始化TriedTree的根节点__root"""
        self.__root = Node()

    def add_word(self, word):
        """添加单词word到Trie树中"""
        current = self.__root
        for char in word:
            current = current.next.setdefault(char, Node())
        current.length = len(word)

    def make(self):
        """nothing to do"""
        pass

    def search(self, content):
        """前向最大匹配.

        对content的文本进行多模匹配，返回后向最大匹配的结果.

        Args:
            content: string类型, 用于多模匹配的字符串

        Returns:
            list类型, 最大匹配单词列表，每个元素为匹配的模式串在句中的起止位置，比如：
            [(0, 2), [4, 7]]

        """
        result = []

        length = len(content)
        current_position = 0
        end_position = 0
        while current_position < length:
            p = self.__root
            matches = []
            for key in content[current_position:]:
                p = p.next.get(key, None)
                if not p:
                    break
                if p.length > 0:
                    end_position = current_position + p.length
                    matches.append((current_position, end_position))
            if len(matches) > 0:
                result.append((matches[-1][0], matches[-1][1]))
            current_position = max(current_position + 1, end_position)

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

        length = len(content)
        for current_position in range(length):
            p = self.__root
            for key in content[current_position:]:
                p =  p.next.get(key, None)
                if not p:
                    break
                if p.length > 0:
                    result.append(
                        (current_position, current_position + p.length))

        return result


if __name__ == "__main__":
    words = ["百度", "家", "家家", "高科技", "技公", "科技", "科技公司"]
    string = '百度是家高科技公司'
    tree = TriedTree()
    for word in words:
        tree.add_word(word)

    for begin, end in tree.search(string):
        print(string[begin:end])


