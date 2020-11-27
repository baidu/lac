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

class TriedTree(object):
    """实现Tried树的类
    Attributes:
        __root: Node类型，Tried树根节点
    """

    def __init__(self):
        """使用单个dict存储tried"""
        self.tree = {}
        
    def add_word(self, word):
        """添加单词word到Trie树中"""
        self.tree[word] = len(word)
        for i in range(1,len(word)):
            wfrag = word[:i]
            self.tree[wfrag] = self.tree.get(wfrag, None)

    def make(self):
        """nothing to do"""
        pass

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
        length = len(content)
        for start in range(length):
            for end in range(start+1, length):
                pos = self.tree.get(content[start:end], -1)
                if pos == -1:
                    break
                if pos and (len(result)==0 or end > result[-1][1]):
                    result.append((start, end))

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
        for start in range(length):
            for end in range(start+1, length):
                pos = self.tree.get(content[start:end], -1)
                if pos == -1:
                    break
                if pos:
                    result.append((start, end))
        return result


if __name__ == "__main__":
    words = ["百度", "家", "家家", "高科技", "技公", "科技", "科技公司"]
    string = '百度是家高科技公司'
    tree = TriedTree()
    for word in words:
        tree.add_word(word)
    
    for begin, end in tree.search(string):
        print(string[begin:end])