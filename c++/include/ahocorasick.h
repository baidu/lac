/* Copyright (c) 2020 Baidu, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef BAIDU_LAC_AHOCORASICK_H
#define BAIDU_LAC_AHOCORASICK_H

#include<vector>
#include<utility>
#include<string>

/* AC自动机树结点 */
struct Node{
    std::vector<Node*> next;    
    std::string key;            // 当前结点的字符
    int value;                  // 结点对应的value，-1表示无
    Node* fail;                 // ac自动机的fail指针

    Node():value(-1),fail(NULL){}

    /* 返回子结点中，字符为str的结点，找不到返回NULL */
    Node* get_child(const std::string &str);

    /* 添加字符为str的子结点并返回，若已存在则直接返回原子结点 */
    Node* add_child(const std::string &str);
};

/* AC自动机 */
class AhoCorasick{
    private:
    Node * _root;

    public:
    AhoCorasick(){
        _root = new Node();
    }

    ~AhoCorasick();
    
    /* 添加AC自动机item */
    void insert(const std::vector<std::string> &chars, int value);

    /* 生成AC自动机的fail指针 */
    void make_fail();

    /* 查询返回多模匹配结果 */
    int search (const std::vector<std::string> &sentence, std::vector<std::pair<int, int>> &res, bool backtrack = false);
};

#endif  // BAIDU_LAC_AHOCORASICK_H