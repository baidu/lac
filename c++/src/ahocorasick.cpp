
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

#include<queue>

#include "ahocorasick.h"

Node* Node::get_child(const std::string &str){
    for (auto i : next){
        if (i->key == str){
            return i;
        }
    }
    return NULL;
}

Node* Node::add_child(const std::string &str){
    for (auto i : next){
        if (i->key == str){
            return i;
        }
    }
    Node* child = new Node();
    child->key = str;
    next.push_back(child);
    return child;
}

AhoCorasick::~AhoCorasick(){
    std::queue <Node*> que;
    que.push(_root);

    /* 广度遍历删除节点 */
    while (!que.empty()){
        Node* tmp = que.front();
        que.pop();
        for (auto child: tmp->next){
            que.push(child);
        }
        delete tmp;
    }
}

/* 添加AC自动机item */
void AhoCorasick::insert(const std::vector<std::string> &chars, int value){
    if (chars.size() == 0 || value < 0){
        return;
    }

    Node* root = _root;
    for (auto i : chars){
        root = root->add_child(i);
    }
    root->value = value;
}

/* 生成AC自动机的fail指针 */
void AhoCorasick::make_fail(){
    
    _root->fail = NULL;
    std::queue <Node*> que;
    for (auto child : _root->next ){
        child->fail = _root;
        que.push(child);
    }

    /* 以广度优先遍历设置fail指针 */
    while (!que.empty()){
        Node* current = que.front();
        que.pop();
        for (auto child : current->next){
            Node* current_fail = current->fail;

            // 若当前节点有fail指针，尝试设置其子结点的fail指针
            while (current_fail){
                if (current_fail->get_child(child->key)){
                    child->fail = current_fail->get_child(child->key);
                    break;
                }
                current_fail = current_fail->fail;
            }

            // 若当前节点的fail指针不存在子结点，令子结点fail指向根节点
            if (current_fail == NULL){
                child->fail = _root;
            }

            que.push(child);
        }
    }
}


/* 查询返回多模匹配结果 */
int AhoCorasick::search(const std::vector<std::string> &sentence, std::vector<std::pair<int, int>> &res, bool backtrack){
    // std::vector<std::pair<int, int> > res;
    Node *child = NULL, *p = _root;
    for (size_t i=0; i< sentence.size(); i++){
        child = p->get_child(sentence[i]);
        while (child == NULL){
            if (p == _root){
                break;
            }
            p = p->fail;
            child = p->get_child(sentence[i]);
        }
        
        if (child){
            p = child;

            while (child != _root){
                // 命中单词
                if (child->value >= 0){
                    res.push_back(std::make_pair(i, child->value));
                }

                // 不回溯，用于最大长度匹配
                if (!backtrack){
                    break;
                }

                child = child->fail;
            }
        }
    }
    return 0;
}
