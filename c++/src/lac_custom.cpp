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

#include<iostream>
#include "lac_custom.h"

/* 从用户词典中进行装载 */
RVAL Customization::load_dict(const std::string &customization_dic_path){
    std::ifstream fin;
    fin.open(customization_dic_path.c_str());
    if (!fin) {
        std::cerr << "Load customization dic failed ! -- " << customization_dic_path
                << " not exist" << std::endl;
        return _FAILD;
    }

    std::string line;
    
    // 中文字符处理临时变量
    std::vector<std::string> line_vector;
    
    while (getline(fin, line))
    {
        if (line.length() < 1){
                continue;
        }
        if (split_tokens(line, " ", line_vector) < _SUCCESS) {
            std::cerr << "Load customization dic failed ! -- format error "
                    << line << std::endl;
            return _FAILD;
        }
        
        // 读取用户字典文件并存入相应数据结构
        std::vector<std::string> chars;
        std::vector<std::string> phrase;
        std::vector<std::string> tags;
        std::vector<int> split;
        int length = 0;
        for (auto kv : line_vector){
            if (kv.length() < 1){
                continue;
            }
            // 将中文字符串拆分为字
            std::string word = kv.substr(0, kv.rfind("/"));
            if (kv.length()>1){
                split_words(word, CODE_UTF8, chars);
            }else{
                split_words(kv, CODE_UTF8, chars);
            }
            
            phrase.insert(phrase.end(), chars.begin(), chars.end());
            length += chars.size();
            std::string tag = (word.length() < kv.size()) ? kv.substr(kv.rfind("/") + 1) : "";
            tags.push_back(tag);
            split.push_back(length);
        }
        int value = _customization_dic.size();
        _customization_dic.push_back(customization_term(tags, split));
        _ac_dict.insert(phrase, value);
    }
    _ac_dict.make_fail();

    fin.close();
    
    std::cerr << "Loaded customization dic -- num = " << _customization_dic.size()
            << std::endl;
    return _SUCCESS;
}

/* 对lac的预测结果进行干预 */
RVAL Customization::parse_customization(const std::vector<std::string> &seq_chars, std::vector<std::string> &tag_ids){
    // AC自动机查询返回结果
    std::vector<std::pair<int, int>> ac_res;
    _ac_dict.search(seq_chars, ac_res);
    
    int pre_begin = -1, pre_end = -1;
    for (auto ac_pair : ac_res){
        int value = ac_pair.second;
        int length = _customization_dic[value].split.back();
        int begin = ac_pair.first - length + 1;

        // 对查询结果进行预处理
        if (pre_begin < begin && pre_end >= begin){
            continue;
        }
        pre_begin = begin;
        pre_end = ac_pair.first;

        // 修正标注中的标签
        for (size_t i=0; i<_customization_dic[value].split.size(); i++){
            std::string tag = _customization_dic[value].tags[i];
            for (int j=0; j<_customization_dic[value].split[i]; j++){
                if (tag.length() < 1){
                    tag_ids[begin][tag_ids[begin].length()-1] = 'I';
                }
                else{
                    tag_ids[begin] = tag + "-I";
                }
                begin ++;
            }
        }

        // 修正标注中的分词
        begin = ac_pair.first - length + 1;
        tag_ids[begin][tag_ids[begin].length()-1] = 'B';
        for (size_t i=0; i<_customization_dic[value].split.size(); i++){
            size_t ind = begin+_customization_dic[value].split[i];
            if (ind < tag_ids.size()){
                tag_ids[ind][tag_ids[ind].length()-1] = 'B';
            }
        }
    }
    return _SUCCESS;
}
