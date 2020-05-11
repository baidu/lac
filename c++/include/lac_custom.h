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

#ifndef BAIDU_LAC_CUSTOM_H
#define BAIDU_LAC_CUSTOM_H

#include<vector>
#include<string>
#include <memory>

#include "lac_util.h"
#include "ahocorasick.h"

/* 干预的item */
struct customization_term{
    std::vector<std::string> tags;
    std::vector<int> split;
    customization_term(const std::vector<std::string>& tags, 
            const std::vector<int>& split):
        tags(tags),
        split(split){}
};

/* 干预使用的类 */
class Customization{
    private:
        // 记录每个item的标签和分词信息
        std::vector<customization_term> _customization_dic;  

        // AC自动机用于item的查询
        AhoCorasick _ac_dict;

        // AC自动机查询返回结果
        std::vector<std::pair<int, int>> _ac_res;

        // 中文字符处理临时变量
        std::vector<std::string> line_vector;
    public:
    Customization(const std::string &customization_dic_path){
        load_dict(customization_dic_path);
    }

    /* 从用户词典中进行装载 */
    RVAL load_dict(const std::string &customization_dic_path);

    /* 对lac的预测结果进行干预 */
    RVAL parse_customization(const std::vector<std::string> &seq_chars, std::vector<std::string> &tag_ids);
};

#endif  //BAIDU_LAC_CUSTOM_H
