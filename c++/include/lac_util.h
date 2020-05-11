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

#ifndef BAIDU_LAC_LAC_UTIL_H
#define BAIDU_LAC_LAC_UTIL_H

#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>

#include "lac.h"
/* 函数返回值 */
enum RVAL
{
        _SUCCESS = 0,
        _FAILD = -1,
};

/* 以pattern作为切割符，对line进行切分并放入tokens中 */
RVAL split_tokens(const std::string &line, const std::string &pattern,
                  std::vector<std::string> &tokens);

/* 装载字符串输入转为数字的词典 */
RVAL load_word2id_dict(const std::string &filepath,
                       std::unordered_map<std::string, int64_t> &kv_dict);

/* 装载字符正则化的词典 */
RVAL load_q2b_dict(const std::string &filepath,
                   std::unordered_map<std::string, std::string> &kv_dict);

/* 装载label解码用的词典 */
RVAL load_id2label_dict(const std::string &filepath,
                        std::unordered_map<int64_t, std::string> &kv_dict);

/* 获取下一个字的长度 */
int get_next_gb18030(const char *str);
int get_next_utf8(const char *str);
int get_next_word(const char *str, CODE_TYPE codetype);

/* 将字符串按照单字切分 */
RVAL split_words(const char *input, int len, CODE_TYPE codetype, std::vector<std::string> &words);
RVAL split_words(const std::string &input, CODE_TYPE codetype, std::vector<std::string> &words);

#endif  // BAIDU_LAC_LAC_UTIL_H


