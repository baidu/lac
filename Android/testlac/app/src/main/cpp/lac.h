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

#ifndef BAIDU_LAC_LAC_H
#define BAIDU_LAC_LAC_H
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include "paddle_api.h"

/* 编码设置 */
enum CODE_TYPE
{
    CODE_GB18030 = 0,
    CODE_UTF8 = 1,
};

/* 模型输出的结构 */
struct OutputItem
{
    std::string word;   // 分词结果
    std::string tag;    // 单词类型
};

#endif

#ifndef LAC_CLASS
#define LAC_CLASS
class LAC
{
private:
    CODE_TYPE _codetype;

    /* 中间变量 */
    std::vector<std::string> _seq_words;
    std::vector<std::vector<std::string>> _seq_words_batch;
    std::vector<std::vector<uint64_t>> _lod;
    std::vector<std::string> _labels;
    std::vector<OutputItem> _results;
    std::vector<std::vector<OutputItem>> _results_batch;

    /* 数据转换词典 */
    std::shared_ptr<std::unordered_map<int64_t, std::string>> _id2label_dict;
    std::shared_ptr<std::unordered_map<std::string, std::string>> _q2b_dict;
    std::shared_ptr<std::unordered_map<std::string, int64_t>> _word2id_dict;
    int64_t _oov_id;

    /* paddle数据结构*/
    std::shared_ptr<paddle::lite_api::PaddlePredictor> _predictor;    // 
    std::unique_ptr<paddle::lite_api::Tensor> _input_tensor;  //  
    std::unique_ptr<const paddle::lite_api::Tensor> _output_tensor; //

private:
    /* 将字符串输入转为Tensor */
    int feed_data(const std::vector<std::string> &querys);

    /* 将模型标签结果转换为模型输出格式 */
    int parse_targets(
        const std::vector<std::string> &tag_ids,
        const std::vector<std::string> &words,
        std::vector<OutputItem> &result);

public:
    /* 初始化：装载模型和词典 */
    explicit LAC(std::string model_dict_path, int threads = 1, CODE_TYPE type = CODE_UTF8);
    /* 更新为单个字典文件, 去除protobuf依赖删除 */
    // explicit LAC(std::string model_dict_path, int threads = 1, CODE_TYPE type = CODE_UTF8);


    /* 调用程序 */
    std::vector<OutputItem> lexer(const std::string &query);                           // 单个query
    std::vector<std::vector<OutputItem>> lexer(const std::vector<std::string> &query); // batch
};
#endif

