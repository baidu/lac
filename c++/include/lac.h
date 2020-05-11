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
#include "paddle_inference_api.h"

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


#ifndef LAC_CLASS
#define LAC_CLASS

// 前向声明, 去除头文件依赖
class Customization;

class LAC
{
public:
    /* 初始化：装载模型和词典 */
    LAC(LAC &lac);      // 
    LAC(const std::string& model_path, CODE_TYPE type = CODE_UTF8);

    /* 调用程序 */
    std::vector<OutputItem> run(const std::string &query);                           // 单个query
    std::vector<std::vector<OutputItem>> run(const std::vector<std::string> &query); // 批量query

    /* 装载用户词典 */
    int load_customization(const std::string& filename);

private:
    /* 将字符串输入转为Tensor */
    int feed_data(const std::vector<std::string> &querys);

    /* 将模型标签结果转换为模型输出格式 */
    int parse_targets(
        const std::vector<std::string> &tag_ids,
        const std::vector<std::string> &words,
        std::vector<OutputItem> &result);

    // 编码类型，需要同时修改字典文件的编码
    CODE_TYPE _codetype;

    // 中间变量
    std::vector<std::string> _seq_words;
    std::vector<std::vector<std::string>> _seq_words_batch;
    std::vector<std::vector<size_t>> _lod;
    std::vector<std::string> _labels;
    std::vector<OutputItem> _results;
    std::vector<std::vector<OutputItem>> _results_batch;

    // 数据转换词典
    std::shared_ptr<std::unordered_map<int64_t, std::string>> _id2label_dict;
    std::shared_ptr<std::unordered_map<std::string, std::string>> _q2b_dict;
    std::shared_ptr<std::unordered_map<std::string, int64_t>> _word2id_dict;
    int64_t _oov_id;

    // paddle数据结构
    paddle::PaddlePlace _place;                             //PaddlePlace::kGPU，KCPU
    std::unique_ptr<paddle::PaddlePredictor> _predictor;    // 预测器
    std::unique_ptr<paddle::ZeroCopyTensor> _input_tensor;  // 输入空间
    std::unique_ptr<paddle::ZeroCopyTensor> _output_tensor; // 输出空间


    // 人工干预词典
    std::shared_ptr<Customization> custom;
};
#endif  // LAC_CLASS

#endif  // BAIDU_LAC_LAC_H
