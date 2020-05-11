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

#include "lac.h"
#include "lac_util.h"
#include "paddle_api.h"
#include "lac_custom.h"

/* LAC构造函数：初始化、装载模型和词典 */
LAC::LAC(const std::string& model_path, CODE_TYPE type)
    : _codetype(type),
      _lod(std::vector<std::vector<size_t> >(1)),
      _id2label_dict(new std::unordered_map<int64_t, std::string>),
      _q2b_dict(new std::unordered_map<std::string, std::string>),
      _word2id_dict(new std::unordered_map<std::string, int64_t>),
      custom(NULL)
{

    // 装载词典
    std::string word_dict_path = model_path + "/conf/word.dic";
    load_word2id_dict(word_dict_path, *_word2id_dict);
    std::string q2b_dict_path = model_path + "/conf/q2b.dic";
    load_q2b_dict(q2b_dict_path, *_q2b_dict);
    std::string label_dict_path = model_path + "/conf/tag.dic";
    load_id2label_dict(label_dict_path, *_id2label_dict);

    // 使用AnalysisConfig装载模型，会进一步优化模型
    this->_place = paddle::PaddlePlace::kCPU;
    paddle::AnalysisConfig config;
    // config.SwitchIrOptim(false);       // 关闭优化
    config.EnableMKLDNN();
    config.SetModel(model_path + "/model");
    config.DisableGpu();
    config.SetCpuMathLibraryNumThreads(1);
    config.SwitchUseFeedFetchOps(false);
    this->_predictor = paddle::CreatePaddlePredictor<paddle::AnalysisConfig>(config);

    // 初始化输入输出变量
    auto input_names = this->_predictor->GetInputNames();
    this->_input_tensor = this->_predictor->GetInputTensor(input_names[0]);
    auto output_names = this->_predictor->GetOutputNames();
    this->_output_tensor = this->_predictor->GetOutputTensor(output_names[0]);
    this->_oov_id = this->_word2id_dict->size() - 1;
    auto word_iter = this->_word2id_dict->find("OOV");
    if (word_iter != this->_word2id_dict->end())
    {
        this->_oov_id = word_iter->second;
    }
}

/* 拷贝构造函数，用于多线程重载 */
LAC::LAC(LAC &lac)
    : _codetype(lac._codetype),
      _lod(std::vector<std::vector<size_t> >(1)),
      _id2label_dict(lac._id2label_dict),
      _q2b_dict(lac._q2b_dict),
      _word2id_dict(lac._word2id_dict),
      _oov_id(lac._oov_id),
      _place(lac._place),
      _predictor(lac._predictor->Clone()),
      custom(lac.custom)
{
    auto input_names = this->_predictor->GetInputNames();
    this->_input_tensor = this->_predictor->GetInputTensor(input_names[0]);
    auto output_names = this->_predictor->GetOutputNames();
    this->_output_tensor = this->_predictor->GetOutputTensor(output_names[0]);
}

/* 装载用户词典 */
int LAC::load_customization(const std::string& filename){
    /* 多线程热加载时容易出问题，多个线程共享custom
    if (custom){
        return custom->load_dict(filename);
    }
    */
    custom = std::make_shared<Customization>(filename);
    return 0;
}

/* 将字符串输入转为Tensor */
int LAC::feed_data(const std::vector<std::string> &querys)
{
    this->_seq_words_batch.clear();
    this->_lod[0].clear();

    this->_lod[0].push_back(0);
    int shape = 0;
    for (size_t i = 0; i < querys.size(); ++i)
    {
        split_words(querys[i], this->_codetype, this->_seq_words);
        this->_seq_words_batch.push_back(this->_seq_words);
        shape += this->_seq_words.size();
        this->_lod[0].push_back(shape);
    }
    this->_input_tensor->SetLoD(this->_lod);
    this->_input_tensor->Reshape({shape, 1});

    int64_t *input_d = this->_input_tensor->mutable_data<int64_t>(this->_place);
    int index = 0;
    for (size_t i = 0; i < this->_seq_words_batch.size(); ++i)
    {
        for (size_t j = 0; j < this->_seq_words_batch[i].size(); ++j)
        {
            // normalization
            std::string word = this->_seq_words_batch[i][j];
            auto q2b_iter = this->_q2b_dict->find(word);
            if (q2b_iter != this->_q2b_dict->end())
            {
                word = q2b_iter->second;
            }

            // get word_id
            int64_t word_id = this->_oov_id;
            auto word_iter = this->_word2id_dict->find(word);
            if (word_iter != this->_word2id_dict->end())
            {
                word_id = word_iter->second;
            }
            input_d[index++] = word_id;
        }
    }
    return 0;
}

/* 对输出的标签进行解码转换为模型输出格式 */
int LAC::parse_targets(
    const std::vector<std::string> &tags,
    const std::vector<std::string> &words,
    std::vector<OutputItem> &result)
{
    result.clear();
    for (size_t i = 0; i < tags.size(); ++i)
    {
        // 若新词，则push_back一个新词，否则append到上一个词中
        if (result.empty() || tags[i].rfind("B") == tags[i].length() - 1 || tags[i].rfind("S") == tags[i].length() - 1)
        {
            OutputItem output_item;
            output_item.word = words[i];
            output_item.tag = tags[i].substr(0, tags[i].length() - 2);
            result.push_back(output_item);
        }
        else
        {
            result[result.size() - 1].word += words[i];
        }
    }
    return 0;
}

std::vector<OutputItem> LAC::run(const std::string &query)
{
    std::vector<std::string> query_vector = std::vector<std::string>({query});
    auto result = run(query_vector);
    return result[0];
}

std::vector<std::vector<OutputItem>> LAC::run(const std::vector<std::string> &querys)
{

    this->feed_data(querys);
    this->_predictor->ZeroCopyRun();

    // 对模型输出进行解码
    int output_size = 0;
    int64_t *output_d = this->_output_tensor->data<int64_t>(&(this->_place), &output_size);
    this->_labels.clear();
    this->_results_batch.clear();
    for (size_t i = 0; i < this->_lod[0].size() - 1; ++i)
    {
        for (size_t j = 0; j < _lod[0][i + 1] - _lod[0][i]; ++j)
        {

            int64_t cur_label_id = output_d[_lod[0][i] + j];
            auto it = this->_id2label_dict->find(cur_label_id);
            this->_labels.push_back(it->second);
        }

        // 装载了用户干预词典，先进行干预处理
        if (custom){
            custom->parse_customization(this->_seq_words_batch[i], this->_labels);
        }

        parse_targets(this->_labels, this->_seq_words_batch[i], this->_results);
        this->_labels.clear();
        _results_batch.push_back(this->_results);
    }

    return this->_results_batch;
}
