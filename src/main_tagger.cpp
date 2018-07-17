/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "main_tagger.h"
#include <string.h>
#include <fstream>
#include <thread>
#include "lac_util.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace lac {
MainTagger::MainTagger() {
    _scope = NULL;
    _executor = NULL;
    _place = NULL;
    _word_dic_oov = 0;
}

MainTagger::~MainTagger() {
    delete _scope;
    _scope = NULL;

    delete _executor;
    _executor = NULL;

    delete _place;
    _place = NULL;
}

MainTagger* MainTagger::create(const char* conf_dir) {
    if (conf_dir == NULL) {
        std::cerr << "conf directory for creating MainTagger is null"
                << std::endl;
        return NULL;
    }
    MainTagger* handle = new MainTagger();
    if (handle == NULL) {
        return NULL;
    }

    //add create
    std::string conf_dir_str(conf_dir);

    std::string word_dic_path = conf_dir_str + "/word.dic";
    if (handle->load_word_dic(word_dic_path) != _SUCCESS) {
        delete handle;
        return NULL;
    }

    std::string tag_dic_path = conf_dir_str + "/tag.dic";
    if (handle->load_tag_dic(tag_dic_path) != _SUCCESS) {
        delete handle;
        return NULL;
    }

    std::string model_path = conf_dir_str + "/model/";
    if (handle->init_model(model_path) != _SUCCESS) {
        delete handle;
        return NULL;
    }

    return handle;
}

RVAL MainTagger::create_buff(void *buff) const {

    if (buff == NULL) {
        std::cerr << "create_buff failed: buff is NULL" << std::endl;
        return _FAILED;
    }

    lac_buff_t *lac_buff = (lac_buff_t*)buff;

    lac_buff->copy_program = std::unique_ptr<paddle::framework::ProgramDesc>(
                new paddle::framework::ProgramDesc(*_inference_program));
    if (!(lac_buff->copy_program)){
        std::cerr << "create copy_program failed" << std::endl;
        return _FAILED;
    }

    std::thread::id tid = std::this_thread::get_id();
    lac_buff->feed_holder_name = "feed_" + paddle::string::to_string(tid);
    lac_buff->fetch_holder_name = "fetch_" + paddle::string::to_string(tid);

    lac_buff->copy_program->SetFeedHolderName(lac_buff->feed_holder_name);
    lac_buff->copy_program->SetFetchHolderName(lac_buff->fetch_holder_name);

    lac_buff->ctx = _executor->Prepare(*(lac_buff->copy_program), 0);

    return _SUCCESS;
}

RVAL MainTagger::reset_buff(void *buff) const
{
    if (buff == NULL){
        std::cerr << "reset_buff failed: buff is NULL" << std::endl;
    }

    lac_buff_t *lac_buff = (lac_buff_t*)buff;
    lac_buff->feed_targets.clear();
    lac_buff->fetch_targets.clear();

    lac_buff->word_model_input_vector.clear();
    lac_buff->model_output_vector.clear();

    return _SUCCESS;
}

void MainTagger::destroy_buff(void* buff) const {
    return;
}

RVAL MainTagger::tagging(lac_buff_t *buff, int max_result_num)
{
    if (buff == NULL || buff->main_tagger_results == NULL || max_result_num < 0) {
        std::cerr << "tagging parameter error" << std::endl;
        return _FAILED;
    }

    const std::vector<std::string> &char_vector_of_query = buff->sent_char_vector;
    const std::vector<int> &origin_char_offsets = buff->sent_offset_vector;

    std::vector<int> &word_model_input_vector = buff->word_model_input_vector;
    std::vector<int> &model_output_vector = buff->model_output_vector;

    if (extract_feature(char_vector_of_query, word_model_input_vector) < _SUCCESS) {
        std::cerr << "extract_feature failed" << std::endl;
        return _FAILED;
    }

    if (predict(word_model_input_vector, model_output_vector, buff) < _SUCCESS) {
        std::cerr << "predict failed" << std::endl;
        return _FAILED;
    }

    int result_num = adapt_result(model_output_vector,
                                  buff->main_tagger_results, max_result_num, origin_char_offsets);

    if (result_num < 0) {
        std::cerr << "adapt result failed" << std::endl;
        return _FAILED;
    }

    buff->main_tagger_result_num = result_num;

    return _SUCCESS;
}

RVAL MainTagger::load_word_dic(const std::string &word_dic_path)
{
    std::ifstream  fin;
    fin.open(word_dic_path.c_str());
    if (!fin) {
        std::cerr << "Load word dic failed ! -- " << word_dic_path
                  << " not exist" << std::endl;
    }
    std::string line;
    std::vector<std::string> v0;

    while (getline(fin, line))
    {
        if (ul_split_tokens(line, "\t", v0) < 0 || 2 > v0.size()) {
            std::cerr << "Load word dic failed ! -- format error "
                      << line << std::endl;
            return _FAILED;
        }

        _word_dic[v0[1]] = atoi(v0[0].c_str());
    }
    fin.close();

    std::map<std::string, int>::const_iterator word_dic_iter
            = _word_dic.find("OOV");

    if (word_dic_iter == _word_dic.end()) {
        _word_dic_oov = _word_dic.size();
        _word_dic["OOV"] = _word_dic_oov;
    } else {
        _word_dic_oov = word_dic_iter->second;
    }

    std::cerr << "Loaded word dic -- num(with oov) = " << _word_dic.size()
              << std::endl;
    return _SUCCESS;
}

RVAL MainTagger::load_tag_dic(const std::string &tag_dic_path)
{
    std::ifstream fin;
    fin.open(tag_dic_path.c_str());
    if (!fin) {
        std::cerr << "Load tag dic failed ! -- " << tag_dic_path
                  << " not exist" << std::endl;
        return _FAILED;
    }
    std::string line;
    std::vector<std::string> v0;

    while (getline(fin, line))
    {
        if (ul_split_tokens(line, "\t", v0) < 0 || 2 > v0.size()) {
            std::cerr << "Load tag dic failed ! -- format error "
                      << line << std::endl;
            return _FAILED;
        }

        _tag_dic[atoi(v0[0].c_str())] = v0[1];
    }
    fin.close();
    std::cerr << "Loaded tag dic -- num = " << _tag_dic.size()
              << std::endl;
    return _SUCCESS;
}

RVAL MainTagger::init_model(const std::string &model_path)
{
    paddle::framework::InitDevices(false);

    _place = new paddle::platform::CPUPlace();
    if (_place == NULL) {
        std::cerr << "create _palce failed" << std::endl;
        return _FAILED;
    }

    _executor = new paddle::framework::Executor(*_place);
    if (_executor == NULL) {
        std::cerr << "create _executor failed" << std::endl;
        return _FAILED;
    }

    _scope = new paddle::framework::Scope();
    if (_scope == NULL) {
        std::cerr << "create _scope failed" << std::endl;
        return _FAILED;
    }

    _inference_program = paddle::inference::Load(_executor, _scope, model_path);
    if (!_inference_program) {
        std::cerr << "create _inference_program failed" << std::endl;
        return _FAILED;
    }

    return _SUCCESS;
}

RVAL MainTagger::extract_feature(const std::vector<std::string> &char_vector_of_query,
        std::vector<int> &word_model_input_vector) const
{
    size_t char_index = 0;
    std::map<std::string, int>::const_iterator word_dic_iter;

    while (char_index < char_vector_of_query.size()) {
        int word_model_input = _word_dic_oov;
        std::string word_feature = char_vector_of_query[char_index];

        word_dic_iter = _word_dic.find(word_feature);
        if (word_dic_iter != _word_dic.end()) {
            word_model_input = word_dic_iter->second;
        }

        word_model_input_vector.push_back(word_model_input);

        ++char_index;
    }

    return _SUCCESS;
}

RVAL MainTagger::predict(const std::vector<int> &word_model_input_vector,
                         std::vector<int> &model_output_vector,
                         lac_buff_t *buff)
{
    paddle::framework::LoDTensor tensor_word;
    paddle::framework::LoDTensor tensor_output;

    size_t query_char_count = word_model_input_vector.size();
    paddle::framework::LoD lod{{0, query_char_count}};

    tensor_word.set_lod(lod);

    paddle::framework::DDim dims = {(long)query_char_count, 1};
    int64_t *input_ptr_word = tensor_word.mutable_data<int64_t>(dims, paddle::platform::CPUPlace());

    for (int i = 0; i < tensor_word.numel(); ++i) {
      input_ptr_word[i] = word_model_input_vector[i];
    }

    std::map<std::string, const paddle::framework::LoDTensor*> &feed_targets = buff->feed_targets;
    std::map<std::string, paddle::framework::LoDTensor*> &fetch_targets = buff->fetch_targets;

    feed_targets["word"] = &tensor_word;
    fetch_targets["crf_decoding_0.tmp_0"] = &tensor_output;

    _executor->RunPreparedContext(buff->ctx.get(), _scope, &feed_targets,
                   &fetch_targets, true, true, buff->feed_holder_name,
                   buff->fetch_holder_name);

    for (int i = 0; i < tensor_output.numel(); ++i) {
        model_output_vector.push_back(tensor_output.data<int64_t>()[i]);
    }

    return _SUCCESS;
}

int MainTagger::adapt_result(const std::vector<int> &model_output_vector,
                             tag_t *results, int max_result_num,
                             const std::vector<int> origin_char_offsets) const
{
    int result_num = 0;

    int c_off = 0;
    std::string last_type;
    std::string cur_type;
    std::string model_tag;
    std::string model_tag_pos;

    std::map<int, std::string>::const_iterator tag_dic_iter;
    for (size_t i = 0; i < model_output_vector.size(); ++i) {
        int output_i = model_output_vector[i];

        cur_type = "";
        tag_dic_iter = _tag_dic.find(output_i);
        if (tag_dic_iter != _tag_dic.end()) {
            model_tag = tag_dic_iter->second;
            cur_type = model_tag.substr(0, model_tag.size() - 2);
            model_tag_pos = model_tag.substr(model_tag.size() - 1, 1);
        }

        if (tag_dic_iter == _tag_dic.end() || cur_type != last_type
                || model_tag_pos == "B") {
            if (last_type != "") {

                if (result_num >= max_result_num) {
                    std::cerr << "error: the result num is beyond the limit" << std::endl;
                    return _FAILED;
                }

                results[result_num].type_confidence = 1;
                results[result_num].offset = origin_char_offsets[c_off];
                results[result_num].length = origin_char_offsets[i]
                        - origin_char_offsets[c_off];

                if (snprintf(results[result_num].type, LAC_TYPE_MAX_LEN, "%s",
                            last_type.c_str()) < 0) {
                    std::cerr << "copy type error" << std::endl;
                    return _FAILED;
                }

                ++ result_num;
            }

            c_off = i;

            if (tag_dic_iter == _tag_dic.end()) {
                ++c_off;
            }
        }

        if (i == model_output_vector.size() - 1) {
            if (cur_type != "") {

                if (result_num >= max_result_num) {
                    std::cerr << "error: the result num is beyond the limit" << std::endl;
                    return _FAILED;
                }

                results[result_num].type_confidence = 1;
                results[result_num].offset = origin_char_offsets[c_off];
                results[result_num].length =
                        origin_char_offsets[i + 1] - origin_char_offsets[c_off];

                if (snprintf(results[result_num].type, LAC_TYPE_MAX_LEN, "%s",
                            cur_type.c_str()) < 0) {
                    std::cerr << "copy type error" << std::endl;
                    return _FAILED;
                }

                ++ result_num;

            }

            c_off = i + 1;
        }

        last_type = cur_type;
    }

    return result_num;
}
}
