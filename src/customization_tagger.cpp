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

#include "customization_tagger.h"
#include <string.h>
#include <fstream>
#include <thread>
#include "lac_util.h"

namespace lac {

CustomizationTagger::CustomizationTagger() {
    _max_customization_word_len = 0;
}

CustomizationTagger::~CustomizationTagger() {
}

CustomizationTagger* CustomizationTagger::create(const char* conf_dir) {
    if (conf_dir == NULL) {
        std::cerr << "conf directory for creating CustomizationTagger is null"
                << std::endl;
        return NULL;
    }
    CustomizationTagger* handle = new CustomizationTagger();
    if (handle == NULL) {
        return NULL;
    }

    std::string conf_dir_str(conf_dir);

    std::string customization_dic_path = conf_dir_str + "/customization.dic";
    if (handle->load_customization_dic(customization_dic_path) != _SUCCESS) {
        if (handle != NULL) {
            delete handle;
        }
        return NULL;
    }

    return handle;
}

RVAL CustomizationTagger::create_buff(void *buff) const
{
    return _SUCCESS;
}

RVAL CustomizationTagger::reset_buff(void *buff) const
{
    return _SUCCESS;
}

void CustomizationTagger::destroy_buff(void* buff) const
{
    return;
}

RVAL CustomizationTagger::load_customization_dic(const std::string &customization_dic_path)
{
    std::ifstream fin;
    fin.open(customization_dic_path.c_str());
    if (!fin) {
        std::cerr << "Load customization dic failed ! -- " << customization_dic_path
                  << " not exist" << std::endl;
        return _FAILED;
    }
    std::string line;
    std::vector<std::string> v0;

    _max_customization_word_len = 0;
    std::string customization_type = "";
    while (getline(fin, line))
    {
        if (ul_split_tokens(line, "\t", v0) < _SUCCESS) {
            std::cerr << "Load customization dic failed ! -- format error "
                      << line << std::endl;
            return _FAILED;
        }

        if (v0[0].size() > 4 && v0[0].substr(0, 3) == "[D:"
                && v0[0][v0[0].size() - 1] == ']') {
            if (v0[0].size() > LAC_TYPE_MAX_LEN) {
                std::cerr << "customization type " << v0[0] << " length "
                          << v0[0].size() << " > " << LAC_TYPE_MAX_LEN
                          << " ! -- type length error" << std::endl;
                return _FAILED;
            }

            customization_type = v0[0];
        } else if (customization_type.size() > 4) {
            _customization_dic[v0[0]] = customization_type;
            if (v0[0].size() > _max_customization_word_len) {
                _max_customization_word_len = v0[0].size();
            }
        }

    }
    fin.close();
    std::cerr << "Loaded customization dic -- num = " << _customization_dic.size()
              << std::endl;
    return _SUCCESS;
}

bool CustomizationTagger::has_customized_words() const
{
    return !(_customization_dic.empty());
}

RVAL CustomizationTagger::tagging(lac_buff_t *buff,
                                            int max_result_num) const
{
    if (buff == NULL || buff->customization_tagger_results == NULL || max_result_num <= 0) {
        std::cerr << "tagging parameter error" << std::endl;
        return _FAILED;
    }

    const std::vector<std::string> &char_vector_of_query = buff->sent_char_vector;
    const std::vector<int> &origin_char_offsets = buff->sent_offset_vector;
    tag_t *results = buff->customization_tagger_results;

    int result_num = 0;
    size_t char_index = 0;

    while (char_index < char_vector_of_query.size()) {
        size_t customization_word_size = 0;
        size_t tmp_len = char_vector_of_query[char_index].size();
        std::vector<std::string> tmp_customization_word_vector;
        std::string tmp_customization_word = "";
        while (tmp_len <= _max_customization_word_len) {
            tmp_customization_word +=
                    char_vector_of_query[char_index + customization_word_size];
            tmp_customization_word_vector.push_back(tmp_customization_word);

            ++customization_word_size;
            if (char_index + customization_word_size >= char_vector_of_query.size()) {
                break;
            }
            tmp_len += char_vector_of_query[char_index + customization_word_size].size();
        }

        for (; customization_word_size > 0; --customization_word_size) {
            std::string customization_word =
                    tmp_customization_word_vector[customization_word_size - 1];

            auto customization_dic_iter = _customization_dic.find(customization_word);
            if (customization_dic_iter != _customization_dic.end()) {

                results[result_num].type_confidence = 1;
                results[result_num].offset = origin_char_offsets[char_index];
                results[result_num].length =
                        origin_char_offsets[char_index + customization_word_size]
                        - origin_char_offsets[char_index];

                if (snprintf(results[result_num].type, LAC_TYPE_MAX_LEN, "%s",
                             (customization_dic_iter->second).c_str()) < 0) {
                    std::cerr << "copy type error" << std::endl;
                    return _FAILED;
                }

                ++result_num;

                char_index += customization_word_size;
                break;
            }
        }

        if (customization_word_size == 0) {
            ++char_index;
        }
    }

    buff->customization_tagger_result_num = result_num;

    return _SUCCESS;
}

}
