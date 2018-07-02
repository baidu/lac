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

#include "lac.h"
#include "lac_util.h"
#include <string>
#include <fstream>

namespace lac {

Lac::Lac() {
    _main_tagger = NULL;
    _customization_tagger = NULL;
}

Lac::~Lac() {
    if (_main_tagger != NULL) {
        delete _main_tagger;
        _main_tagger = NULL;
    }

    if (_customization_tagger != NULL) {
        delete _customization_tagger;
        _customization_tagger = NULL;
    }
}

Lac* Lac::create(const char* conf_dir) {
    if (conf_dir == NULL) {
        std::cerr << "conf directory for creating lac is null"
                << std::endl;
        return NULL;
    }

    Lac* handle = new Lac();
    if (handle == NULL) {
        return NULL;
    }

    std::string q2b_dic_path = std::string(conf_dir) + "/q2b.dic";
    if (handle->load_q2b_dic(q2b_dic_path) != _SUCCESS) {
        if (handle != NULL) {
            delete handle;
        }
        return NULL;
    }

    std::string strong_punc_path = std::string(conf_dir) + "/strong_punc.dic";
    if (handle->load_strong_punc(strong_punc_path) != _SUCCESS) {
        if (handle != NULL) {
            delete handle;
        }
        return NULL;
    }
    handle->_strong_punc.insert("\n");
    handle->_strong_punc.insert("\r");

    handle->_main_tagger = MainTagger::create(conf_dir);
    if (handle->_main_tagger == NULL) {
        std::cerr << "in lac: create _main_tagger error" << std::endl;
        delete handle;
        return NULL;
    }

    handle->_customization_tagger = CustomizationTagger::create(conf_dir);
    if (handle->_customization_tagger == NULL) {
        std::cerr << "in lac: create _customization_tagger error" << std::endl;
        delete handle;
        return NULL;
    }

    return handle;
}

void* Lac::create_buff() const {
    lac_buff_t* buff = new lac_buff_t();
    if (buff == NULL) {
        std::cerr << "create lac_buff_t object failed" << std::endl;
        return NULL;
    }

    buff->main_tagger_results = new tag_t[MAX_TOKEN_COUNT];
    buff->main_tagger_result_num = 0;

    if (buff->main_tagger_results == NULL) {
        std::cerr << "create main_tagger_results buffer failed" << std::endl;
        destroy_buff(buff);
        delete buff;
        return NULL;
    }

    buff->customization_tagger_results = new tag_t[MAX_TOKEN_COUNT];
    buff->customization_tagger_result_num = 0;

    if (buff->customization_tagger_results == NULL) {
        std::cerr << "create customization_tagger_results biffer failed" << std::endl;
        destroy_buff(buff);
        delete buff;
        return NULL;
    }

    if (NULL != _main_tagger)
    {
        if (_main_tagger->create_buff(buff) < _SUCCESS)
        {
            std::cerr << "error: create _main_tagger buff error!" << std::endl;
            destroy_buff(buff);
            delete buff;
            return NULL;
        }
    }

    if (NULL != _customization_tagger)
    {
        if (_customization_tagger->create_buff(buff) < _SUCCESS)
        {
            std::cerr << "error: create _customization_tagger buff error!" << std::endl;
            destroy_buff(buff);
            delete buff;
            return NULL;
        }
    }

    return buff;
}

void Lac::destroy_buff(void* buff) const
{
    if (buff != NULL) {
        if (_main_tagger != NULL) {
            _main_tagger->destroy_buff(buff);
        }

        if (_customization_tagger != NULL) {
            _customization_tagger->destroy_buff(buff);
        }

        lac_buff_t* lac_buff = (lac_buff_t*) buff;

        if (lac_buff->main_tagger_results != NULL) {
            delete[] lac_buff->main_tagger_results;
            lac_buff->main_tagger_results = NULL;
        }

        if (lac_buff->customization_tagger_results != NULL) {
            delete[] lac_buff->customization_tagger_results;
            lac_buff->customization_tagger_results = NULL;
        }

        lac_buff->main_tagger_result_num = 0;
        lac_buff->customization_tagger_result_num = 0;
    }

    return;
}
RVAL Lac::reset_buff(void* buff) const
{
    if (buff == NULL) {
        std::cerr << "buff is null" << std::endl;
        return _FAILED;
    }

    if (_main_tagger != NULL) {
        if (_main_tagger->reset_buff(buff) < _SUCCESS) {
            std::cerr << "_main_tagger reset buff failed" << std::endl;
            return _FAILED;
        }
    }

    if (_customization_tagger != NULL) {
        if (_customization_tagger->reset_buff(buff) < _SUCCESS) {
            std::cerr << "_customization_tagger reset buff failed" << std::endl;
            return _FAILED;
        }
    }

    lac_buff_t *lac_buff = (lac_buff_t*)buff;
    lac_buff->sent_char_vector.clear();
    lac_buff->sent_offset_vector.clear();

    lac_buff->main_tagger_result_num = 0;
    lac_buff->customization_tagger_result_num = 0;

    lac_buff->main_border_set.clear();

    return _SUCCESS;
}

int Lac::tagging(const char* query, void* buff, tag_t* results,
        int max_result_num) {
    if (query == NULL || buff == NULL || results == NULL) {
        std::cerr << "query, lac buff or results is NULL" << std::endl;
        return _FAILED;
    }

    if (max_result_num <= 0) {
        std::cerr << "result_num is not positive" << std::endl;
        return _FAILED;
    }

    std::vector<std::string> norm_char_vector;
    std::vector<int> origin_char_offsets;

    if (string_normal(query, norm_char_vector, origin_char_offsets) != _SUCCESS) {
        std::cerr << "query normalize failed" << std::endl;
        return _FAILED;
    }

    int start = 0;
    int tcnt = 0;
    int results_num = 0;
    while ((tcnt = seg_sent_iter(norm_char_vector, start)) > 0) {
        reset_buff(buff);
        lac_buff_t* lac_buff = (lac_buff_t*) buff;
        lac_buff->sent_char_vector.assign(norm_char_vector.begin() + start,
                                            norm_char_vector.begin() + start + tcnt);
        lac_buff->sent_offset_vector.assign(origin_char_offsets.begin() + start,
                                            origin_char_offsets.begin() + start + tcnt + 1);

        if (_main_tagger) {
            if (_main_tagger->tagging(lac_buff, max_result_num) != _SUCCESS) {
                std::cerr << "_main_tagger tagging failed" << std::endl;
                return _FAILED;
            }
        }

        if (_customization_tagger && _customization_tagger->has_customized_words()) {
            if (_customization_tagger->tagging(lac_buff, max_result_num) != _SUCCESS) {
                std::cerr << "_customization_tagger tagging failed" << std::endl;
                return _FAILED;
            }
        } else {
            lac_buff->customization_tagger_result_num = 0;
        }

        results_num = merge_result(lac_buff, results, results_num, max_result_num);

        if (results_num < 0 || results_num > max_result_num) {
            std::cerr << "merge failed" << std::endl;
            return _FAILED;
        }

        start += tcnt;
    }

    return results_num;
}

RVAL Lac::load_q2b_dic(const std::string &q2b_dic_path)
{
    std::ifstream  fin;
    fin.open(q2b_dic_path.c_str());
    if (!fin) {
        std::cerr << "Load q2b dic failed ! -- " << q2b_dic_path
                  << " not exist" << std::endl;
        return _FAILED;
    }
    std::string line;
    std::vector<std::string> v0;

    while (getline(fin, line))
    {
        if (ul_split_tokens(line, "\t", v0) < 0 || 2 > v0.size()) {
            std::cerr << "Load q2b dic failed ! -- format error "
                      << line << std::endl;
            return _FAILED;
        }

        _q2b_dic[v0[0]] = v0[1];
    }
    fin.close();
    std::cerr << "Loaded q2b dic -- num = " << _q2b_dic.size()
              << std::endl;
    return _SUCCESS;
}

RVAL Lac::load_strong_punc(const std::string &strong_punc_path)
{
    std::ifstream  fin;
    fin.open(strong_punc_path.c_str());
    if (!fin) {
        std::cerr << "Load strong punc failed ! -- " << strong_punc_path
                  << " not exist" << std::endl;
        return _FAILED;
    }

    std::string line;

    while (getline(fin, line))
    {
        if (line.size() <= 0) {
            std::cerr << "Load strong punc failed ! -- blank line " << std::endl;
            return _FAILED;
        }

        _strong_punc.insert(line);
    }
    fin.close();
    std::cerr << "Loaded strong punc -- num = " << _strong_punc.size()
              << std::endl;
    return _SUCCESS;

}

RVAL Lac::string_normal(const char *query,
                              std::vector<std::string> &norm_char_vector,
                              std::vector<int> &origin_char_offsets) const
{
    if (query == NULL) {
        return _FAILED;
    }

    int query_index = 0;
    int query_len = strlen(query);
    std::map<std::string, std::string>::const_iterator q2b_iter;

    while (query_index < query_len) {
        origin_char_offsets.push_back(query_index);
        int letter_len = ul_next_utf8((const unsigned char *)(query + query_index));
        if (letter_len <= 0) {
            std::cerr << "invalid char at position " << query_index
                      << ". Query must be encoded in UTF-8. query = " << query << std::endl;
            ++query_index;
            continue;
        }

        std::string letter(query + query_index, letter_len);

        q2b_iter = _q2b_dic.find(letter);
        if (q2b_iter != _q2b_dic.end()) {
            letter = q2b_iter->second;
        }

        norm_char_vector.push_back(letter);

        query_index += letter_len;
    }

    origin_char_offsets.push_back(query_len);

    return _SUCCESS;
}

int Lac::seg_sent_iter(std::vector<std::string> &norm_char_vector, int start) const
{
    int tpos = start;
    int char_count = norm_char_vector.size();

    if (start < 0 || start >= char_count) {
        return 0;
    }

    while (tpos < char_count && tpos - start < MAX_TOKEN_COUNT) {
        if (_strong_punc.find(norm_char_vector[tpos]) != _strong_punc.end()) {
            ++tpos;
            break;
        }

        ++tpos;
    }

    return tpos - start;
}

int Lac::merge_result(lac_buff_t *buff, tag_t *results, int results_num,
                            int max_result_num) const
{
    if (buff->main_tagger_results == NULL || buff->main_tagger_result_num < 0 ||
            buff->customization_tagger_results == NULL ||
            buff->customization_tagger_result_num < 0 ||
            results == NULL || results_num < 0 || max_result_num < 0) {
        std::cerr << "merge_result parameters error" << std::endl;
        return _FAILED;
    }

    tag_t *main_tagger_results = buff->main_tagger_results;
    int main_tagger_result_num = buff->main_tagger_result_num;
    tag_t *customization_tagger_results = buff->customization_tagger_results;
    int customization_tagger_result_num = buff->customization_tagger_result_num;

    int main_i = 0;

    if (customization_tagger_result_num > 0){
        std::set<int> &main_border_set = buff->main_border_set;
        for (int i = 0; i < main_tagger_result_num; ++i) {
            main_border_set.insert(main_tagger_results[i].offset);
        }
        main_border_set.insert(main_tagger_results[main_tagger_result_num - 1].offset
                + main_tagger_results[main_tagger_result_num - 1].length);


        for (int customization_i = 0; customization_i < customization_tagger_result_num;
            ++customization_i) {

            int begin = customization_tagger_results[customization_i].offset;
            int end = begin + customization_tagger_results[customization_i].length;

            if (main_border_set.find(begin) != main_border_set.end()
                    && main_border_set.find(end) != main_border_set.end()) {

                while (main_tagger_results[main_i].offset < begin) {

                    if (results_num >= max_result_num) {
                        std::cerr << "merge result failed: the result num is beyond the limit"
                                  << std::endl;
                        return _FAILED;
                    }

                    results[results_num] = main_tagger_results[main_i];
                    ++results_num;
                    ++main_i;
                }

                if (results_num >= max_result_num) {
                    std::cerr << "merge result failed: the result num is beyond the limit"
                              << std::endl;
                    return _FAILED;
                }

                results[results_num] = customization_tagger_results[customization_i];
                ++results_num;

                while (main_tagger_results[main_i].offset < end) {
                    ++main_i;
                }
            }
        }
    }

    while (main_i < main_tagger_result_num) {

        if (results_num >= max_result_num) {
            std::cerr << "merge result failed: the result num is beyond the limit" << std::endl;
            return _FAILED;
        }

        results[results_num] = main_tagger_results[main_i];
        ++results_num;
        ++main_i;
    }

    return results_num;
}
}
