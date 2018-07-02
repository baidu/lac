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

#ifndef BAIDU_LAC_LAC_H
#define BAIDU_LAC_LAC_H
#include "main_tagger.h"
#include "customization_tagger.h"

class MainTagger;

namespace lac {

///
/// \brief The Lac class, the main controller and entry of the lac
///
class Lac {
private:
    explicit Lac();
public:
    ~Lac();
    ///
    /// \brief create, the factory function
    /// \param conf_path, path of config
    /// \return pointer of a new Lac object, NULL when failed
    ///
    static Lac* create(const char* conf_dir);

    ///
    /// \brief create_buff, initialize therad variables
    /// \param buff, pointer of struct of therad variables
    /// \return _SUCCESS or _FAILED
    ///
    void* create_buff() const;
    ///
    /// \brief reset_buff, reset therad variables
    /// \param buff, pointer of struct of therad variables
    /// \return _SUCCESS or _FAILED
    ///
    RVAL reset_buff(void* buff) const;
    ///
    /// \brief destroy_buff, destroy therad variables
    /// \param buff, pointer of struct of therad variables
    /// \return _SUCCESS or _FAILED
    ///
    void destroy_buff(void* buff) const;

    ///
    /// \brief tagging, tag lac tags using the taggers
    /// \param query, query to tag
    /// \param buff, pointer of the struct of therad variables
    /// \param results, tagged results
    /// \param max_result_num, limit of tagged results,
    ///         tagging failed when the number of results exceeds the limit
    /// \return number of tagged results, or _FAILED
    ///
    int tagging(const char* query, void* buff, tag_t* results, int max_result_num);
private:

    MainTagger *_main_tagger; /* model tagger */
    CustomizationTagger *_customization_tagger; /* customization tagger */
    std::set<std::string> _strong_punc; /* strong punctuations that cut
                                    query into sentences, like period */
    std::map<std::string, std::string> _q2b_dic; /* full-width characters
                    to half-width characters, uppercase to lower case */

    ///
    /// \brief load_q2b_dic, load the dictionary of full-width characters to half-width characters
    /// \param q2b_dic_path, path of the dictionary
    /// \return _SUCCESS or _FAILED
    ///
    RVAL load_q2b_dic(const std::string &q2b_dic_path);
    ///
    /// \brief load_strong_punc, load the dictionary of punctuations that cut query into sentences, like period
    /// \param strong_punc_path, path of the dictionary
    /// \return _SUCCESS or _FAILED
    ///
    RVAL load_strong_punc(const std::string &strong_punc_path);
    ///
    /// \brief string_normal, convert the query from a char array to a character vector,
    ///         and convert full-width characters to half-width characters
    /// \param query, query in form of char*
    /// \param norm_char_vector, converted query in form of character vector
    /// \param origin_char_offsets, the offset of each character in query in the original char array
    /// \return _SUCCESS or _FAILED
    ///
    RVAL string_normal(const char* query,
                       std::vector<std::string> &norm_char_vector,
                       std::vector<int> &origin_char_offsets) const;

    ///
    /// \brief seg_sent_iter, get next sentence of the query splited by strong punctuations
    /// \param norm_char_vector, query in the form of character vector
    /// \param start, the starting index of next sentence
    /// \return the size of gotten sentence
    ///
    int seg_sent_iter(std::vector<std::string> &norm_char_vector, int start) const;

    ///
    /// \brief merge_result, merget results of the taggers
    ///         and append to the results of previous sentences
    /// \param include results of main tagger and customization tagger
    /// \param results, the results of previous sentences
    /// \param results_num, the number of the results of previous sentences
    /// \param max_result_num, the number limit of the lac results,
    ///         tagging failed when the number of results exceeds the limit
    /// \return number of tagged results, or _FAILED
    ///
    int merge_result(lac_buff_t *buff, tag_t *results, int results_num,
                     int max_result_num) const;
};
}
#endif
