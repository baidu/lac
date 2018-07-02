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

#ifndef BAIDU_LAC_MAIN_TAGGER_H
#define BAIDU_LAC_MAIN_TAGGER_H
#include <map>
#include <set>
#include <vector>
#include "lac_glb.h"
#include "paddle/fluid/framework/init.h"
#include "paddle/fluid/inference/io.h"

namespace lac {

///
/// \brief The MainTagger class, the tagger of model
///
class MainTagger {
private:
    explicit MainTagger();
public:
    ~MainTagger();

    ///
    /// \brief create, the factory function
    /// \param conf_path, path of config
    /// \return pointer of a new MainTagger object, NULL when failed
    ///
    static MainTagger* create(const char *conf_path);

    ///
    /// \brief create_buff, initialize therad variables associated with MainTagger
    /// \param buff, pointer of struct of therad variables
    /// \return _SUCCESS or _FAILED
    ///
    RVAL create_buff(void *buff) const;

    ///
    /// \brief reset_buff, reset therad variables associated with MainTagger
    /// \param buff, pointer of struct of therad variables
    /// \return _SUCCESS or _FAILED
    ///
    RVAL reset_buff(void *buff) const;

    ///
    /// \brief destroy_buff, destroy therad variables associated with CustomizationTagger
    /// \param buff, pointer of struct of therad variables
    ///
    void destroy_buff(void *buff) const;

    ///
    /// \brief tagging, tag lac tags using model
    /// \param buff, pointer of struct of therad variables
    /// \param max_result_num, limit of tagged results,
    ///         tagging failed when the number of results exceeds the limit
    /// \return number of tagged results, or _FAILED
    ///
    RVAL tagging(lac_buff_t *buff, int max_result_num);

private:
    std::map<std::string, int> _word_dic; /* character to its index in model */
    int _word_dic_oov; /* oov index in model */
    std::map<int, std::string> _tag_dic; /* tag index to tag */

    std::unique_ptr<paddle::framework::ProgramDesc> _inference_program; /*
                                    [fluid] neural network representation */
    paddle::framework::Executor *_executor; /* [fluid] stateless executor, only
                                                related to the device */
    paddle::framework::Scope *_scope; /* [fluid]  record the variables */
    paddle::platform::CPUPlace *_place; /* [fluid] device */

    ///
    /// \brief load_word_dic, load word dictionary
    /// \param word_dic_path, path of word dictionary
    /// \return _SUCCESS or _FAILED
    ///
    RVAL load_word_dic(const std::string &word_dic_path);

    ///
    /// \brief load_tag_dic, load tag dictionary
    /// \param tag_dic_path, path of tag dictionary
    /// \return _SUCCESS or _FAILED
    ///
    RVAL load_tag_dic(const std::string &tag_dic_path);

    ///
    /// \brief init_model, initialize fluid and load model
    /// \param model_path, path of model
    /// \return _SUCCESS or _FAILED
    ///
    RVAL init_model(const std::string &model_path);

    ///
    /// \brief extract_feature, extract model input feature from query
    /// \param char_vector_of_query, query in the form of character vector
    /// \param word_model_input_vector, the extracted character feature vector to be model input
    /// \return _SUCCESS or _FAILED
    ///
    RVAL extract_feature(const std::vector<std::string> &char_vector_of_query,
                         std::vector<int> &word_model_input_vector) const;

    ///
    /// \brief predict, predict with the model
    /// \param word_model_input_vector, the character feature vector of the model input
    /// \param model_output_vector, the output vector of the model
    /// \param buff, pointer of struct of therad variables
    /// \return _SUCCESS or _FAILED
    ///
    RVAL predict(const std::vector<int> &word_model_input_vector,
                 std::vector<int> &model_output_vector, lac_buff_t *buff);

    ///
    /// \brief adapt_result, convert the model output to lac results
    /// \param model_output_vector, the output vector of the model
    /// \param results, the lac output
    /// \param max_result_num, the number limit of the lac results,
    ///         tagging failed when the number of results exceeds the limit
    /// \param origin_char_offsets, The offset of each character in query in the original char array
    /// \return number of lac results, or _FAILED
    ///
    int adapt_result(const std::vector<int> &model_output_vector, tag_t *results,
                     int max_result_num, const std::vector<int> origin_char_offsets) const;
};
}
#endif
