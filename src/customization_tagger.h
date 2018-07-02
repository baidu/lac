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

#ifndef BAIDU_LAC_CUSTOMIZATION_TAGGER_H
#define BAIDU_LAC_CUSTOMIZATION_TAGGER_H
#include <map>
#include <set>
#include <vector>
#include "lac_glb.h"

namespace lac {

///
/// \brief The CustomizationTagger class, supports customization
///
class CustomizationTagger {
private:
    explicit CustomizationTagger();
public:
    ~CustomizationTagger();

    ///
    /// \brief create, the factory function
    /// \param conf_path, path of config
    /// \return a new CustomizationTagger handle, NULL when failed
    ///
    static CustomizationTagger* create(const char *conf_path);

    ///
    /// \brief create_buff, initialize therad variables associated with CustomizationTagger
    /// \param buff, struct of therad variables
    /// \return _SUCCESS or _FAILED
    ///
    RVAL create_buff(void *buff) const;

    ///
    /// \brief reset_buff, reset therad variables associated with CustomizationTagger
    /// \param buff, struct of therad variables
    /// \return _SUCCESS or _FAILED
    ///
    RVAL reset_buff(void *buff) const;

    ///
    /// \brief destroy_buff, destroy therad variables associated with CustomizationTagger
    /// \param buff, struct of therad variables
    ///
    void destroy_buff(void *buff) const;

    ///
    /// \brief tagging, tag customized lac tags
    /// \param buff, struct of therad variables, include handle of results
    /// \param max_result_num, the number limit of tagged results,
    ///         tagging failed when the number of results exceeds the limit
    /// \return _SUCCESS or _FAILED
    ///
    RVAL tagging(lac_buff_t *buff, int max_result_num) const;

    ///
    /// \brief load_customization_dic, load the customization dictionary
    /// \param customization_dic_path, path of the customization dictionary
    /// \return _SUCCESS or _FAILED
    ///
    RVAL load_customization_dic(const std::string &customization_dic_path);

    ///
    /// \brief has_customization_word, test whether there are customized words
    /// \return true or false
    ///
    bool has_customized_words() const;

private:
    std::map<std::string, std::string> _customization_dic; /* customization word to its tag */
    size_t _max_customization_word_len; /* maximum length of customization word */
};
}
#endif
