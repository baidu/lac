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

#ifndef BAIDU_LAC_LAC_UTIL_H
#define BAIDU_LAC_LAC_UTIL_H

#include <vector>
#include <string>
#include "lac_glb.h"

namespace lac {

///
/// \brief ul_split_tokens, split, used for loading dictionary
/// \param line, string to split
/// \param pattern, separator
/// \param tokens, the results
/// \return _SUCCESS or _FAILED
///
RVAL ul_split_tokens(const std::string &line, const std::string &pattern,
        std::vector<std::string> &tokens);

///
/// \brief ul_next_utf8, find the next character in the utf8 encoded text
/// \param word, the starting pointer of the lookup
/// \return the number of bytes of the next character
///
int ul_next_utf8(const unsigned char *word);

}

#endif


