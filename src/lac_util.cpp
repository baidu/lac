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

#include "lac_util.h"

namespace lac {
RVAL ul_split_tokens(const std::string &line,
                               const std::string &pattern,
                               std::vector<std::string> &tokens)
{
    if ("" == line || "" == pattern) {
        return _FAILED;
    }
    tokens.clear();
    size_t pos = 0;
    size_t size = line.size();
    for (size_t i = 0; i < size; i++)
    {
        pos = line.find(pattern, i);
        if (pos != std::string::npos)
        {
            tokens.push_back(line.substr(i, pos-i));
            i = pos + pattern.size() - 1;
        } else {
            tokens.push_back(line.substr(i));
            break;
        }
    }// end of for
    return _SUCCESS;
}

int ul_next_utf8(const unsigned char *word)
{
    if (word[0] <= u'\x7f') {
        return 1;
    }

    if (word[0] >= u'\xc0' && word[0] <= u'\xdf'
            && word[1] >= u'\x80' && word[1] <= u'\xbf') {
        return 2;
    }

    if (word[0] >= u'\xe0' && word[0] <= u'\xef'
            && word[1] >= u'\x80' && word[1] <= u'\xbf'
            && word[2] >= u'\x80' && word[2] <= u'\xbf') {
        return 3;
    }

    if (word[0] >= u'\xf0' && word[0] <= u'\xf7'
            && word[1] >= u'\x80' && word[1] <= u'\xbf'
            && word[2] >= u'\x80' && word[2] <= u'\xbf'
            && word[3] >= u'\x80' && word[3] <= u'\xbf') {
        return 4;
    }

    return _FAILED;
}
}
