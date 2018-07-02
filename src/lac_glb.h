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

#ifndef BAIDU_LAC_LAC_GLB_H
#define BAIDU_LAC_LAC_GLB_H
#include "stdlib.h"
#include <iostream>
#include <memory>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/executor.h"

namespace lac {
#ifndef RETURN_VAL
#define RETURN_VAL
enum RVAL {
    _SUCCESS = 0, _FAILED = -1,
};
#endif

#ifndef OUTPUT_TYPE
#define OUTPUT_TYPE

const int LAC_TYPE_MAX_LEN = 32; /* maxinum number of bytes of a type name */
const int MAX_TOKEN_COUNT = 256; /* maximum number of characters of a sentence
                                    given to the taggers */

///
/// \brief tag_t, the struct of lac result
///
typedef struct TAG {
    int offset; /* byte offset in query */
    int length; /* byte length */
    char type[LAC_TYPE_MAX_LEN]; /* word type */
    double type_confidence; /* confidence of type */
} tag_t;
#endif

#ifndef LAC_BUFF
#define LAC_BUFF

///
/// \brief lac_buff_t, thread variables
///
typedef struct {
    ///
    /// \brief fluid variables
    ///
    std::unique_ptr<paddle::framework::ProgramDesc> copy_program; /* [fluid] copy of
                                                    neural network representation */
    std::unique_ptr<paddle::framework::ExecutorPrepareContext> ctx; /* [fluid]
                                                                      context */
    std::string feed_holder_name; /* [fluid] feed holder name when predict, unique
                                    in each thread */
    std::string fetch_holder_name; /* [fluid] fetch holder name when predict, unique
                                        in each thread */
    std::map<std::string, const paddle::framework::LoDTensor*> feed_targets; /*
                                                        [fluid] inputs of model */
    std::map<std::string, paddle::framework::LoDTensor*> fetch_targets; /* [fluid]
                                                                outputs of model */

    std::vector<std::string> sent_char_vector; /* sentence of character vcetor form
                                                    given to the taggers */
    std::vector<int> sent_offset_vector; /* offset of each character in sentence in
                                            the original char array */

    tag_t* main_tagger_results; /* results of main tagger */
    int main_tagger_result_num; /* number of results of main tagger */

    tag_t* customization_tagger_results; /* results of customization tagger */
    int customization_tagger_result_num; /* number of results of customization tagger */

    std::set<int> main_border_set; /* border positions of main tagger results */

    std::vector<int> word_model_input_vector; /* word feature to input into the model */
    std::vector<int> model_output_vector; /* output vector of the model */

} lac_buff_t;
#endif

}
#endif
