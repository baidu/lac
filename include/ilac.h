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

#ifndef BAIDU_LAC_ILAC_H
#define BAIDU_LAC_ILAC_H

#ifndef I_OUTPUT_TYPE
#define I_OUTPUT_TYPE

const int LAC_TYPE_MAX_LEN = 32; /* the maxinum number of bytes of a type name */
///
/// \brief tag_t, struct of lac result
///
typedef struct TAG {
    int offset; /* byte offset in query */
    int length; /* byte length */
    char type[LAC_TYPE_MAX_LEN]; /* word type */
    double type_confidence; /* confidence of type */
} tag_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

///
/// \brief lac_create, create an Lac handle
/// \param conf_path, path of config
/// \return a new Lac handle, NULL when failed
///
void* lac_create(const char* conf_dir);

///
/// \brief lac_destroy, destroy an Lac handle
/// \param lac_handle, the Lac handle to destroy
///
void lac_destroy(void* lac_handle);

///
/// \brief lac_buff_create, create and initialize thread variables
/// \param lac_handle, the Lac handle
/// \return the created struct of thread variables
///
void* lac_buff_create(void* lac_handle);

///
/// \brief lac_buff_destroy, destroy thread variables
/// \param lac_handle, the Lac handle
/// \param lac_buff, the struct of thread variables to destroy
///
void lac_buff_destroy(void* lac_handle, void* lac_buff);

///
/// \brief lac_tagging, tag the query
/// \param lac_buff, the struct of thread variables
/// \param query, query to tag
/// \param results, tagged results
/// \param max_result_num, limit of tagged results,
///     tagging failed when the number of results exceeds the limit
/// \return number of tagged results, or _FAILED
///
int lac_tagging(void* lac_handle, void* lac_buff,
    const char* query, tag_t* results, int max_result_num);

#ifdef __cplusplus
}
#endif

#endif
