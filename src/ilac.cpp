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

#include "ilac.h"
#include "lac.h"
#include "lac_glb.h"

using lac::lac_buff_t;
using lac::Lac;
using lac::_FAILED;

void* lac_create(const char* conf_dir) {
    Lac* handle = Lac::create(conf_dir);
    if (handle == NULL) {
        std::cerr << "create lac handle failed" << std::endl;
        return NULL;
    }
    return (void*) handle;
}
void lac_destroy(void* lac_handle) {
    if (lac_handle != NULL) {
        delete (Lac*) lac_handle;
    }

    return;
}
void* lac_buff_create(void* lac_handle) {
    if (lac_handle == NULL) {
        std::cerr << "lac_buff_create: lac_handle is null"
                << std::endl;
        return NULL;

    }
    void *lac_buff = ((Lac*) lac_handle)->create_buff();

    return lac_buff;
}

void lac_buff_destroy(void* lac_handle, void* lac_buff) {
    if (lac_handle == NULL && lac_buff != NULL) {
        ((Lac*) lac_handle)->destroy_buff(lac_buff);
    }
}

int lac_tagging(void* lac_handle, void* lac_buff,
        const char* query, tag_t* results, int max_result_num) {
    if (lac_handle == NULL) {
        std::cerr << "lac_tagging: lac_handle is null" << std::endl;
        return _FAILED;
    }

    int result_num = ((Lac*) lac_handle)->tagging(query,
            lac_buff, (lac::tag_t *)results, max_result_num);

    return result_num;
}
