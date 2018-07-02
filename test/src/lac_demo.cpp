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

#include <map>
#include <list>
#include <vector>
#include <string>
#include <sys/time.h>
#include <iostream>
#include "stdlib.h"
#include "ilac.h"

void* g_lac_handle = NULL;
int g_line_count = 0;
long g_usec_used = 0;

pthread_mutex_t g_mutex;

class TimeUsing {
public:
    explicit TimeUsing() {
        start();
    }
    virtual ~TimeUsing() {
    }
    void start() {
        gettimeofday(&_start, NULL);
    }
    long using_time() {
        gettimeofday(&_end, NULL);
        long using_time = (long) (_end.tv_sec - _start.tv_sec) * (long) 1000000
                + (long) (_end.tv_usec - _start.tv_usec);
        return using_time;
    }
private:
    struct timeval _start;
    struct timeval _end;
};

int init_dict(const char* conf_dir) {
    g_lac_handle = lac_create(conf_dir);
    std::cerr << "create lac handle successfully" << std::endl;
    return 0;
}

int destroy_dict() {
    lac_destroy(g_lac_handle);
    return 0;
}

int tagging(int max_result_num) {
    if (g_lac_handle == NULL) {
        std::cerr << "creat g_lac_handle error" << std::endl;
        return -1;
    }

    void* lac_buff = lac_buff_create(g_lac_handle);
    if (lac_buff == NULL) {
        std::cerr << "creat lac_buff error" << std::endl;
        return -1;
    }
    std::cerr << "create lac buff successfully" << std::endl;
    std::string line;
    tag_t *results = new tag_t[max_result_num];
    while (1) {

        pthread_mutex_lock(&g_mutex);
        getline(std::cin, line);
        if (!(std::cin)) {
            pthread_mutex_unlock(&g_mutex);
            break;
        }

        g_line_count++;
        pthread_mutex_unlock(&g_mutex);

        int result_num = lac_tagging(g_lac_handle,
                lac_buff, line.c_str(), results, max_result_num);
        if (result_num < 0) {
            std::cerr << "lac tagging failed : line = " << line
                    << std::endl;
            continue;
        }

        pthread_mutex_lock(&g_mutex);

        for (int i = 0; i < result_num; i++) {
            std::string name = line.substr(results[i].offset,
                    results[i].length);
            if (i >= 1) {
                std::cout << "\t";
            }
            std::cout << name << " " << results[i].type << " "
                    << results[i].offset << " " << results[i].length;
        }
        std::cout << std::endl;

        pthread_mutex_unlock(&g_mutex);
    }
    lac_buff_destroy(g_lac_handle, lac_buff);
    delete [] results;
    return 0;
}

typedef struct {
    int max_result_num;
} thread_args;

void* thread_worker(void *arg) {
    thread_args tharg = *(thread_args*) arg;
    tagging(tharg.max_result_num);
    pthread_exit(NULL);
}

int test_main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0]
                  << " + conf_dir + max_tokens + thread_num"
                  << std::endl;
        exit(-1);
    }
    const char* conf_dir = argv[1];
    int max_result_num = atoi(argv[2]);
    int thread_num = atoi(argv[3]);

    init_dict(conf_dir);

    thread_args tharg;
    tharg.max_result_num = max_result_num;

    pthread_t ids[thread_num];

    TimeUsing t;
    for (int i = 0; i < thread_num; i++) {
        pthread_create(&ids[i], NULL, thread_worker, (void*) &tharg);
    }
    for (int i = 0; i < thread_num; i++) {
        pthread_join(ids[i], NULL);
    }
    g_usec_used += t.using_time();

    double time_using = (double) g_usec_used / 1000000.0;
    std::cerr << "page num: " << g_line_count << std::endl;
    std::cerr << "using time: " << time_using << std::endl;
    std::cerr << "page/s : " << g_line_count / time_using << std::endl;

    destroy_dict();
    return 0;
}

int main(int argc, char* argv[]) {
    test_main(argc, argv);
    return 0;
}
