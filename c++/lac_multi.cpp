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


#include <vector>
#include <string>
#include <sys/time.h>
#include<time.h>
#include <iostream>
#include "lac.h"

using namespace std;

LAC* g_lac_handle = NULL;   // 多线程共用模型
int g_line_count = 0;
long g_usec_used = 0;

pthread_mutex_t g_mutex;

/* 计时器用于测试性能 */
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


/* 线程函数 */
void* thread_worker(void *arg) {
    
    if (g_lac_handle == NULL) {
        cerr << "creat g_lac_handle error" << endl;
        pthread_exit(NULL);
    }

    LAC lac(*g_lac_handle);

    string query;    
    timeval start;
    timeval end;
    double time_cost = 0;
    int query_num  = 0;
    while (true) {
        // 数据读取
        pthread_mutex_lock(&g_mutex);
        if (!getline(cin, query)) {
            pthread_mutex_unlock(&g_mutex);
            break;
        }
        g_line_count ++;
        pthread_mutex_unlock(&g_mutex);

        gettimeofday(&start, NULL);
        auto result = lac.run(query);
        gettimeofday(&end, NULL);
        time_cost += 1000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000;
        query_num += 1;
        
        // 打印输出结果
        pthread_mutex_lock(&g_mutex);
        for (size_t i = 0; i < result.size(); i++){
            if(result[i].tag.length() == 0){
                cout << result[i].word <<" ";
            }
            else{
                cout << result[i].word << "/" << result[i].tag << " ";
            }
        }
        cout << endl;
        pthread_mutex_unlock(&g_mutex);
    }
    cerr << "query_num: " << query_num << endl;
    cerr << "ave time per query: " << time_cost / query_num << endl;
    cerr << "qps: " << query_num * 1000 / time_cost << endl; 
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0]
                  << " + model_dir + thread_num"
                  << endl;
        exit(-1);
    }

    // 默认路径
    string model_path = argv[1];
    int thread_num = atoi(argv[2]);
    
    // 装载模型
    g_lac_handle = new LAC(model_path);

    // 启动多线程
    pthread_t ids[thread_num];
    TimeUsing t;
    for (int i = 0; i < thread_num; i++) {
        pthread_create(&ids[i], NULL, thread_worker, NULL);
    }
    for (int i = 0; i < thread_num; i++) {
        pthread_join(ids[i], NULL);
    }
    g_usec_used += t.using_time();

    double time_using = (double) g_usec_used / 1000000.0;
    cerr << "page num: " << g_line_count << endl;
    cerr << "using time: " << time_using << endl;
    cerr << "page/s : " << g_line_count / time_using << endl;
    delete g_lac_handle;
    return 0;
}
