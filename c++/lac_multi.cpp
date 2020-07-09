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
#include <thread>
#include <mutex>
#include <iostream>

#include "lac.h"

using namespace std;

mutex g_cin_mutex;
mutex g_cout_mutex;


/* 线程函数 */
void thread_worker(LAC& g_model) {
    // Clone model
    LAC lac(g_model);

    string query;    
    while (true) {
        // 数据读取
        g_cin_mutex.lock();
        if (!getline(cin, query)) {
            g_cin_mutex.unlock();
            break;
        }
        g_cin_mutex.unlock();

        auto result = lac.run(query);
        
        // 打印输出结果
        g_cout_mutex.lock();
        for (size_t i = 0; i < result.size(); i++){
            if(result[i].tag.length() == 0){
                cout << result[i].word <<" ";
            }
            else{
                cout << result[i].word << "/" << result[i].tag << " ";
            }
        }
        cout << endl;
        g_cout_mutex.unlock();
    }
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

    // 装载模型, 多线程共用
    LAC g_model(model_path);
    // 启动多线程
    std::vector<std::thread> threads;
    for (int i = 0; i < thread_num; i++) {
        thread th(thread_worker, ref(g_model));
        threads.push_back(move(th));
    }

    for (thread& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    return 0;
}
