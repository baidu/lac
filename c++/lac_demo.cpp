/* Copyright (c) 2020 Baidu, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <ctime>
#include <sys/time.h>
#include <iostream>
#include "lac.h"

using namespace std;

int main(int argc, char* argv[]){
    // 读取命令行参数
    string model_path = "lac_model";
    string dict_path = "";
    if (argc > 1){
        model_path = argv[1];
    }
    if (argc > 2){
        dict_path = argv[2];
    }

    // 装载模型和用户词典
    LAC lac(model_path);
    if (dict_path.length() > 1){
        lac.load_customization(dict_path);
    }

    string query;

    // 计时器用于测试性能
    struct timeval start;
    struct timeval end;
    int cnt = 0;
    int char_cnt = 0;
    gettimeofday(&start, NULL);
    
    while (true)
    {
        if(!getline(cin, query)){
            break;
        }
        cnt ++;
        char_cnt += query.length();

        // 执行与打印输出结果
        auto result = lac.run(query);
        for (size_t i = 0; i < result.size(); i ++){
            if(result[i].tag.length() == 0){
                cout << result[i].word << " ";
            }else{
                cout << result[i].word << "/" << result[i].tag << " ";
            }
        }
        cout << endl;
    }
    gettimeofday(&end, NULL);
    double time = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec)/1000000.0;
    cerr << "using time: " << time << " \t qps:" << cnt/time << "\tc/s:" << char_cnt/time << endl;

}
