## LAC的C++调用

C++ 的代码可用于预测使用，除了我们预先提供的模型以外，还可直接调用[Python](../README.md)代码增量训练后保存的模型

### 代码示例

```c
#include <iostream>
#include "lac.h"

// 选择不同的模型进行装载
lac = LAC(model_path = "./lac_model")
// lac = LAC(model_path = "./seg_model")

// 可选，装载干预词典
lac.load_customization("custom.txt")

// 执行并返回结果
auto lac_res = lac.run("百度是一家高科技公司");

// 打印结果
for (int i=0; i<lac_res.size(); i++)
    std::cout<<lac_res[i].word<<"\001"<<lac_res[i].tag<<" ";
```

### 编译与运行

<h4 id="依赖库准备">1. 依赖库准备</h4>

LAC是基于Paddle训练所得的模型，需依赖Paddle的预测库，可通过下载预编译好的库或进行源码编译两种形式获取

##### 直接下载

- **Windows**：可于[Paddle官网](https://www.paddlepaddle.org.cn)下载已经编译好的[Windows预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/windows_cpp_inference.html)（`fluid_inference.tgz`），选择合适版本进行下载并解压

- **Linux**：可于[Paddle官网](https://www.paddlepaddle.org.cn)下载已经编译好的[Linux预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)（`fluid_inference.tgz`），选择合适版本进行下载并解压

- **MacOS**: 可于[release界面](https://github.com/baidu/lac/releases/)下载已经编译好的Mac预测库（`paddle_library_mac.zip`），下载并解压

  > 因官网未提供预编译好的MacOS预测库，为了方便开发者使用，我们自行编译并上传该库。如有需要亦可参考源码编译的内容进行自行编译。

##### 源码编译

对于**上述预编译库无法适配**的系统，可基于源码进行编译。

下面提供了**编译预测库的脚本**，需要安装[cmake](https://cmake.org/download/)，并修改脚本中的`PADDLE_ROOT`变量为保存路径后可直接运行。如果有什么问题，可参照[Paddle官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/index_cn.html)中关于此部分内容的介绍。

```sh
# 下载源码
git clone https://github.com/PaddlePaddle/Paddle.git

# 选择其中一个稳定的分支
cd Paddle
git checkout v1.6.2

# 创建并进入build目录
mkdir build
cd build

# 编译结果保存路径，需要需改
PADDLE_ROOT=/path/of/paddle

# 编译运行
cmake -DFLUID_INFERENCE_INSTALL_DIR=$PADDLE_ROOT \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_PYTHON=OFF \
      -DWITH_MKL=OFF \
      -DWITH_GPU=OFF  \
      -DON_INFER=ON \
      ../
      
 make
 make inference_lib_dist
```

#### 2. 运行测试
> 此处介绍主要面向Mac和Linux用户，Windows系统下的编译依赖于[Visual Studio](https://visualstudio.microsoft.com/zh-hans/)【大于或等于2015版本】，可参考[Windows编译文档](../compile4windows.md)进行编译运行

我们写了一个单线程调用示例lac_demo.cpp和多线程调用示例lac_multi.cpp，下面展示了编译和运行示例的方法

##### 编译

通过[cmake](https://cmake.org/download/)去完成编译，修改下列脚本中的`PADDLE_ROOT`变量为[1.依赖库准备](#依赖库准备)中得到的预测库文件夹的路径，然后直接执行该脚本即可

```sh
# 代码下载
git clone https://github.com/baidu/lac.git
cd lac

# /path/to/paddle是第1步中获取的Paddle依赖库路径
# 即下载解压后的文件夹路径或编译产出的文件夹路径
PADDLE_ROOT=/path/of/paddle

# 编译
mkdir build
cd build
cmake -DPADDLE_ROOT=$PADDLE_ROOT \
      -DWITH_DEMO=ON \
      -DWITH_JNILIB=OFF \
      ../

make install # 编译产出在 ../output 下
```

##### 运行

- 下载模型文件：

  在[release界面](https://github.com/baidu/lac/releases/)下载模型文件`models_general.zip`，解压文件夹中包含两个模型

  - `seg_model`：仅实现分词的模型
  - `lac_model`：实现分词、词性标注、实体识别于一体的词法分析模型

- 执行demo程序：

  - `output/bin/lac_demo`是一个单线程的demo程序，其源码请参考`c++/lac_demo.cpp`
  - `output/bin/lac_multi`是一个多线程的demo程序，其源码请参考`c++/lac_multi.cpp`

```sh
# 运行测试
./lac_demo <model_dir> 
./lac_multi <model_dir> <thread_num>
# model_dir: 模型文件路径，即上述下载解压后的路径，如 "./models_general/lac_model"
# thread_num: 线程数
```

程序从标准输入逐行读取句子，然后给出句子的分析结果。

示例输入：
`百度是一家高科技公司`

示例输出：
`百度/ORG 是/v 一家/m 高科技/n 公司/n`

