## LAC的Java调用

JAVA的接口是通过jni的形式调用[C++的接口](../c++/README.md)，除了我们预先提供的模型之外，还可直接调用[Python](../README.md)进行增量训练后保存的模型。因为使用的是[C++的接口](../c++/README.md)的jni调用，故而在使用前需要准备Paddle依赖库，并产出lacjni的动态库

#### 代码示例

```java
// 选择不同的模型进行装载
LAC lac = new LAC("lac_model");

// 可选，装载干预词典
lac.loadCustomization("custom.txt")

// 声明返回结果的变量
ArrayList<String> words = new ArrayList<>();
ArrayList<String> tags = new ArrayList<>();

// 执行并返回结果
lac.run("百度是一家高科技公司", words, tags);
System.out.println(words);
System.out.println(tags);
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

#### 2. jni编译，生成Java调用包

我们编写了jni调用c++库的接口，需要先编译生成Java可直接调用的包，通过[cmake](https://cmake.org/download/)去完成该过程。安装[cmake](https://cmake.org/download/)后，修改下列脚本中`PADDLE_ROOT`变量和`JAVA_HMOE`变量后，直接运行即可完成该过程：

- `PADDLE_ROOT`为[1.依赖库准备](#依赖库准备)中得到的依赖库文件夹的路径
- `JAVA_HMOE`为JAVA安装所在路径，该目录中应有`include/jni.h`文件

```sh
# 代码下载
git clone https://github.com/baidu/lac.git

# /path/to/paddle是第1步中获取的Paddle依赖库路径
# 即下载解压后的文件夹路径或编译产出的文件夹路径
PADDLE_ROOT=/path/of/paddle

# JAVA的HOME目录，应存在文件${JAVA_HOME}/include/jni.h
JAVA_HOME=/path/of/java

# 编译
mkdir build 
cd build
cmake -DPADDLE_ROOT=$PADDLE_ROOT \
      -DJAVA_HOME=$JAVA_HOME \
      -DWITH_JNILIB=ON \
      -DWITH_DEMO=OFF \
      ../

make install # 编译产出在 ../output/java 下
```

#### 3. 运行测试

执行完上述编译后，我们可以在`output/java`目录下查看到`com.baidu.nlp`的Java库以及调用示例LacDemo.java，我们可以直接运行LacDemo.java进行调用测试

- 下载模型文件：

  在[release界面](https://github.com/baidu/lac/releases/)下载模型文件`models_general.zip`，解压文件夹中包含两个模型

  - `seg_model`：仅实现分词的模型
  - `lac_model`：实现分词、词性标注、实体识别于一体的词法分析模型

- 运行Java的demo：

```sh
# 运行测试
javac LacDemo.java
java LacDemo <model_dir>
# model_dir: 模型文件路径，即上述下载解压后的路径，如 "./models_general/lac_model"
```

程序从标准输入逐行读取句子，然后给出句子的分析结果。

示例输入：
`百度是一家高科技公司`

示例输出：
`百度/ORG 是/v 一家/m 高科技/n 公司/n`