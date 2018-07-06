# 中文词法分析（LAC）

本项目依赖Paddle v0.14.0版本。如果您的Paddle安装版本低于此要求，请按照[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/fluid/zh/build_and_install/index_cn.html)中的说明更新Paddle安装版本。

为了达到和机器运行环境的最佳匹配，我们建议基于源码编译安装Paddle，后文也将展开讨论一些编译安装的细节。当然，如果您发现符合机器环境的预编译版本在官网发布，也可以尝试直接选用。

需要说明的是，本文档的是基于源码编译安装流程撰写的。如果在使用Paddle预编译版本过程中存在问题，请自己动手解决，但本文档所述的一些细节，也许可以作为有用的参考信息。

## 目录

- [代码结构](#代码结构)
- [简介](#简介)
- [模型](#模型)
- [数据](#数据)
- [安装](#安装)
- [运行](#运行) 
- [定制](#定制)
- [贡献代码](#贡献代码)

## 代码结构

```text
.
├── AUTHORS              # 贡献者列表
├── CMakeLists.txt       # cmake配置文件
├── conf                 # 运行本例所需的模型及字典文件
├── data                 # 运行本例所需要的数据依赖
├── include              # 头文件
├── LICENSE              # 许可证信息
├── python               # 训练使用的python文件
├── README.md            # 本文档
├── src                  # 源码
├── technical-report     # 技术报告
└── test                 # Demo程序
```

## 简介

中文分词(Word Segmentation)是将连续的自然语言文本，切分出具有语义合理性和完整性的词汇序列的过程。因为在汉语中，词是承担语义的最基本单位，切词是文本分类、情感分析、信息检索等众多自然语言处理任务的基础。
词性标注（Part-of-speech Tagging）是为自然语言文本中的每一个词汇赋予一个词性的过程，这里的词性包括名词、动词、形容词、副词等等。
命名实体识别（Named Entity Recognition，NER）又称作“专名识别”，是指识别自然语言文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等。
我们将这三个任务统一成一个联合任务，称为词法分析任务，基于深度神经网络，利用海量标注语料进行训练，提供了一个端到端的解决方案。

我们把这个联合的中文词法分析解决方案命名为LAC。LAC既可以认为是**Lexical Analysis of Chinese**的首字母缩写，也可以认为是**LAC Analyzes Chinese**的递归缩写。

## 引用

如果您的学术工作成果中使用了LAC，请您增加下述引用（https://arxiv.org/abs/1807.01882）。我们非常欣慰LAC能够对您的学术工作带来帮助。

> @article{jiao2018LAC,
> 	title={Chinese Lexical Analysis with Deep Bi-GRU-CRF Network},
> 	author={Jiao, Zhenyu and Sun, Shuqi and Sun, Ke},
> 	journal={arXiv preprint arXiv:1807.018825},
> 	year={2018}
> }


## 模型

词法分析任务的输入是一个字符串（我们后面使用『句子』来指代它），而输出是句子中的词边界和词性、实体类别。序列标注是词法分析的经典建模方式。我们使用基于GRU的网络结构学习特征，将学习到的特征接入CRF解码层完成序列标注。CRF解码层本质上是将传统CRF中的线性模型换成了非线性神经网络，基于句子级别的似然概率，因而能够更好的解决标记偏置问题。模型要点如下，具体细节请参考`python/train.py`代码。
1. 输入采用one-hot方式表示，每个字以一个id表示
2. one-hot序列通过字表，转换为实向量表示的字向量序列；
3. 字向量序列作为双向GRU的输入，学习输入序列的特征表示，得到新的特性表示序列，我们堆叠了两层双向GRU以增加学习能力；
4. CRF以GRU学习到的特征为输入，以标记序列为监督信号，实现序列标注。

词性和专名类别标签集合如下表，其中词性标签24个（小写字母），专名类别标签4个（大写字母）。这里需要说明的是，人名、地名、机名和时间四个类别，在上表中存在两套标签（PER / LOC / ORG / TIME 和 nr / ns / nt / t），被标注为第二套标签的词，是模型判断为低置信度的人名、地名、机构名和时间词。开发者可以基于这两套标签，在四个类别的准确、召回之间做出自己的权衡。

| 标签 | 含义     | 标签 | 含义     | 标签 | 含义     | 标签 | 含义     |
| ---- | -------- | ---- | -------- | ---- | -------- | ---- | -------- |
| n    | 普通名词 | f    | 方位名词 | s    | 处所名词 | t    | 时间     |
| nr   | 人名     | ns   | 地名     | nt   | 机构名   | nw   | 作品名   |
| nz   | 其他专名 | v    | 普通动词 | vd   | 动副词   | vn   | 名动词   |
| a    | 形容词   | ad   | 副形词   | an   | 名形词   | d    | 副词     |
| m    | 数量词   | q    | 量词     | r    | 代词     | p    | 介词     |
| c    | 连词     | u    | 助词     | xc   | 其他虚词 | w    | 标点符号 |
| PER  | 人名     | LOC  | 地名     | ORG  | 机构名   | TIME | 时间     |


## 数据

训练使用的数据可以由用户根据实际的应用场景，自己组织数据。数据由两列组成，以制表符分隔，第一列是utf8编码的中文文本，第二列是对应每个字的标注，以空格分隔。我们采用IOB2标注体系，即以X-B作为类型为X的词的开始，以X-I作为类型为X的词的持续，以O表示不关注的字（实际上，在词性、专名联合标注中，不存在O）。示例如下：

```text
在抗日战争时期,朝鲜族人民先后有十几万人参加抗日战斗  p-B vn-B vn-I n-B n-I n-B n-I w-B nz-B nz-I nz-I n-B n-I d-B d-I v-B m-B m-I m-I n-B v-B v-I vn-B vn-I vn-B vn-I
```

+ 我们随同代码一并发布了完全版的模型和相关的依赖数据。但是，由于模型的训练数据过于庞大，我们没有发布训练数据，仅在`data`目录下的`train_data`和`test_data`文件中放置少数样本用以示例输入数据格式。

+ 模型依赖数据包括：
    1. 输入文本的词典，在`conf`目录下，对应`word.dic`
    2. 对输入文本中特殊字符进行转换的字典，在`conf`目录下，对应`q2b.dic`
    3. 标记标签的词典,在`conf`目录下，对应`tag.dic`

+ 在训练和预测阶段，我们都需要进行原始数据的预处理，具体处理工作包括：

    1. 从原始数据文件中抽取出句子和标签，构造句子序列和标签序列
    2. 将句子序列中的特殊字符进行转换
    3. 依据词典获取词对应的整数索引

    在训练阶段，这些工作由`python/train.py`调用`python/reader.py`完成；在预测阶段，由C++代码完成。

## 安装

### 安装Paddle

Paddle可以在符合要求的原生Linux环境或docker环境下编译，编译依赖请参考[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/fluid/zh/build_and_install/build_from_source_cn.html)。对于docker环境，我们建议基于Paddle的Dockerfile自己构建镜像。

如果Paddle官方发布了符合机器运行环境的镜像，也可以尝试直接选用，省去下文所述第一步至第四步的工作。

但是，无论是官方镜像，还是基于源码的默认编译命令，都不包含Fluid预测库部分。Fluid预测库的安装要放在单独的步骤解决（见下文第五步）。

##### 第一步，克隆Paddle代码并检出 v0.14.0

```shell
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
git checkout release/0.14.0 # Paddle正式发布后，请检出v0.14.0
```

Paddle v0.14.0是Paddle团队将要重点主推的版本。在这个截至本文档发布时，Paddle v0.14.0版还没有正式发布，目前可用的是一个release分支`release/0.14.0`。

注意，`release/0.14.0`分支当前如果开启`mkldnn`的支持，会出现Segmentation Falut。这个问题在正式发布时也许会修复。而在这之前，请在关闭`mkldnn`支持的情况下编译，具体在后文详述。

##### 第二步（可选），构建docker镜像

对于非Linux环境（macOS， Windows……），需构建Paddle的docker镜像用于Paddle的编译和运行（主要是预测部分）。当然，在Linux环境下，也可以选择构建镜像。

```shell
# 可以使用自己喜欢的Ubuntu镜像，加快下载速度
docker build -t paddle:dev --build-arg UBUNTU_MIRROR='http://mirrors.ustc.edu.cn/ubuntu/' .
```

Paddle的docker镜像依赖Ubuntu基础镜像，大量软件包基于apt-get安装，因此可以配置Ubuntu镜像加速这一过程。另外需要注意的是，GPU支持库要从`developer.download.nvidia.com`下载，但近期中国区服务器的文件Checksum出现了异常。如有遇到，可以更改Docker的DNS配置，尝试使用港澳台或者海外的DNS，以便从其他区域服务器下载相关库。

##### 第三步，编译Paddle基础库

这一步骤会产出Paddle的基础库，以及python版的wheel包。

如前所述，当前`release/v0.14.0`分支需要关闭`mkldnn`库的支持。我们直接使用`cmake`命令完成编译。

```shell
# 假设$PWD是Paddle代码所在目录
docker run -it -v $PWD:/paddle -w /paddle paddle:dev /bin/bash # 启动shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_MKLDNN=OFF -DWITH_GPU=OFF -DWITH_FLUID_ONLY=ON ..
make -j <num_cpu_cores> # 并发编译可提高速度, <num_cpu_cores>表示设置的并发编译线程数
```

编译过程中，与LAC紧密相关的几个常用参数列在下表中。`WITH_AVX`和`WITH_MKL`选项会由`cmake`根据CPU的检测结果自动设定，其余参数如果需要设为默认值以外的值，需要手工指定。具体细节可以参考`CMakeLists.txt`。

| 选项            | 说明                                                         | 默认值 |
| --------------- | ------------------------------------------------------------ | ------ |
| WITH_GPU        | 如果需要CPU环境，请设为`OFF`。本文档中，LAC是基于CPU环境编译的。 | ON     |
| WITH_FLUID_ONLY | 只编译Fluid API，建议设为`ON`。                              | OFF    |
| WITH_AVX        | 是否编译含有AVX指令集的二进制文件，较新的CPU都支持AVX指令集。 | 自动   |
| WITH_MKL        | 是否使用MKL数学库，如果为否则使用OpenBLAS。该选项与`WITH_AVX`绑定，如果`WITH_AVX`为`ON`，`WITH_MKL`也为`ON`。如果CPU支持AVX2指令集，还会引入Intel的`mkldnn`库，除非显式设定`WITH_MKLDNN`为`OFF`。 | 自动   |

##### 第四步，安装python包

```shell
# 假设$PWD是Paddle代码所在目录
pip install build/python/dist/*.whl
```

更多安装细节，如升级现有包等操作，请参考[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/fluid/zh/build_and_install/build_from_source_cn.html)。

##### 第五步，编译Fluid预测库

Fluid预测不包含在默认的官方镜像，以及默认的源码编译产出中。需要单独编译。

Paddle官方也在维护Fluid预测库的预编译包，请看[这里](http://www.paddlepaddle.org/docs/develop/documentation/fluid/zh/howto/inference/build_and_install_lib_cn.html)。如果有符合机器运行环境的预编译包，也可以尝试直接选用。

在第三步的`make`成功后，直接继续执行：

```shell
make -j <num_cpu_cores> inference_lib_dist # 并发编译可提高速度, <num_cpu_cores>表示并发编译的线程数
```

基于`cmake`直接编译时，Fluid预测库的编译产出会生成在`build/fluid_install_dir`目录。您可以把它拷贝到任何您喜欢的位置。

如果您使用的是自带基础库和Python包的Paddle官方镜像，或者由于其他原因不需要安装基础库和Python包，那么在第三步的`cmake`命令之后，直接`make -j <num_cpu_cores> inference_lib_dist`即可。

### 编译LAC

LAC依赖Paddle的Fluid预测库。编译、运行环境与Paddle相关环境保持一致，以避免各种意外的出现。LAC本身的编译操作相对简单：

```shell
git clone https://github.com/baidu/lac.git
cd lac
mkdir build
cd build
# /path/to/fluid_inference_lib是上节第五步的对应的Fluid预测库编译产出路径
# LAC的demo程序，以及依赖LAC静态库的程序，都要依赖这个路径作为动态库的搜索路径
# 如果这个路径在编译完成之后有变动，需要手工设置LD_LIBRARY_PATH环境变量
cmake -DPADDLE_ROOT=/path/to/fluid_inference_lib ..
make
make install # 编译产出在 ../output 下
```

## 运行

### 训练部分

训练过程，我们使用python实现。

1. 准备好数据和字典。请将训练数据放在一个目录下，测试数据放在另一个目录下。如果有多份不同类型的训练语料，可以在训练目录下使用不同的前缀来区分不同的训练数据，比如使用novel_xxx表示小说类的训练语料，使用news_xxx表示新闻类的训练预料，训练支持同时按照一定的比例混合输入各种不同类型的语料。
2. 查看训练支持的不同选项的含义，可以使用

    ```python
    python python/train.py -h
    ```
    查看训练脚本支持的不同选项，通过设置不同的选项，对自己的训练实现定制化。其中以下选项可能较为常用：
    ```text
    --traindata_dir           指定训练数据所在的路径
    --testdata_dir            指定验证数据所在的路径
    --model_save_dir          指定模型保存的路径
    --corpus_type_list        指定使用训练数据目录下哪些类型的语料，比如使用新闻和小说语料，可以设置为news novel
    --corpus_proportion_list  指定使用训练数据目录下每种语料的比例，与corpus_type_list中的语料类型一一对应
    ```
3. 运行命令 `python python/train.py` ，**需要注意：直接运行使用的是示例数据及默认参数，实际应用时请替换真实的标记数据并修改相应配置项。** 我们可以使用不同选项来改变训练的配置，如只使用新闻语料和标题语料，可以使用命令`python python/train.py --corpus_type_list news title --corpus_proportion_list 0.5 0.5`。

### 预测部分

预测部分基于Fluid预测库实现。

#### 数据接口说明
因为分词、词性标注和专名识别常常作为其他模块的基础依赖，因此我们提供了C语言的预测接口

```text
const int LAC_TYPE_MAX_LEN = 32;
```
词性名称、专名类别名称、定制化类别名称的最大长度限制为32。模块内置的词性名称和专名类别名称的长度都不会超过此值，定制化类别名称也不能超过词此长度，否则会导致字典加载失败。

```text
typedef struct TAG {
    int offset; /* 在输入文本中的字节偏移 */
    int length; /* 字节长度 */
    char type[LAC_TYPE_MAX_LEN]; /* 类别（词性、专名类别或定制化类别） */
    double type_confidence; /* 类别置信度 */
} tag_t;
```
输出结构，offset和length代表该词在输入query中的字节偏移和长度，type代表该词的标注类别，type_confidence代表类别置信度（目前统一为1）

#### 预测流程
1. 初始化，加载字典和模型
```text
void* lac_handle = lac_create(conf_dir);
```
2. 初始化线程变量
```text
void* lac_buff = lac_buff_create(lac_handle);
```
3. 进行预测，获取结果
```text
tag_t *results = new tag_t[max_result_num];
int result_num = lac_tagging(lac_handle,
                lac_buff, query, results, max_result_num);
```
4. 释放资源
```text
lac_destroy(lac_handle);
```

#### 示例程序

`output/demo/lac_demo`是一个多线程的demo程序，其源码请参考`test/src/lac_demo.cpp`。Demo程序的使用方式为：

```shell
./lac_demo <conf_dir> <max_tokens> <thread_num>
# conf_dir:   模型与字典的路径，随项目一同发布在conf目录中
# max_tokens: 单个句子的最大长度，单位是字符
# thread_num: 线程数
```

程序从标准输入逐行读取句子，然后给出句子的分析结果。

示例输入：

```text
2003年10月15日北京时间9时,杨利伟乘由长征二号F火箭运载的神舟V号飞船首次进入太空, 象征着中国太空事业向前迈进一大步,起到了里程碑的作用。
```

示例输出：

```text
2003年10月15日 TIME 0 17	北京 LOC 17 6	时间 n 23 6	9时 TIME 29 4	, w 33 1	杨利伟 PER 34 9	乘 v 43 3	由 p 46 3	长征二号F nz 49 13	火箭 n 62 6	运载 v 68 6	的 u 74 3	神舟V号 nz 77 10	飞船 n 87 6	首次 m 93 6	进入 v 99 6	太空 s 105 6	,  v 111 2	象征 v 113 6	着 u 119 3	中国 LOC 122 6	太空 n 128 6	事业 n 134 6	向前 d 140 6	迈进 v 146 6	一大步 m 152 9	, w 161 1	起 v 162 3	到 v 165 3	了 u 168 3	里程碑 n 171 9	的 u 180 3	作用 n 183 6	。 w 189 3
```

输出格式为：

```text
word1 type1 offset1 length1 \t word2 type2 offset2 length2 \t ... 
word:   词
type：  词性、专名类型
offset：偏移量，单位为字节
length：长度，单位为字节
```

## 定制

在模型输出的基础上，LAC还支持用户配置定制化的专名类型输出。当定制化的专名词出现在输入query中时，如果该词与原有的词法分析结果不存在边界冲突，则会用定制化专名类型替代原有的标签。
配置定制化专名的方法是修改conf/customization.dic。专名类型对应的词写在类型名称下方，专名名称形如[D:XXX]。例如：
```text
[D:season]
春天
夏天
秋天
冬天
[D:flower]
花
[D:wind]
风
```
以输入query“春天的花开秋天的风以及冬天的落阳”为例，原本输出结果为：
```text
春天 TIME 0 6   的 u 6 3    花开 v 9 6  秋天 TIME 15 6  的 u 21 3   风 n 24 3   以及 c 27 6 冬天 TIME 33 6  的 u 39 3
```
添加定制化专名之后的结果为：
```text
春天 [D:season] 0 6 的 u 6 3    花开 v 9 6  秋天 [D:season] 15 6    的 u 21 3   风 [D:wind] 24 3    以及 c 27 6 冬天 [D:season] 33 6    的 u 39 3   落阳 vn 42 6
```
可以看到，“春天”“秋天”“冬天”的类别变成了[D:season]，“风”的类别变成了[D:wind]。而定制化专名词“花”虽然出现在输入query中，但是由于它和原本的结果“花开”存在边界冲突，所以不会被识别。

## 贡献代码

我们欢迎开发者向LAC贡献代码。如果您开发了新功能，发现了bug……欢迎提交Pull request与issue到Github。
