## 工具介绍
LAC全称Lexical Analysis of Chinese，是百度自然语言处理部研发的一款联合的词法分析工具，实现中文分词、词性标注、专名识别等功能。该工具具有以下特点与优势：
- **效果好**：通过深度学习模型联合学习分词、词性标注、专名识别任务，整体效果F1值超过0.91，词性标注F1值超过0.94，专名识别F1值超过0.85，效果业内领先。
- **效率高**：精简模型参数，结合Paddle预测库的性能优化，CPU单线程性能达800QPS，效率业内领先。
- **可定制**：实现简单可控的干预机制，精准匹配用户词典对模型进行干预。词典支持长片段形式，使得干预更为精准。
- **调用便捷**：**支持一键安装**，同时提供了Python、Java和C++调用接口与调用示例，实现快速调用和集成。
- **支持移动端**: 定制超轻量级模型，体积仅为2M，主流千元手机单线程性能达200QPS，满足大多数移动端应用的需求，同等体积量级效果业内领先。

## 安装与使用
在此我们主要介绍Python安装与使用，其他语言使用：
- [C++](./c++/README.md)
- [JAVA](./java/README.md)
- [Android](./Android/README.md)

### 安装说明
代码兼容Python2/3
- 全自动安装: `pip install lac`
- 半自动下载：先下载[http://pypi.python.org/pypi/lac/](http://pypi.python.org/pypi/lac/)，解压后运行 `python setup.py install`
- 安装完成后可在命令行输入`lac`或`lac --segonly`启动服务，进行快速体验

  > 国内网络可使用百度源安装，安装速率更快：`pip install lac -i https://mirror.baidu.com/pypi/simple`

### 功能与使用
#### 分词
- 代码示例：
```python
from LAC import LAC

# 装载分词模型
lac = LAC(mode='seg')

# 单个样本输入，输入为Unicode编码的字符串
text = u"LAC是个优秀的分词工具"
seg_result = lac.run(text)

# 批量样本输入, 输入为多个句子组成的list，平均速率会更快
texts = [u"LAC是个优秀的分词工具", u"百度是一家高科技公司"]
seg_result = lac.run(texts)
```
- 输出：

```text
【单样本】：seg_result = [LAC, 是, 个, 优秀, 的, 分词, 工具]
【批量样本】：seg_result = [[LAC, 是, 个, 优秀, 的, 分词, 工具], [百度, 是, 一家, 高科技, 公司]]
```

#### 词性标注与实体识别
- 代码示例：
```python
from LAC import LAC

# 装载LAC模型
lac = LAC(mode='lac')

# 单个样本输入，输入为Unicode编码的字符串
text = u"LAC是个优秀的分词工具"
lac_result = lac.run(text)

# 批量样本输入, 输入为多个句子组成的list，平均速率更快
texts = [u"LAC是个优秀的分词工具", u"百度是一家高科技公司"]
lac_result = lac.run(texts)
```
- 输出：

>每个句子的输出其切词结果word_list以及对每个单词的标注tags_list，其格式为（word_list, tags_list)
```text
【单样本】： lac_result = ([百度, 是, 一家, 高科技, 公司], [ORG, v, m, n, n])
【批量样本】：lac_result = [
                    ([百度, 是, 一家, 高科技, 公司], [ORG, v, m, n, n]),
                    ([LAC, 是, 个, 优秀, 的, 分词, 工具], [nz, v, q, a, u, n, n])
                ]
```

词性和专名类别标签集合如下表，其中我们将最常用的4个专名类别标记为大写的形式：

| 标签 | 含义     | 标签 | 含义     | 标签 | 含义     | 标签 | 含义     |
| ---- | -------- | ---- | -------- | ---- | -------- | ---- | -------- |
| n    | 普通名词 | f    | 方位名词 | s    | 处所名词  | nw   | 作品名   |
| nz   | 其他专名 | v    | 普通动词 | vd   | 动副词   | vn   | 名动词   |
| a    | 形容词   | ad   | 副形词   | an   | 名形词   | d    | 副词     |
| m    | 数量词   | q    | 量词     | r    | 代词     | p    | 介词     |
| c    | 连词     | u    | 助词     | xc   | 其他虚词 | w    | 标点符号 |
| PER  | 人名     | LOC  | 地名     | ORG  | 机构名   | TIME | 时间     |

#### 定制化功能

在模型输出的基础上，LAC还支持用户配置定制化的切分结果和专名类型输出。当模型预测匹配到词典的中的item时，会用定制化的结果替代原有结果。为了实现更加精确的匹配，我们支持以由多个单词组成的长片段作为一个item。

我们通过装载词典文件的形式实现该功能，词典文件每行表示一个定制化的item，由一个单词或多个连续的单词组成，每个单词后使用'/'表示标签，如果没有'/'标签则会使用模型默认的标签。每个item单词数越多，干预效果会越精准。

- 词典文件示例

  > 这里仅作为示例，展现各种需求情况下的结果。后续还将开放以通配符配置词典的模式，敬请期待。
```text
春天/SEASON
花/n 开/v
秋天的风
落 阳
```
- 代码示例
```python
from LAC import LAC
lac = LAC()

# 装载干预词典, sep参数表示词典文件采用的分隔符，为None时默认使用空格或制表符'\t'
lac.load_customization('custom.txt')

# 干预后结果
custom_result = lac.run(u"春天的花开秋天的风以及冬天的落阳")
```

- 以输入“春天的花开秋天的风以及冬天的落阳”为例，原本输出结果为：
```text
春天/TIME 的/u 花开/v 秋天/TIME 的/u 风/n 以及/c 冬天/TIME 的/u 落阳/n
```
- 添加示例中的词典文件后的结果为：

```text
春天/SEASON 的/u 花/n 开/v 秋天的风/n 以及/c 冬天/TIME 的/u 落/n 阳/n
```

#### 增量训练
我们也提供了增量训练的接口，用户可以使用自己的数据，进行增量训练，首先需要将数据转换为模型输入的格式，并且所有数据文件均为"UTF-8"编码：

##### 1. 分词训练

- 数据样例

  >  与大多数开源分词数据集格式一致，使用空格作为单词切分标记，如下所示：

```text
LAC 是 个 优秀 的 分词 工具 。
百度 是 一家 高科技 公司 。
春天 的 花开 秋天 的 风 以及 冬天 的 落阳 。
```

- 代码示例

```Python
from LAC import LAC

# 选择使用分词模型
lac = LAC(mode = 'seg')

# 训练和测试数据集，格式一致
train_file = "./data/seg_train.tsv"
test_file = "./data/seg_test.tsv"
lac.train(model_save_dir='./my_seg_model/',train_data=train_file, test_data=test_file)

# 使用自己训练好的模型
my_lac = LAC(model_path='my_seg_model')
```

##### 2. 词法分析训练

- 样例数据

  > 在分词数据的基础上，每个单词以“/type”的形式标记其词性或实体类别。值得注意的是，词法分析的训练目前仅支持标签体系与我们一致的数据。后续也会开放支持新的标签体系，敬请期待。

```text
LAC/nz 是/v 个/q 优秀/a 的/u 分词/n 工具/n 。/w
百度/ORG 是/v 一家/m 高科技/n 公司/n 。/w
春天/TIME 的/u 花开/v 秋天/TIME 的/u 风/n 以及/c 冬天/TIME 的/u 落阳/n 。/w
```

- 代码示例
```Python
from LAC import LAC

# 选择使用默认的词法分析模型
lac = LAC()

# 训练和测试数据集，格式一致
train_file = "./data/lac_train.tsv"
test_file = "./data/lac_test.tsv"
lac.train(model_save_dir='./my_lac_model/',train_data=train_file, test_data=test_file)

# 使用自己训练好的模型
my_lac = LAC(model_path='my_lac_model')
```

文件结构
---

```text
.
├── python                      # Python调用的脚本
├── c++                         # C++调用的代码
├── java                        # Java调用的代码
├── Android                     # Android调用的示例
├── README.md                   # 本文件
└── CMakeList.txt               # 编译C++和Java调用的脚本
```

## 在论文中引用LAC

如果您的学术工作成果中使用了LAC，请您增加下述引用。我们非常欣慰LAC能够对您的学术工作带来帮助。

```text
@article{jiao2018LAC,
	title={Chinese Lexical Analysis with Deep Bi-GRU-CRF Network},
	author={Jiao, Zhenyu and Sun, Shuqi and Sun, Ke},
	journal={arXiv preprint arXiv:1807.01882},
	year={2018},
	url={https://arxiv.org/abs/1807.01882}
}
```

贡献代码
---
我们欢迎开发者向LAC贡献代码。如果您开发了新功能，发现了bug……欢迎提交Pull request与issue到Github。
