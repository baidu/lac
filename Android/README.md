## LAC的Android调用

Android的调用中，我们采用[NDK](https://developer.android.google.cn/ndk/)调用[C++的接口](../c++/README.md)的代码。NDK的使用可以参照[NDK官网](https://developer.android.google.cn/ndk/)或其他博客的教程，这里我们主要展示如何将我们的LAC模型集成到Android应用中

### 示例运行

#### 1. 直接安装运行

- 我们编译了一个**APK安装包**`lac_demo.apk`，可在[release界面](https://github.com/baidu/lac/releases/)下载文件、安装到Android手机中进行测试，目前该demo仅集成了armeabi-v7a或arm64-v8a的库。

#### 2. 编译APK文件

- 下载代码后，可直接使用Android Studio打开[testlac](./testlac)这个文件夹的项目，编译运行该项目即可直接生成一个Android手机的apk文件（支持armeabi-v7a或arm64-v8a）。
- 项目依赖于[NDK](https://developer.android.google.cn/ndk/)和[CMake](https://developer.android.google.cn/ndk/guides/cmake)进行编译，如果编译过程中提示NDK配置有误，可参照此处进行安装配置：
  - **安装**：依次打开Tools>SDK Manager>SDK Tools，勾选LLDB、CMake和NDK进行下载
  - **配置**：打开File > Project Structure > SDK Location，选择默认NDK的路径

### 模型集成过程

LAC模型是使用Paddle训练所得的模型，若要在移动端的调用自己训练的Paddle模型，需要进行以下两项工作

- 集成依赖库[PaddleLite](https://paddle-lite.readthedocs.io/zh/latest/index.html)，该库Paddle为移动端调用模型所定制的轻量库，并集成模型压缩和优化等相关功能
- 为了适配移动端设备，还需要使用PaddleLite的工具对模型进行优化

#### 1. 依赖库准备

`testlac`项目仅支持armeabi-v7a和arm64-v8a是因为在该项目中仅集成了armeabi-v7a和arm64-v8a的PaddleLite依赖库，即testlac项目中jniLibs中的文件。如需更多平台依赖库的支持，需要集成对应的libs文件：

- 直接下载：https://paddle-lite.readthedocs.io/zh/latest/user_guides/release_lib.html
- 自行编译：https://paddle-lite.readthedocs.io/zh/latest/user_guides/source_compile.html

下载或编译完成后，参照示例将其放于testlac项目的jniLibs文件夹，同时修改build.gradle中ndk的abiFilters选项，即可完成其他框架的支持。

> 注：PaddleLite预测库的Libs链接需要在CMakeLists.txt文件中声明，如需在自己项目中进行集成，可参考testlac项目中CMakeLists.txt文件的写法。


#### 2. 模型优化

为了适配于移动端设备，我们需要通过PaddleLite的工具对模型进行优化。具体模型优化可以参考PaddleLite官网的介绍：[模型优化工具 opt](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html)，在此我们给出一个简单的使用示例：

- 到[官网](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html)中下载优化工具opt
- 执行下列命令生成优化后的模型

  ```sh
  # valid_targets可选(arm|opencl|x86|npu|xpu)
  ./opt --model_dir=./lac_model/model \
        --valid_targets=arm \
        --optimize_out_type=naive_buffer \
        --optimize_out=lac_model_opt
  ```

  > 经过转换后的模型，对应于testlac项目assets目录下的lac_model/model.nb文件。

- 我们在[release界面](https://github.com/baidu/lac/releases/)提供了经过优化转换的LAC模型`models_android.zip`，以供大家下载使用，解压后其中包含三个不同的模型：
  - `seg_model`：仅实现分词的模型，体积小
  - `lac_model`：实现分词、词性标注、实体识别于一体的词法分析模型
  - `laclite_model`：`lac_model`的轻量化版本，效果会稍差于`lac_model`

#### 3. 其他

- 目前PaddleLite已支持直接使用Java接口调用模型，而不用通过jni的形式调用c++接口，具体可参照官网提供的[Android Demo](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/android_app_demo.html)进行修改
- PaddleLite官网也提供了[iOS Demo](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/ios_app_demo.html)，有需要的同学可参照[iOS Demo](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/ios_app_demo.html)实现在iOS上的应用