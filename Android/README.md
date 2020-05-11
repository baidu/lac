## LAC的Android调用
Android的调用中，我们采用[NDK](https://developer.android.google.cn/ndk/)调用[C++的接口](../c++/README.md)的代码，其原理与[Java接口](../java/README.md)中通过jni掉用[C++的接口](../c++/README.md)的代码是一致的。NDK的使用可以参照[NDK官网](https://developer.android.google.cn/ndk/)或其他博客的教程，这里我们主要展示如何将我们的LAC模型集成到Android应用中

#### 示例运行
- 于[release界面](https://github.com/baidu/lac/releases/)下载`testlac.apk`文件可直接安装到armeabi-v7a或arm64-v8a手机中进行测试
- 可通过Android Studio打开[testlac](./testlac)这个文件夹项目，编译运行该项目即可直接生成一个在armeabi-v7a或arm64-v8a平台中运行的apk文件。
- 项目依赖于[NDK](https://developer.android.google.cn/ndk/)和[CMake](https://developer.android.google.cn/ndk/guides/cmake)进行编译，如果编译过程中提示NDK配置有误，可先查阅相关资料进行安装与配置，这里给出简单的配置步骤：
    - **安装**：依次打开Tools>SDK Manager>SDK Tools，勾选LLDB、CMake和NDK进行下载
    - **配置**：打开File > Project Structure > SDK Location，选择默认NDK的路径

### 集成过程
移动端的调用与PC端的调用在依赖库和模型格式上是存在差异:
- 移动端依赖库为[PaddleLite](https://paddle-lite.readthedocs.io/zh/latest/index.html)，是移动端调用Paddle模型的轻量级库，其中也集成模型压缩和裁减等相关功能
- 为了适配于移动端设备，我们还需要通过PaddleLite的工具对模型进行优化

#### 1. 依赖库准备
testlac的demo示例仅支持armeabi-v7a和arm64-v8a是因为在该项目中仅集成了armeabi-v7a和arm64-v8a的PaddleLite依赖库，即项目中jniLibs中的文件。如需更多平台依赖库的支持，需要集成对应的libs文件：
- 直接下载：https://paddle-lite.readthedocs.io/zh/latest/user_guides/release_lib.html
- 自行编译：https://paddle-lite.readthedocs.io/zh/latest/user_guides/source_compile.html

下载或编译完成后将其testlac项目中的jniLibs文件夹中即可，修改build.gradle中ndk的abiFilters选项即可完成其他框架的支持。
> 注：PaddleLite预测库的Libs链接需要在CMakeLists.txt文件中声明，如需在自己项目中进行集成，可参考testlac项目中CMakeLists.txt文件的写法。


#### 2. 模型优化
为了适配于移动端设备，我们需要通过PaddleLite的工具对模型进行优化

LAC优化后的模型对应于testlac项目assets目录下的lac_model/lac_model.nb文件，对比[C++的接口](../c++/README.md)调用的模型我们可以发现其实有比较大的区别，其中较为浅显直观的是模型文件从多个变为一个，编译app对模型文件的分发。

具体模型优化可以参考PaddleLite官网的介绍：[模型优化工具 opt](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html)，在此我们给出一个简单的示例：
- [官网](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html)下载优化工具opt
- 执行下列命令生成优化后的模型
```sh
# valid_targets可选(arm|opencl|x86|npu|xpu)
./opt --model_dir=./lac_model/model \
      --valid_targets=arm \
      --optimize_out_type=naive_buffer \
      --optimize_out=lac_model_opt
```

#### 3. 其他
- 目前PaddleLite已支持直接使用Java接口调用模型，而不用通过jni的形式调用c++接口，具体可参照官网提供的[Android Demo](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/android_app_demo.html)进行修改
- PaddleLite官网也提供了[iOS Demo](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/ios_app_demo.html)，有需要的同学可参照[iOS Demo](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/ios_app_demo.html)实现在iOS上的应用