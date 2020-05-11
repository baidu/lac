#include <jni.h>
#include <string>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sys/time.h>
#include <android/log.h>
#include "lac.h"
#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, "lac_demo::", __VA_ARGS__))

std::unique_ptr<LAC> uPtr_Lac;

extern "C"
JNIEXPORT void JNICALL
Java_com_example_testlac_MainActivity_initLac(JNIEnv *env, jobject thiz, jstring model_path) {
    std::unique_ptr<LAC> lac(new LAC(env->GetStringUTFChars(model_path, 0), 1));
    uPtr_Lac = std::move(lac);
}


/*
Java_com_example_testlac_MainActivity_initLac(JNIEnv *env, jobject thiz, jbyteArray buffer, jint length) {

    jbyte * lac_dict = new jbyte[length];
    env->GetByteArrayRegion(buffer, 0, length, lac_dict);
    std::unique_ptr<LAC> lac(new LAC((void*) lac_dict, length, 1));
    uPtr_Lac = std::move(lac);
} */

extern "C"
JNIEXPORT void JNICALL
Java_com_example_testlac_MainActivity_releaseLac(JNIEnv *env, jobject thiz) {
    uPtr_Lac.release();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_testlac_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "测试分词from LAC";
    return env->NewStringUTF(hello.c_str());
}
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_testlac_MainActivity_stringCutFromJNI(JNIEnv *env, jobject thiz,
                                                       jstring source_text) {
    const char *utf8 = env->GetStringUTFChars(source_text, NULL);
    std::string str_source_text = std::string(utf8);
    env->ReleaseStringUTFChars(source_text, utf8);


    auto result = uPtr_Lac->lexer(str_source_text);
    std::string output_str = "";
    for (int i=0; i<result.size(); i++) {
        if(result[i].tag.length() == 0){
            output_str += (result[i].word + " ");
        }
        else{
            output_str += (result[i].word + "\001" + result[i].tag + " ");
        }
    }
    LOGI("test_lac input: %s, output: %s", str_source_text.c_str(), output_str.c_str());

    return env->NewStringUTF(output_str.c_str());
}

