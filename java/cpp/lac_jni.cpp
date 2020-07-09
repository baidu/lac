#include "lac.h"
#include "lac_jni.h"
#include <string>
#include <cstring>
#include <iostream>

#ifdef __cplusplus
extern "C"
{
#endif
 
  // 返回Java类别中的self_ptr地址，用于指向创建的LAC对象
  static jfieldID _get_self_id(JNIEnv *env, jobject thisObj)
  {
    static int init = 0;
    static jfieldID fidSelfPtr;
    if (!init)
    {
      jclass thisClass = env->GetObjectClass(thisObj);
      fidSelfPtr = env->GetFieldID(thisClass, "self_ptr", "J");
    }
    return fidSelfPtr;
  }

  // 设置self_ptr地址，指向创建的LAC对象
  static void _set_self(JNIEnv *env, jobject thisObj, LAC *self)
  {
    jlong selfPtr = *(jlong *)&self;
    env->SetLongField(thisObj, _get_self_id(env, thisObj), selfPtr);
  }

  // 返回LAC对象的指针
  static LAC *_get_self(JNIEnv *env, jobject thisObj)
  {
    jlong selfPtr = env->GetLongField(thisObj, _get_self_id(env, thisObj));
    return *(LAC **)&selfPtr;
  }


  /*
 * Class:     LAC
 * Method:    init
 * Signature: (Ljava/lang/String;)V
 */
  JNIEXPORT void JNICALL Java_com_baidu_nlp_LAC_init(JNIEnv *env, jobject thisObj, jstring model_dir)
  {

    LAC *self = new LAC(env->GetStringUTFChars(model_dir, 0));
    _set_self(env, thisObj, self);
  }

  /*
 * Class:     LAC
 * Method:    copy
 * Signature: (Jlong)V
 */
  JNIEXPORT void JNICALL Java_com_baidu_nlp_LAC_copy(JNIEnv *env, jobject thisObj, jlong selfPtr)
  {
    LAC *self  = *(LAC **)&selfPtr;
    if (self){
      _set_self(env, thisObj, new LAC(*self));
    }
  }

  /*
 * Class:     LAC
 * Method:    release
 * Signature: (Jlong)V
 */
  JNIEXPORT void JNICALL Java_com_baidu_nlp_LAC_release(JNIEnv *env, jobject thisObj, jlong selfPtr)
  {
    if (selfPtr){
      delete (LAC *)selfPtr;
    }
  }

  /*
 * Class:     LAC
 * Method:    load_customization
 * Signature: (Ljava/lang/String;)Jint
 */
JNIEXPORT jint JNICALL Java_com_baidu_nlp_LAC_loadCustomization
  (JNIEnv *env, jobject thisObj, jstring dict_path)
  {
    LAC *self = _get_self(env, thisObj);
    return self->load_customization(env->GetStringUTFChars(dict_path, 0));
  }

/*
 * Class:     LAC
 * Method:    run
 * Signature: (Ljava/lang/String;Ljava/lang/ArrayList;Ljava/lang/ArrayList)Jint
 */
JNIEXPORT jint JNICALL Java_com_baidu_nlp_LAC_run
  (JNIEnv *env, jobject thisObj, jstring sentence, jobject words, jobject tags)
  {
    jclass list_jcs = env->FindClass("java/util/ArrayList");
    if (list_jcs == NULL) {
        std::cerr<<"JNICALL Java_com_baidu_nlp_LAC_run: ArrayList Not Found"<<std::endl;
        return -1;
    }
    
    //获取ArrayList构造函数id，用于Return使用
    // jmethodID list_init = env->GetMethodID(list_jcs, "<init>", "()V");
    //创建一个ArrayList对象
    // jobject list_obj = env->NewObject(list_jcs, list_init, "");

    //获取ArrayList对象的add()的methodID
    jmethodID list_add = env->GetMethodID(list_jcs, "add",
              "(Ljava/lang/Object;)Z");

    //获取ArrayList对象的clear()的methodID
    jmethodID list_clear = env->GetMethodID(list_jcs, "clear", "()V");
    env->CallVoidMethod(words, list_clear);
    env->CallVoidMethod(tags, list_clear);

    LAC *self = _get_self(env, thisObj);
    std::string input = env->GetStringUTFChars(sentence, 0);
    auto result = self->run(input);

    for (size_t i = 0; i < result.size(); i++)
    {
      env->CallBooleanMethod(words, list_add, env->NewStringUTF(result[i].word.c_str()));
      env->CallBooleanMethod(tags, list_add, env->NewStringUTF(result[i].tag.c_str()));
    }
  
    return 0;

  }

#ifdef __cplusplus
}
#endif