package com.baidu.nlp;

import java.util.ArrayList;

public class LAC {
    static {
        // 装载链接库，需要lac_jni的链接库文件放到当前目录下
        System.setProperty("java.library.path", ".");
        System.loadLibrary("lacjni");
    }

    // 用于指向创建的LAC对象
    private long self_ptr;

    public LAC(String model_dir) {
        System.loadLibrary("lacjni");
        init(model_dir);
    }

    // 装载model_path路径的模型
    private native void init(String model_path);

    // 装载dict_path路径的词典
    public native int loadCustomization(String dict_path);

    // 运行LAC，并将结果返回到words和tags中
    public native int run(String sentence, ArrayList<String> words, ArrayList<String> tags);

    public static void main(String[] args) {
        // 默认模型路径
        String model_path = new String("./lac_model");
        
        if (args.length > 0) {
            model_path = args[0];
        }

        LAC lac = new LAC(model_path);

        if (args.length > 1) {
            lac.loadCustomization(args[1]);
        }

        ArrayList<String> words = new ArrayList<>();
        ArrayList<String> tags = new ArrayList<>();

        lac.run("百度是一家高科技公司", words, tags);
        System.out.println(words);
        System.out.println(tags);

        lac.run("LAC是一个优秀的分词工具", words, tags);
        System.out.println(words);
        System.out.println(tags);
    }
}
