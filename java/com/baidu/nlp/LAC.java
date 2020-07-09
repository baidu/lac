package com.baidu.nlp;

import java.util.ArrayList;

public class LAC {

    // as c++ self pointer
    private long self_ptr;

    public LAC(String model_dir) {
        init(model_dir);
    }

    public LAC(LAC model){
        copy(model.self_ptr);
    }

    @Override
    protected void finalize() throws Throwable {
        close();
        super.finalize();
    }

    public void close() {
        if(self_ptr != 0) {
            release(self_ptr);
            self_ptr = 0;
        }
    }

    // load model from model_path
    private native void init(String model_path);

    // load model from existing model's self_ptr
    private native void copy(long self_ptr);

    // release lac model
    private static native void release(long self_ptr);

    // load dict from dict_path
    public native int loadCustomization(String dict_path);

    // run lac, results save in words and tags
    public native int run(String sentence, ArrayList<String> words, ArrayList<String> tags);

}
