import java.util.ArrayList;
import java.io.InputStreamReader;
import java.io.BufferedReader;

import com.baidu.nlp.LAC;

public class LacDemo {

    static {
        // 装载LAC链接库，需要lacjni的链接库文件放到当前目录下
        // System.setProperty("java.library.path", ".");
        System.loadLibrary("lacjni");
    }

    public static void main(String[] args) {

        // 默认模型路径
        String model_path = new String("./lac_model");
        
        if (args.length > 0) {
            model_path = args[0];
        }

        LAC lac = new LAC(model_path);

        // 装载用户词典
        if (args.length > 1) {
            lac.loadCustomization(args[1]);
        }

        ArrayList<String> words = new ArrayList<>();
        ArrayList<String> tags = new ArrayList<>();
        BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
        String query = null;
        
        try {
            while ((query = stdin.readLine()) != null) {
                lac.run(query, words, tags);
                System.out.println(words);
                System.out.println(tags);
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }
}
