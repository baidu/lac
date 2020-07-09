import java.util.ArrayList;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import com.baidu.nlp.LAC;

public class LacMulti {
    static {
        // 装载LAC链接库，需要lacjni的链接库文件放到当前目录下
        // System.setProperty("java.library.path", ".");
        System.loadLibrary("lacjni");
    }

    public static void main(String[] args) {

        // 默认模型路径
        String model_path = new String("./lac_model");
        if (args.length < 2){
            System.out.println("Usage: "+argv[0]+ " + model_dir + thread_num");
            System.exit(1);
        }
 
        model_path = args[0];
        int thread_num = Integer.parseInt(args[1]);

        LAC lac = new LAC(model_path);
        LacRunnable lacrunner = new LacRunnable(lac);

        Thread threads[]=new Thread[thread_num];
        for (int i = 0; i< thread_num; i++){
            threads[i] = new Thread(lacrunner);
            threads[i].start();
        }
    }
    
}

class LacRunnable implements Runnable{

    
    public LacRunnable(LAC lac){
        g_lac = lac;
    }

    // share model and lock
    private LAC g_lac;
    private Lock printLock = new ReentrantLock();
    private Lock readLock = new ReentrantLock();

    public void run() {
        ArrayList<String> words = new ArrayList<>();
        ArrayList<String> tags = new ArrayList<>();
        BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
        String query = null;
        
        try {
            LAC thread_lac = new LAC(g_lac);
            while (true) {
                readLock.lock();
                if ((query = stdin.readLine()) == null){
                    readLock.unlock();
                    break;
                }
                readLock.unlock();
                thread_lac.run(query, words, tags);
 
                printLock.lock();
                System.out.println(words);
                System.out.println(tags);
                printLock.unlock();
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }
}