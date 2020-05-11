package com.example.testlac;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
    private TextView mCutResult;
    private EditText mInputText;
    private Button mButton;

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native_lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mInputText = (EditText) findViewById(R.id.input_text);
        mCutResult = (TextView) findViewById(R.id.cut_result);
        mButton = (Button) findViewById((R.id.cut_button));
        // Example of a call to a native method
        TextView tv = findViewById(R.id.sample_text);
        if (!checkAllPermissions()) {
            requestAllPermissions();
        }
        tv.setText(stringFromJNI());
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String inputText = mInputText.getText().toString();
                if (!TextUtils.isEmpty(inputText)) {
                    mCutResult.setText(stringCutFromJNI(inputText));
                } else {
                    mCutResult.setText(stringCutFromJNI("输入太短"));
                }
            }
        });

        String model_path = copyFromAssetsToCache("lac_model", this);
        initLac(model_path);

        /* load model for protobuf file, need protobuf lib, abandon
        Resources res = getResources();
        try {
            //获取文件的字节数
            InputStream lac_dict = res.openRawResource(R.raw.lac_dict_model);
            int length = lac_dict.available();
            //创建byte数组
            byte[]  buffer = new byte[length];
            //将文件中的数据读到byte数组中
            lac_dict.read(buffer);
            // 初始化模型
            initLac(buffer, length);
        } catch (IOException e) {
            e.printStackTrace();
        }
         */

    }

    @Override
    protected void onDestroy() {
        releaseLac();
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            new AlertDialog.Builder(MainActivity.this)
                    .setTitle("Permission denied")
                    .setMessage("Click to force quit the app, then open Settings->Apps & notifications->Target " +
                            "App->Permissions to grant all of the permissions.")
                    .setCancelable(false)
                    .setPositiveButton("Exit", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            MainActivity.this.finish();
                        }
                    }).show();
        }
    }

    public static String copyFromAssetsToCache(String modelPath, Context context) {
        String newPath = context.getCacheDir() + "/" + modelPath;
        // String newPath = "/sdcard/" + modelPath;
        File desDir = new File(newPath);

        try {
            if (!desDir.exists()) {
                desDir.mkdir();
            }
            for (String fileName : context.getAssets().list(modelPath)) {
                InputStream stream = context.getAssets().open(modelPath + "/" + fileName);
                OutputStream output = new BufferedOutputStream(new FileOutputStream(newPath + "/" + fileName));

                byte data[] = new byte[1024];
                int count;

                while ((count = stream.read(data)) != -1) {
                    output.write(data, 0, count);
                }

                output.flush();
                output.close();
                stream.close();
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return desDir.getPath();
    }

    private void requestAllPermissions() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE}, 0);
    }

    private boolean checkAllPermissions() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     *
     */
//    public native void initLac(byte[] lac_dict, int length);

    public native void initLac(String model_path);

    public native void releaseLac();

    public native String stringFromJNI();

    public native String stringCutFromJNI(String source_text);
}
