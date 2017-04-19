package com.compilesense.liuyi.mcldroid.mcldroid;

import android.util.Log;

/**
 * Created by shenjingyuan002 on 2017/3/27.
 */

public class ActivationLayer extends BaseLayer {
    private static final String TAG = "ActivationLayer";
    //ReLU,PReLU,TanH,Abs
    public static final int TYPE_RELU = 1;
    public static final int TYPE_PRELU = 2;
    public static final int TYPE_TANH = 3;
    public static final int TYPE_ABS= 4;

    private String paramsFilePath;

    public ActivationLayer(String name, int type){
        super(name, LAYER_TYPE_ACTIVATION);
        nativeObject = createActivationLayer(name, type);
    }

    public ActivationLayer setParamsFilePath(String filePath){
        paramsFilePath = filePath;
        return this;
    }

    public void loadParams(){
        double time = System.currentTimeMillis();
        Log.d(TAG,"--------------------------------------------");
        Log.d(TAG,"paramFilePath:"+ paramsFilePath);
        loadParamsNative(nativeObject, paramsFilePath);
        Log.d(TAG,"完成参数加载,用时:"+(System.currentTimeMillis() - time));
        Log.d(TAG,"--------------------------------------------");

    }

    @Override
    public Object compute(Object input) {
        return null;
    }

    private static native long createActivationLayer(String name, int type);

    private native void loadParamsNative(long objPtr, String paramFilePath);
}
