package com.compilesense.liuyi.mcldroid.mcldroid;

import android.util.Log;

import com.compilesense.liuyi.mcldroid.messagepack.ParamUnpacker;


/**
 * Created by shenjingyuan002 on 2017/3/7.
 */

public class FullyConnectedLayer extends BaseLayer {
    private static final String TAG = "FullyConnectedLayer";
    private String name;
    private boolean nonLinear;
    private String paramFilePath;
    private boolean paramHasLoad = false;
    private float[] weight;
    private float[] bias;

    public FullyConnectedLayer(String name, boolean nonLinear){
        super(name, LAYER_TYPE_FULLY_CONNECTED);
        this.nonLinear = nonLinear;
        this.nativeObject = createNativeObject(name, nonLinear);
    }
    public void releaseFcLayer(){
        deleteNativeObject(this.nativeObject);
        if (weight != null){
            weight = null;
        }
        if (bias != null){
            bias = null;
        }
    }

    public void loadParam(){
        double time = System.currentTimeMillis();
        ParamUnpacker paramUnpacker = new ParamUnpacker();
        Log.d(TAG,"paramFilePath:"+paramFilePath);
        Object[] objects = paramUnpacker.unpackFunction(paramFilePath, new Class[]{float[].class, float[].class});
        weight = (float[]) objects[0];
        bias = (float[]) objects[1];
        Log.d(TAG,"--------------------------------------------");
        Log.d(TAG,"完成参数加载,用时:"+(System.currentTimeMillis() - time));
        Log.d(TAG,"参数大小: weight:"+(weight.length));
        Log.d(TAG,"参数大小: bias:"+(bias.length));
        Log.d(TAG,"--------------------------------------------");
        setParam(this.nativeObject,weight, bias);
        paramHasLoad = true;
    }

    public void loadParamNative(){
        double time = System.currentTimeMillis();
        loadKernelNative(nativeObject, paramFilePath);
        Log.d(TAG,"--------------------------------------------");
        Log.d(TAG,"paramFilePath:"+paramFilePath);
        Log.d(TAG,"完成参数加载,用时:"+(System.currentTimeMillis() - time));
        Log.d(TAG,"--------------------------------------------");
        paramHasLoad = true;
    }

    public FullyConnectedLayer setParamPath(String paramPath) {
        this.paramFilePath = paramPath;
        return this;
    }

    @Override
    public Object compute(Object input) {
        return null;
    }

    public void releaseLayer() {
//        releaseFcLayer();
    }

    private native long createNativeObject(
            String name,
            boolean nonLinear);
    private native void deleteNativeObject(long objPrt);
    private native void setParam(long objPrt, float[] weight,  float[] bias);
    private native void loadKernelNative(long objPtr, String paramFilePath);
}
