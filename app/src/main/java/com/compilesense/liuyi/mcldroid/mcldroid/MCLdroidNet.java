package com.compilesense.liuyi.mcldroid.mcldroid;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.compilesense.liuyi.mcldroid.messagepack.ModelInput;
import com.compilesense.liuyi.mcldroid.messagepack.ParamUnpacker;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by shenjingyuan002 on 2017/3/24.
 */

public class MCLdroidNet {
    private MCLdroidNet(){}
    private final static MCLdroidNet mcl = new MCLdroidNet();
    public static MCLdroidNet getInstance() {
        return mcl;
    }

    private List<BaseLayer> layerList = new ArrayList<>();

    public void setUpNet(Context context){
        try {
            ModelInput.getInstance()
                    .setLayersListener(new ModelInput.IHandLayers() {
                        @Override
                        public void handLayers(int size, List<BaseLayer> layerList_) {
                            layerList = layerList_;
                            long[] nativeObjArray = new long[layerList.size()];
                            for (int i = 0; i < layerList.size(); i++){
                                nativeObjArray[i] = layerList.get(i).nativeObject;
                            }
                            if (nativeObjArray.length>0){
                                setupNet(nativeObjArray);
                            }
                        }
            })
                    .resetStatus()
//                    .readNetFileFromAssert(context,"Cifar10_def.txt");
                    .readNetFileFromAssert(context,"CaffeNet_def.txt");
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public void testInputBitmap(Bitmap bitmap){
        bitmapProcess(bitmap);
//        getInputMean();
        compute();
    }

    //对于 cifar10 需要对输入进行处理。
    void getInputMean(){
        ParamUnpacker pu = new ParamUnpacker();
        float[][][] mean = (float[][][]) pu.unpackFunction("/sdcard/Data_Cifar10/mean.msg", float[][][].class);
        int[] shape = new int[]{mean.length, mean[0].length,mean[0][0].length};
        float[] m = float3DArrayTo1D(mean, shape);
        setInputMean(m, shape);
    }

    private float[] float3DArrayTo1D(float[][][] input, int[] shape){
        int size = 1;
        for (int s : shape) {
            size *= s;
        }
        if (size <= 0){
            return null;
        }

        float[] outPut = new float[size];
        int index = 0;


        for (float[][] j: input){
            for (float[] k: j){
                for (float l : k) {
                    outPut[index] = l;
                    index++;
                }
            }
        }

        return outPut;
    }

    native void addLayer(long nativeObject);

    native void setupNet(long[] nativeObjectArray);

    native void bitmapProcess(Bitmap bitmap);

    native void compute();

    native void setInputMean(float[] mean,int[] meanShape);

    native void getBitmapProcessed(Bitmap bitmap);
}
