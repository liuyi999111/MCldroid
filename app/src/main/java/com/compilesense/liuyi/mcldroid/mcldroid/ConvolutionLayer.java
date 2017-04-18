package com.compilesense.liuyi.mcldroid.mcldroid;


import android.util.Log;

import com.compilesense.liuyi.mcldroid.messagepack.ParamUnpacker;


/**
 * TODO 修改多分组时计算的正确性
 * Created by shenjingyuan002 on 2017/3/3.
 */

public class ConvolutionLayer extends BaseLayer {
    private static final String TAG = "ConvolutionLayer";
    private int stride[];
    private int pad[];
    private int group;
    private boolean nonLinear;
    private String paramFilePath;
    private boolean paramHasLoad = false;
    private float[][][][] weight;
    private int[] weightShape;
    private float[] bias;

    public ConvolutionLayer(String name, int[] stride, int[] pad, int group, boolean nonLinear) {
        super(name, LAYER_TYPE_CONVOLUTION);
        this.stride = stride;
        this.pad = pad;
        this.group = group;
        this.nonLinear = nonLinear;
        nativeObject = createConvolutionLayer(name, stride, pad, group, nonLinear);
    }

    public void releaseConvolutionLayer(){
        deleteConvolutionLayer(this.nativeObject);
        if (weight != null){
            weight = null;
        }
        if (bias != null){
            bias = null;
        }
    }

    public ConvolutionLayer setParamPath(String paramPath) {
        this.paramFilePath = paramPath;
        return this;
    }

    public ConvolutionLayer loadParam(){
        double time = System.currentTimeMillis();
        ParamUnpacker paramUnpacker = new ParamUnpacker();
        Log.d(TAG,"paramFilePath:"+paramFilePath);
        Object[] objects = paramUnpacker.unpackFunction(paramFilePath, new Class[]{float[][][][].class, float[].class});
        weight = (float[][][][]) objects[0];
        bias = (float[]) objects[1];
        weightShape = new int[4];
        weightShape[0] = weight.length;
        weightShape[1] = weight[0].length;
        weightShape[2] = weight[0][0].length;
        weightShape[3] = weight[0][0][0].length;
        Log.d(TAG,"--------------------------------------------");
        Log.d(TAG,"完成参数加载,用时:"+(System.currentTimeMillis() - time));
        Log.d(TAG,"参数大小: weight:"+( weightShape[0] )+","
                +( weightShape[1] ) + ","
                +( weightShape[2] ) + ","
                +( weightShape[3] ));
        Log.d(TAG,"weight last data: "+ (weight
                [ weightShape[0] -1]
                [ weightShape[1] -1]
                [ weightShape[2] -1]
                [ weightShape[3] -1]));
        Log.d(TAG,"参数大小: bias:"+(bias.length));
        Log.d(TAG,"bias last data: "+ (bias[bias.length - 1]));
        Log.d(TAG,"--------------------------------------------");

        setKernel(this.nativeObject, float4DArrayTo1D(weight, weightShape), weightShape, bias);
        paramHasLoad = true;
        return this;
    }

    public ConvolutionLayer loadParamNative(){
        double time = System.currentTimeMillis();
        Log.d(TAG,"--------------------------------------------");
        Log.d(TAG,"paramFilePath:"+paramFilePath);
        loadKernelNative(nativeObject, paramFilePath);
        Log.d(TAG,"完成参数加载,用时:"+(System.currentTimeMillis() - time));
        Log.d(TAG,"--------------------------------------------");
        paramHasLoad = true;
        return this;
    }

    public void setNonLinear(){
        setNonlinearType(nativeObject, 1);
    }

    @Override
    public Object compute(Object input) {
        return null;
    }

    public void releaseLayer() {
        Log.d(TAG,"releaseLayer, name:"+ name + " , prt:"+ nativeObject);
        releaseConvolutionLayer();
    }

    private float[] float4DArrayTo1D(float[][][][] input, int[] shape){
        int size = 1;
        for (int s : shape) {
            size *= s;
        }
        if (size <= 0){
            return null;
        }

        float[] outPut = new float[size];
        int index = 0;

        for (float[][][] i : input){
            for (float[][] j: i){
                for (float[] k: j){
                    for (float l : k) {
                        outPut[index] = l;
                        index++;
                    }
                }
            }
        }
        return outPut;
    }

    /**
     * 创建 ConvolutionLayer 的 native 层对象。
     * @param name 名称
     * @param stride 步长
     * @param pad 留白
     * @param group 分组 (目前在 group > 1 时计算结果不正确)
     * @param nonLinear 是否包含非线性部分
     * @return
     */
    private native long createConvolutionLayer(
            String name,
            int stride[],
            int pad[],
            int group,
            boolean nonLinear
    );

    private native void deleteConvolutionLayer(long nativeObject);

    private native void setKernel(long objPrt,float[] weight ,int[] weightShape,  float[] bias);

    /**
     * 加载 kernel
     * @param objPrt native 层 c++ 类指针
     * @param paramFilePath 存放 weight 和 bias 的文件路径
     */
    private native void loadKernelNative(long objPrt, String paramFilePath);

    /**
     * 设置 convolution layer 的非线性部分
     * @param objPtr
     * @param type 目前只支持 ReLu
     */
    private native void setNonlinearType(long objPtr, int type);
}
