package com.compilesense.liuyi.mcldroid.mcldroid;

/**
 * Created by shenjingyuan002 on 2017/3/27.
 */

public class PoolingLayer extends BaseLayer {
    private static final String TAG = "PoolingLayer";
    private int stride;
    private int pad;
    private boolean kernelSize;

    public static final int TYPE_MAX = 1;
    public static final int TYPE_MEAN = 2;

    public PoolingLayer(String name, int type, int pad, int stride, int kernelSize){
        super(name, LAYER_TYPE_POOLING);
        nativeObject = createPoolingLayerNative(name, type, pad, stride, kernelSize);
    }
    @Override
    public Object compute(Object input) {
        return null;
    }

    public void releaseLayer() {

    }

    private static native long createPoolingLayerNative(String name, int type, int pad, int stride, int kernelSize);
}
