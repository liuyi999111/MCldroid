package com.compilesense.liuyi.mcldroid.mcldroid;

/**
 * Created by shenjingyuan002 on 2017/4/14.
 */

public class LRNLayer extends BaseLayer {

    private static final String TAG = "SoftmaxLayer";

    public LRNLayer(String name, int localSize, float alpha, float beta){
        super(name, LAYER_TYPE_LRN);
        nativeObject = createLRNLayer(name, localSize, alpha, beta);
    }

    @Override
    public Object compute(Object input) {
        return null;
    }

    public void releaseLayer() {
        //空
    }

    private static native long createLRNLayer(String name, int localSize, float alpha, float beta );
}
