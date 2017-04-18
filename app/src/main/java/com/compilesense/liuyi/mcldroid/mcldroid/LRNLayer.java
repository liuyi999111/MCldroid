package com.compilesense.liuyi.mcldroid.mcldroid;

/**
 * Created by shenjingyuan002 on 2017/4/14.
 */

public class LRNLayer extends BaseLayer {

    private static final String TAG = "SoftmaxLayer";

    public LRNLayer(String name, int localSize, float alpha, float beta){
        this.name = name;
        nativeObject = createLRNLayer(name, localSize, alpha, beta);
    }

    @Override
    public Object compute(Object input) {
        return null;
    }

    @Override
    public void releaseLayer() {

    }

    private static native long createLRNLayer(String name, int localSize, float alpha, float beta );
}
