package com.compilesense.liuyi.mcldroid.mcldroid;

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

    public ActivationLayer(String name, int type){
        super(name, LAYER_TYPE_ACTIVATION);
        nativeObject = createActivationLayer(name, type);
    }
    @Override
    public Object compute(Object input) {
        return null;
    }

    private static native long createActivationLayer(String name, int type);
}
