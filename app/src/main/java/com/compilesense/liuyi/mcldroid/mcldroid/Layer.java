package com.compilesense.liuyi.mcldroid.mcldroid;

/**
 * Created by shenjingyuan002 on 2017/3/3.
 */

public interface Layer {
    int LAYER_TYPE_ACTIVATION = 111;
    int LAYER_TYPE_CONVOLUTION = 222;
    int LAYER_TYPE_FULLY_CONNECTED = 333;
    int LAYER_TYPE_LRN = 444;
    int LAYER_TYPE_POOLING = 555;
    int LAYER_TYPE_SOFTMAX = 666;

    Object compute(Object input);
}
