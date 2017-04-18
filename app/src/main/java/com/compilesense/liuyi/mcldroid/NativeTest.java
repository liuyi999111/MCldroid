package com.compilesense.liuyi.mcldroid;

import android.graphics.Bitmap;

/**
 * Created by shenjingyuan002 on 2017/3/21.
 */

public class NativeTest {

    static public native void testConvolutionLayerCompute();
    static public native void poolingLayerTest();
    static public native void multiDimensionDataTest();
    static public native void sofymaxLayerTest();

    static public native void testComputeRelu();
    static public native void testComputePrelu();
    static public native void testComputeTanh();
    static public native void testEltwiseLayer();

    static public native void bitmapProcess(Bitmap bitmap);

    static public native void logSystemTest(String path, Bitmap bitmap);

//    static public native void testProtobuf(String filePath);
    static public native void testMsgPack(String filePath);
    static public native void testModelInput(String filePath);

    static public native void lrnLayerTest();
}
