package com.compilesense.liuyi.mcldroid.mcldroid;

/**
 * Created by shenjingyuan002 on 2017/3/3.
 */

public class NetFile {
    public interface RootType {
        String ROOT_DIRECTORY = "root_directory";
        String ALLOCATED_RAM = "allocated_ram";
        String EXECUTION_MODE ="execution_mode";
        String AOTO_TUNING = "auto_tuning";
        String LAYER = "layer";
    }

    public interface LayerType{
        String CONVOLUTION = "convolution";
        String RELU = "relu";
        String PRELU = "prelu";
        String POOLING = "pooling";
        String LRN = "lrn";
        String FULLY_CONNECTED = "fullyconnected";
        String SOFT_MAX = "softmax";
    }

    public interface LayerParams {
        String NAME = "name";
        String TYPE = "type";
        String PARAM_FILE = "parameters_file";
        String PAD = "pad";
        String STRIDE = "stride";
        String GROUP = "group";
        String POOL = "pool";
        String KERNEL_SIZE ="kernel_size";
        String NORM_REGION = "norm_region";
        String LOCAL_SIZE = "local_size";
        String ALPHA = "alpha";
        String BETA = "beta";
    }
}
