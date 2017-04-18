package com.compilesense.liuyi.mcldroid.mcldroid;

/**
 * Created by shenjingyuan002 on 2017/3/27.
 */

public abstract class BaseLayer implements Layer {

    protected String name;
    protected long nativeObject;
    public final int LATER_TYPE;

    protected BaseLayer(String name, int TYPE){
        this.name = name;
        this.LATER_TYPE = TYPE;
    }

}
