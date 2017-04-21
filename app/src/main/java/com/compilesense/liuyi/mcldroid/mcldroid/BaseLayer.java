package com.compilesense.liuyi.mcldroid.mcldroid;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by shenjingyuan002 on 2017/3/27.
 */

public abstract class BaseLayer implements Layer {

    protected String name;
    protected long nativeObject;
    public final int LATER_TYPE;
    protected List<BaseLayer> nextLayers = new ArrayList<>();

    protected BaseLayer(String name, int TYPE){
        this.name = name;
        this.LATER_TYPE = TYPE;
    }

    public void addNextLayer(BaseLayer nextLayer){
        nextLayers.add(nextLayer);
        addNextLayerNative(this.nativeObject, nextLayer.nativeObject);
    }

    private native void addNextLayerNative(long nativeObject, long nextLayer);
}
