package com.compilesense.liuyi.mcldroid.messagepack;

import android.content.Context;
import android.util.Log;

import com.compilesense.liuyi.mcldroid.mcldroid.ActivationLayer;
import com.compilesense.liuyi.mcldroid.mcldroid.BaseLayer;
import com.compilesense.liuyi.mcldroid.mcldroid.ConvolutionLayer;
import com.compilesense.liuyi.mcldroid.mcldroid.FullyConnectedLayer;
import com.compilesense.liuyi.mcldroid.mcldroid.LRNLayer;
import com.compilesense.liuyi.mcldroid.mcldroid.Layer;
import com.compilesense.liuyi.mcldroid.mcldroid.NetFile;
import com.compilesense.liuyi.mcldroid.mcldroid.PoolingLayer;
import com.compilesense.liuyi.mcldroid.mcldroid.SoftmaxLayer;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

/**
 * Created by shenjingyuan002 on 2017/3/3.
 */

public class ModelInput {
    private static final String TAG = "ModelInput";

    private ModelInput(){}
    private final static ModelInput mi = new ModelInput();
    public static ModelInput getInstance() {
        return mi;
    }

    private static long MAX_PARAM_SIZE = 419430400;
    private long allocatedRAM = -1;
    private boolean loadAtStart[];
    private int loadIndex = 0;
    private String root; //参数文件的目录
    private int layerNum; //layer层数
    private boolean hadInput = false;

    private List<BaseLayer> layerList = new ArrayList<>();

    private IHandLayers layersListener;
    static {
        Runtime rt=Runtime.getRuntime();
        long maxMemory = rt.maxMemory();//最大申请内存大小
        MAX_PARAM_SIZE = maxMemory/2;
        Log.d(TAG,"maxMemory:"+ maxMemory+", MAX_PARAM_SIZE:"+MAX_PARAM_SIZE);
    }

    public ModelInput setLayersListener( IHandLayers layersListener){
        this.layersListener = layersListener;
        return this;
    }

    public ModelInput resetStatus(){
        layerList.clear();
        return this;
    }

    public void readNetFileFromAssert (Context context, String fileName) throws Exception{
        InputStream stream = context.getAssets().open(fileName);
        Log.d(TAG,"参数预解析");
        Log.d(TAG,"---------------------------------------------");
        Scanner s = new Scanner(stream);
        List<Long> paramSize = new ArrayList<>();
        //TODO 在之前java 层读取较大 model 时内存限制,预读取一下,现在改为 native 层直接读取,应该不多余的步骤了。
        preParesNetFile(s, paramSize);
        s.close();
        stream.close();
        Log.d(TAG,"参数解析");
        Log.d(TAG,"---------------------------------------------");
        InputStream stream2 = context.getAssets().open(fileName);
        Scanner s2 = new Scanner(stream2);
        paresNetFile(s2);
        s2.close();
        stream2.close();
    }



    //主要是检查参数文件的大小
    private void preParesNetFile(Scanner s, List<Long> paramSize) throws Exception{

        while (s.hasNextLine()) {
            String str = s.nextLine();
            str = str.trim();
            String strLow = str.toLowerCase();

            if (strLow.startsWith(NetFile.RootType.ROOT_DIRECTORY)) {

                str = str.substring(NetFile.RootType.ROOT_DIRECTORY.length());
                root = deriveStr(str);

            } else if (strLow.startsWith(NetFile.RootType.ALLOCATED_RAM)) {

                str = str.substring(NetFile.RootType.ALLOCATED_RAM.length());
                long l = Long.parseLong(deriveNum(str));
                if (l < MAX_PARAM_SIZE)
                    allocatedRAM = l * 1024 * 1024;
                else
                    allocatedRAM = MAX_PARAM_SIZE;

            } else if (strLow.startsWith(NetFile.LayerParams.PARAM_FILE)){    //预先读取参数数据的整体大小

                str = str.substring(NetFile.LayerParams.PARAM_FILE.length());
                String fName = deriveStr(str);
                File pf = new File(root + fName);
                if (pf.exists())
                    paramSize.add(pf.length());
                else {
                    String temp = root + fName;
                    Log.e("CNNdroid", "Error: Missing parameters file \"" + temp + "\"");
                    throw new Exception("CNNdroid parameter file does not exist.");
                }

            }
        }

        if (root.equals("")) {
            Log.e("CNNdroid", "Error: root_directory is not specified in the network structure definition file");
            throw new Exception("CNNdroid root directory is not specified.");
        }else {
            Log.d(TAG,"root:"+ root);
        }

        if (allocatedRAM == -1) {
            Log.e("CNNdroid", "Error: allocated_ram is not specified in the network structure definition file");
            throw new Exception("CNNdroid allocated RAM is not specified.");
        }else {
            Log.d(TAG,"allocatedRAM:"+ allocatedRAM/(1024*1024));
        }

        //判断参数是否需要在开始就加载
        long[] params = longArray(paramSize);
        int[] index = mergeSort(params, 0, params.length - 1);
        loadAtStart = new boolean[params.length];
        long sum = allocatedRAM;
        for (int i = 0; i < params.length; ++i) {
            if (sum - params[index[i]] >= 0) {
                sum -= params[index[i]];
                loadAtStart[index[i]] = true;
            }
        }

    }

    private void paresNetFile(Scanner s) throws Exception{
        while (s.hasNextLine()) {
            String str = s.nextLine();
            str = str.trim();
            String strLow = str.toLowerCase();
            StringBuilder stringBuilder = new StringBuilder();
            if (strLow.startsWith(NetFile.RootType.LAYER)) {
                ++layerNum;
                stringBuilder.append(
                        str.substring(NetFile.RootType.LAYER.length()));
                String tempStr;
                while (s.hasNextLine()) {
                    stringBuilder.append( "\n" );
                    tempStr = s.nextLine();
                    stringBuilder.append( tempStr);
                    if (tempStr.contains("}")){
                        break;
                    }
                }
                str = stringBuilder.toString();
                if (!deriveLayer(str)) {
                    Log.e("CNNdroid", "Error: Layer number " + layerNum + " is not defined correctly in the network structure definition file");
                    throw new Exception("CNNdroid layer number " + layerNum + " is not defined correctly.");
                }
            }
        }


        layersListener.handLayers(layerNum, layerList);

    }

    //取:后, " " 中的参数值
    private String deriveStr(String str) {
        str = str.trim();
        if (!str.startsWith(":"))
            return "";
        str = str.substring(1);

        str = str.trim();
        if (!str.startsWith("\""))
            return "";
        str = str.substring(1);

        int i = str.indexOf("\"");
        if (i == -1)
            return "";

        String retStr = str.substring(0, i);

        if (i != str.length() - 1)
        {
            str = str.substring(i);
            str = str.trim();
            if (!str.equals(""))
                return "";
        }

        return retStr;
    }

    private String deriveNum(String str) {
        str = str.trim();
        if (!str.startsWith(":"))
            return "";
        str = str.substring(1);

        str = str.trim();
        int i = str.indexOf(" ");

        if (i == -1)
            return str;

        if (!str.substring(i).trim().equals(""))
            return "";
        else
            return str.substring(0, i);
    }

    private boolean deriveLayer(String str) {
        String[] strArr = str.split("\n");

        strArr[0] = strArr[0].trim();
        strArr[1] = strArr[1].trim();

        if (strArr[0].startsWith("{"))
            strArr[0] = strArr[0].substring(1);
        else if (strArr[1].startsWith("{"))
            strArr[1] = strArr[1].substring(1);
        else
            return false;

        String endStr = strArr[strArr.length - 1];
        endStr = endStr.trim();
        if (!endStr.startsWith("}"))
            return false;
        endStr = endStr.substring(1);
        endStr = endStr.trim();
        if (!endStr.equals(""))
            return false;

        String type = "";
        String name = "";
        List<String> args = new ArrayList<String>();
        List<String> values= new ArrayList<String>();

        for (int i = 0; i < strArr.length - 1; ++i) {

            strArr[i] = strArr[i].trim();
            if (strArr[i].equals("")){
                continue;
            }

            int i1 = strArr[i].indexOf(":");
            int i2 = strArr[i].indexOf(" ");
            int j = (i1 < i2) ? i1 : i2;
            String tempArg = strArr[i].substring(0, j);
            String tempValue;
            String temp = strArr[i].substring(j);
            if (temp.contains("\"")){
                tempValue = deriveStr(temp);
            } else{
                tempValue = deriveNum(temp);
            }
            if (tempArg.equalsIgnoreCase(NetFile.LayerParams.NAME))
                name = tempValue;
            else if (tempArg.equalsIgnoreCase(NetFile.LayerParams.TYPE))
                type = tempValue;
            else
            {
                args.add(tempArg);
                values.add(tempValue);
            }
        }

        //处理卷积层。
        if (type.equalsIgnoreCase(NetFile.LayerType.CONVOLUTION)) {
            String parametersFile = null;
            int pad = -1;
            int stride = -1;
            int group = -1;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase(NetFile.LayerParams.PARAM_FILE))
                    parametersFile = tempValue;
                else if (tempArg.equalsIgnoreCase(NetFile.LayerParams.PAD))
                    pad = Integer.parseInt(tempValue);
                else if (tempArg.equalsIgnoreCase(NetFile.LayerParams.STRIDE))
                    stride = Integer.parseInt(tempValue);
                else if (tempArg.equalsIgnoreCase(NetFile.LayerParams.GROUP))
                    group = Integer.parseInt(tempValue);
                else
                    return false;
            }
            if (parametersFile == null || pad == -1 || stride == -1 || group == -1)
                return false;
//            Log.d(TAG,"处理 conv 参数: "+"pad:"+pad+",stride:"+stride+",group:"+group);

            //跳过 conv4-2
            if (name.equals("conv4-2")){
                return true;
            }

            ConvolutionLayer convolutionLayer =
                    new ConvolutionLayer(
                            name,
                            new int[]{stride, stride},
                            new int[]{pad, pad},
                            group,
                            false
                    );
            convolutionLayer.setParamPath(root+parametersFile);
            convolutionLayer.loadParamNative();
            layerList.add(convolutionLayer);
            return true;
        }
        else if (type.equalsIgnoreCase(NetFile.LayerType.POOLING)) {
            String pool = null;
            int kernelSize = -1;
            int pad = -1;
            int stride = -1;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase(NetFile.LayerParams.POOL))
                    pool = tempValue;
                else if (tempArg.equalsIgnoreCase(NetFile.LayerParams.KERNEL_SIZE))
                    kernelSize = Integer.parseInt(tempValue);
                else if (tempArg.equalsIgnoreCase(NetFile.LayerParams.PAD))
                    pad = Integer.parseInt(tempValue);
                else if (tempArg.equalsIgnoreCase(NetFile.LayerParams.STRIDE))
                    stride = Integer.parseInt(tempValue);
                else
                    return false;
            }
            if (pool == null || pad == -1 || stride == -1 || kernelSize == -1)
                return false;
            //处理 pool params
            int poolingType;
            if (pool.equals("max")){
                poolingType = PoolingLayer.TYPE_MAX;
            }else {
                poolingType = PoolingLayer.TYPE_MEAN;
            }
            PoolingLayer poolingLayer = new PoolingLayer(name, poolingType, pad, stride, kernelSize);
            layerList.add(poolingLayer);
            return true;
        }
        else if (type.equalsIgnoreCase(NetFile.LayerType.LRN)) {
            String normRegion = null;
            int localSize = -1;
            double alpha = -1.0;
            double beta = -1.0;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase(NetFile.LayerParams.NORM_REGION))
                    normRegion = tempValue;
                else if (tempArg.equalsIgnoreCase(NetFile.LayerParams.LOCAL_SIZE))
                    localSize = Integer.parseInt(tempValue);
                else if (tempArg.equalsIgnoreCase(NetFile.LayerParams.ALPHA))
                    alpha = Double.parseDouble(tempValue);
                else if (tempArg.equalsIgnoreCase(NetFile.LayerParams.BETA))
                    beta = Double.parseDouble(tempValue);
                else
                    return false;
            }
            if (normRegion == null || localSize == -1 || alpha == -1.0 || beta == -1.0)
                return false;

            LRNLayer lrn =  new LRNLayer(name,localSize, (float) alpha, (float) beta);
            layerList.add(lrn);
            return true;
        }
        else if (type.equalsIgnoreCase(NetFile.LayerType.FULLY_CONNECTED)) {
            String parametersFile = null;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase(NetFile.LayerParams.PARAM_FILE))
                    parametersFile = tempValue;
                else
                    return false;
            }
            if (parametersFile == null)
                return false;
            FullyConnectedLayer fcLayer = new FullyConnectedLayer(name, false);
            fcLayer.setParamPath(root+parametersFile);
            fcLayer.loadParamNative();
            layerList.add(fcLayer);
            return true;
        }
        else if (type.equalsIgnoreCase("Accuracy")) {
//            Log.e(TAG,"Accuracy 不支持");
//            String parametersFile = null;
//            int topk = -1;
//            for (int i = 0; i < args.size(); ++i) {
//                String tempArg = args.get(i);
//                String tempValue = values.get(i);
//                if (tempArg.equalsIgnoreCase("parameters_file"))
//                    parametersFile = tempValue;
//                else if (tempArg.equalsIgnoreCase("topk"))
//                    topk = Integer.parseInt(tempValue);
//                else
//                    return false;
//            }
//            if (parametersFile == null || topk == -1)
//                return false;
//
//            loadIndex++;
////            Accuracy a = new Accuracy(topk, rootDir + parametersFile, name);
////            lastLayer = a;
////            layers.add(a);
            return true;
        }
        else if (type.equalsIgnoreCase(NetFile.LayerType.SOFT_MAX)) {
            if (args.size() != 0) return false;

            SoftmaxLayer softmaxLayer = new SoftmaxLayer(name);
            layerList.add(softmaxLayer);
            return true;
        }

        else if (type.equalsIgnoreCase(NetFile.LayerType.RELU)) {
            /**
             * 如果 relu 之前的层是 convolution layer 或者是 fully connect layer
             * 则可以将这部分非线性计算直接添加到前面的层中
             * 目前性能提升并不明显!
             */
//            BaseLayer lastLayer = layerList.get(layerList.size() - 1);
//            if (lastLayer.LATER_TYPE == Layer.LAYER_TYPE_CONVOLUTION){
//                ConvolutionLayer layer = (ConvolutionLayer) lastLayer;
//                layer.setNonLinear();
//                return true;
//            }

            ActivationLayer activationLayer = new ActivationLayer(name, ActivationLayer.TYPE_RELU);
            layerList.add(activationLayer);
            return true;
        }

        else if (type.equalsIgnoreCase(NetFile.LayerType.PRELU)){
            String parametersFile = null;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase(NetFile.LayerParams.PARAM_FILE))
                    parametersFile = tempValue;
                else
                    return false;
            }
            if (parametersFile == null)
                return false;
            ActivationLayer activationLayer = new ActivationLayer(name, ActivationLayer.TYPE_PRELU);
            activationLayer.setParamsFilePath(root+parametersFile)
                    .loadParams();
            layerList.add(activationLayer);
            return true;
        }
        else
            return false;
    }

    long[] longArray(List<Long> ll) {
        long[] l = new long[ll.size()];
        Iterator<Long> i = ll.iterator();

        int j = 0;
        while (i.hasNext()) {
            l[j] = i.next();
            ++j;
        }

        return l;
    }

    // 递减 order
    private int[] mergeSort(long[] a, int s, int e) {
        if (s > e)
            return null;

        if (s == e)
            return new int[]{s};

        int m = (s + e) / 2;
        int[] index1 = mergeSort(a, s, m);
        int[] index2 = mergeSort(a, m + 1, e);

        return merge(a, index1, index2);
    }

    private int[] merge(long a[], int[] index1, int[] index2) {
        int l1 = index1.length;
        int l2 = index2.length;

        int[] index = new int[l1 + l2];
        int i = 0, j = 0, k = 0;

        while (k < l1 + l2) {
            if ((i < l1) && ((j >= l2) || (a[index1[i]] > a[index2[j]]))) {
                index[k] = index1[i];
                ++i;
            } else {
                index[k] = index2[j];
                ++j;
            }
            ++k;
        }

        return index;
    }


    public interface IHandLayers{
        void handLayers(int size, List<BaseLayer> layerList);
    }
}
