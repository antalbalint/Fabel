package org.fabel.common;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;
import java.io.IOException;

/**
 * Created by IntelliJ IDEA.
 * User: Balint
 * Date: 2011.12.07.
 * Time: 14:39
 * To change this template use File | Settings | File Templates.
 */
public class DatasetUtil {

    public static void convertDataset(String in, String out) throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setFile(new File(in));
        ArffSaver saver = new ArffSaver();
        Instances is = loader.getDataSet();
        //is.setClassIndex(is.numAttributes() - 1);
        saver.setInstances(is);
        saver.setFile(new File(out));
        saver.writeBatch();
    }

    public static Instances loadDataset(String in) throws IOException {
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(in));
        return loader.getDataSet();
    }

    public static Instances convertClassType(Instances is) throws Exception {

        NumericToNominal filter = new NumericToNominal();
        filter.setInputFormat(is);
        filter.setAttributeIndices(String.valueOf(is.classIndex() + 1));
        filter.setInvertSelection(false);
        for (int i = 0; i < is.numInstances(); i++) {
            filter.input(is.instance(i));
        }
        filter.batchFinished();
        Instances newTrainData = filter.getOutputFormat();
        Instance processed;
        while ((processed = filter.output()) != null) {
            newTrainData.add(processed);
        }
        return newTrainData;
    }

    public static void main(String[] args) {
        try {
            convertDataset("d:\\datasets\\uci\\poker-hand-training.data", "d:\\datasets\\uci\\poker-hand-training.arff");
        } catch (IOException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }
    }

}

