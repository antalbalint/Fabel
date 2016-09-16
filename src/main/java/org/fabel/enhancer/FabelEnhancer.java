package org.fabel.enhancer;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.FastVector;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.util.Enumeration;
import java.util.Random;

/**
 * Created by Balint on 2015-07-11.
 */
public class FabelEnhancer {

    private double[] actual;
    private double[] origPredictions;
    private double[][] labels;
    private Classifier cls;
    private Instances instances;
    private Instances subTrain;
    private Instances subTest;
    private double target;
    private int numberOfTestInstances;
    private int numberOfClassifiers = 5;
    private double energyThreshold = 0.01;
    private double tempChange = 0.95;
    private double temperature = 4;
    private double penalty = 1;
    private int step = 2;
    private double splitPercentage = 0.5;
    private Random random;

    public FabelEnhancer(Classifier cls, Instances instances, Random random) {
        this.cls = cls;
        this.instances = instances;
        this.random = random;
    }

    protected void splitDataset() throws Exception {
        RemovePercentage rp = new RemovePercentage();
        rp.setPercentage(splitPercentage);
        subTrain = Filter.useFilter(instances, rp);
        rp.setInvertSelection(true);
        subTest = Filter.useFilter(instances, rp);
    }

    protected void getOriginalPredictions() throws Exception {
        Evaluation eval = new Evaluation(subTrain);
        eval.evaluateModel(cls, subTest);
        FastVector fv = eval.predictions();
        Enumeration en = fv.elements();
        numberOfTestInstances = subTest.numInstances();
        origPredictions = new double[numberOfTestInstances];
        actual = new double[numberOfTestInstances];
        int idx = 0;
        while (en.hasMoreElements()) {
            Prediction pred = (Prediction)en.nextElement();
            origPredictions[idx] = pred.predicted();
            actual[idx++] = pred.actual();
        }
        target = eval.errorRate();
    }

    public double[][] createFalseLabels() throws Exception {
        splitDataset();
        getOriginalPredictions();
        labels = new double[numberOfClassifiers][numberOfTestInstances];
        labels[0] = origPredictions;
        for (int i = 1; i < numberOfClassifiers; ++i)  {
            labels[i] = new double[numberOfTestInstances];
        }
        int numClasses = subTrain.numClasses();
        double[] classValues = new double[numClasses];
        for (int i = 0; i < numClasses; ++i) {
            classValues[i] = i;
        }
        EnhancerSA sa = new EnhancerSA(labels, actual, classValues, energyThreshold, tempChange, temperature, penalty, step, random);
        return sa.start(target);
    }

    public Classifier[] createClassifiers() throws Exception {
        if (labels == null) {
            createFalseLabels();
        }
        Classifier[] classifiers = new Classifier[numberOfClassifiers];
        classifiers[0] = cls;
        Class classifierClass = cls.getClass();
        for (int i = 1; i < numberOfClassifiers; ++i) {
            Instances falseLabelled = new Instances(subTest);
            double[] falseLabels = labels[i];
            for (int j = 0; j < falseLabels.length; ++j) {
                falseLabelled.instance(j).setClassValue(falseLabels[j]);
            }
            Classifier falseCls = (Classifier) classifierClass.newInstance();
            falseCls.buildClassifier(falseLabelled);
            classifiers[i] = falseCls;
        }
        return classifiers;
    }

    public double[][] getLabels() {
        return labels;
    }
}
