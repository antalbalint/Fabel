package org.fabel.enhancer;


import org.fabel.common.DatasetUtil;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.io.File;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;

public class Enhancer {

    private Instances dataset;
    private Instances train;
    private Instances test;
    private Classifier cls;
    private double[][] labels;
    private double[] actual;

    public Enhancer(Classifier cls, Instances dataset) {
        this.cls = cls;
        this.dataset = dataset;
    }

    public Enhancer(Classifier cls, Instances train, Instances test) {
        this.cls = cls;
        this.train = train;
        this.test = test;
    }

    public void test() throws Exception {

        if (test == null || train == null) {
            dataset.randomize(new Random());
            train = dataset.trainCV(2, 0);
            test = dataset.testCV(2, 0);
        }
        //System.out.println(test.toSummaryString());
        Instances subTrain = train.trainCV(2, 0);
        //System.out.println(subTrain.toSummaryString());
        Instances subTest = train.trainCV(2, 0);
        //System.out.println(subTest.toSummaryString());
        System.out.println(subTrain.classAttribute().toString());
        cls.buildClassifier(subTrain);

        Evaluation eval = new Evaluation(subTrain);
        eval.evaluateModel(cls, test);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }

    public void test(int numFolds, Classifier classifier) throws Exception {
        DescriptiveStatistics stat = new DescriptiveStatistics();
        for (int i = 0; i < numFolds; ++i) {
            Resample filter = new Resample();
            filter.setInputFormat(dataset);
            filter.setBiasToUniformClass(1);
            dataset.randomize(new Random());
            dataset = Filter.useFilter(dataset, filter);
            train = dataset.trainCV(2, 0);
            test = dataset.testCV(2, 0);
            classifier.buildClassifier(train);
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(classifier, test);
            stat.addValue(eval.pctCorrect());
        }
        DecimalFormat form = new DecimalFormat();
        form.setGroupingUsed(false);
        form.setMinimumFractionDigits(2);
        form.setMaximumFractionDigits(2);
        System.out.print(form.format(stat.getMean() / 100.0) + " $\\pm$ " + form.format(stat.getStandardDeviation() / 100.0) + " & ");

    }

    public void crossValidate(int numFolds) throws Exception {
        DescriptiveStatistics stat = new DescriptiveStatistics();
        for (int i = 0; i < numFolds; ++i) {
            Resample filter = new Resample();
            filter.setInputFormat(dataset);
            filter.setBiasToUniformClass(1);
            dataset.randomize(new Random());
            //dataset = Filter.useFilter(dataset, filter);
            train = dataset.trainCV(numFolds, i);
            test = dataset.testCV(numFolds, i);
            cls = new NaiveBayes();
            cls.buildClassifier(train);
            Evaluation eval = new Evaluation(test);
            eval.evaluateModel(cls, test);
            stat.addValue(eval.pctCorrect());
        }
        DecimalFormat form = new DecimalFormat();
        form.setGroupingUsed(false);
        form.setMinimumFractionDigits(2);
        form.setMaximumFractionDigits(2);
        System.out.print(form.format(stat.getMean() / 100.0) + " $\\pm$ " + form.format(stat.getStandardDeviation() / 100.0) + " & ");
        /*Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(cls, dataset, numFolds, new Random());
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));*/
    }

    public void enhanceOnTest() throws Exception {
        dataset.randomize(new Random());
        train = dataset.trainCV(2, 0);
        test = dataset.testCV(2, 0);
        System.out.println(test.toSummaryString());
        Instances subTrain = train.trainCV(2, 0);
        System.out.println(subTrain.toSummaryString());
        Instances subTest = train.testCV(2, 0);
        System.out.println(subTest.toSummaryString());
        System.out.println(subTrain.classAttribute().toString());
        cls.buildClassifier(subTrain);

        Evaluation eval = new Evaluation(subTrain);
        eval.evaluateModel(cls, subTest);
        FastVector fv = eval.predictions();
        /*System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());*/
        Enumeration en = fv.elements();
        int n = subTest.numInstances();
        double[] origPredictions = new double[n];
        actual = new double[n];
        int idx = 0;
        while (en.hasMoreElements()) {
            Prediction pred = (Prediction)en.nextElement();
            origPredictions[idx] = pred.predicted();
            actual[idx++] = pred.actual();
            //System.out.println(pred.predicted() + " " + pred.actual());
        }
        int numOfClassifiers = 33;
        int step = numOfClassifiers/2;
        labels = new double[numOfClassifiers][n];
        labels[0] = origPredictions;
        for (int i = 1; i < numOfClassifiers; ++i)  {
           labels[i] = new double[n];
        }
        int numClasses = train.numClasses();
        double[] classValues = new double[numClasses];
        for (int i = 0; i < numClasses; ++i) {
            classValues[i] = i;
        }
        EnhancerSA sa = new EnhancerSA(labels, actual, classValues, 0.01, 0.95, 4, 1, 30,1);
        double[][] ensemble = sa.start(eval.errorRate());
        for (int i = 0; i < ensemble.length; ++i) {
            System.out.println(Arrays.toString(ensemble[i]));
        }
        System.out.println(Arrays.toString(actual));
        PearsonsCorrelation pc = new PearsonsCorrelation();
        RealMatrix corr = pc.computeCorrelationMatrix(labels);
        double sum = 0;
        double count = 0;
        for (int i = 0; i < numOfClassifiers; ++i) {
            for (int j = i+1; j < numOfClassifiers; ++j) {
                double c = corr.getEntry(i, j);
                if (!Double.isNaN(c)) {
                    sum += Math.abs(c);
                    count++;
                }

            }
        }
        if (count != 0) {
            sum /= count;
        }

        System.out.println("sum of correlations = " + sum);
        Classifier[] classifiers = new Classifier[numOfClassifiers];
        classifiers[0] = cls;
        for (int i = 1; i < numOfClassifiers; ++i) {
            Instances falseLabelled = new Instances(subTest);
            double[] falseLabels = ensemble[i];
            for (int j = 0; j < falseLabels.length; ++j) {
                falseLabelled.instance(j).setClassValue(falseLabels[j]);
            }
            Classifier falseCls = new MultilayerPerceptron();
            falseCls.buildClassifier(falseLabelled);
            /*eval = new Evaluation(test);
            eval.evaluateModel(falseCls, test);
            System.out.println(eval.toSummaryString("\nResults (" + i + ")\n======\n", false));*/
            classifiers[i] = falseCls;
        }
        /*eval = new Evaluation(subTrain);
        eval.evaluateModel(cls, test);

        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());*/
        int inst = test.numInstances();
        double error = 0;
        int tp = 0, fp = 0, tn = 0, fn = 0;
        int tp2 = 0, fp2 = 0, tn2 = 0, fn2 = 0;
        for (int i = 0; i < inst; ++i) {
            Instance instance = test.instance(i);
            double orig = instance.classValue();
            double vote = 0;
            double vote2 = classifiers[0].classifyInstance(instance);
            for (Classifier c: classifiers) {
                vote += c.classifyInstance(instance);
            }
            if (vote > step) {
                if (orig != 1.0) {
                    fp++;
                }
                else {
                    tp++;
                }
            }
            else {
                if (orig != 0.0) {
                    fn++;
                }
                else {
                    tn++;
                }
            }
            if (vote2 == 1.0) {
                if (orig != 1.0) {
                    fp2++;
                }
                else {
                    tp2++;
                }
            }
            else {
                if (orig != 0.0) {
                    fn2++;
                }
                else {
                    tn2++;
                }
            }
        }
        System.out.println(tp2 + " " + fn2);
        System.out.println(fp2 + " " + tn2);
        error = fp2+fn2;
        System.out.println(1.0 - (error/inst));

        System.out.println(tp + " " + fn);
        System.out.println(fp + " " + tn);
        error = fp+fn;
        System.out.println(1.0 - (error/inst));


    }


    public void enhanceOnTestCV(int numFolds, int numOfClassifiers) throws Exception {
        DescriptiveStatistics meanAccStat = new DescriptiveStatistics();
        DescriptiveStatistics meanCorrStat = new DescriptiveStatistics();
        DescriptiveStatistics correct = new DescriptiveStatistics();
        DescriptiveStatistics correct2 = new DescriptiveStatistics();
        DescriptiveStatistics corrStat = new DescriptiveStatistics();
        DescriptiveStatistics timeStat = new DescriptiveStatistics();
        //System.out.println("numOfClassifiers = " + numOfClassifiers);
        //System.out.println(dataset.numInstances());
        for (int k = 0; k < numFolds; ++k) {
            long time = System.currentTimeMillis();
            //System.out.println("k = " + k);
            Resample filter = new Resample();
            filter.setInputFormat(dataset);
            filter.setBiasToUniformClass(1);
            dataset.randomize(new Random());
            dataset = Filter.useFilter(dataset, filter);
            //dataset.stratify(2);
            train = dataset.trainCV(2, 0);

            train = Filter.useFilter(train, filter);
            test = dataset.testCV(2, 0);
            //train.randomize(new Random());
            /*train = Filter.useFilter(train, filter);
            test = Filter.useFilter(test, filter);*/
            //System.out.println(test.toSummaryString());
            //train.stratify(2);
            Instances subTrain = train.trainCV(2, 0);
            //System.out.println(subTrain.toSummaryString());
            Instances subTest = train.testCV(2, 0);
            /*System.out.println(subTest.toSummaryString());
            System.out.println(subTrain.classAttribute().toString());*/
            cls = new NaiveBayes();
            cls.buildClassifier(subTrain);

            Evaluation eval = new Evaluation(subTrain);
            eval.evaluateModel(cls, subTest);
            FastVector fv = eval.predictions();
        /*System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());*/
            Enumeration en = fv.elements();
            int n = subTest.numInstances();
            double[] origPredictions = new double[n];
            actual = new double[n];
            int idx = 0;
            while (en.hasMoreElements()) {
                Prediction pred = (Prediction)en.nextElement();
                origPredictions[idx] = pred.predicted();
                actual[idx++] = pred.actual();
                //System.out.println(pred.predicted() + " " + pred.actual());
            }
            //int numOfClassifiers = 33;
            int step = numOfClassifiers/2;
            labels = new double[numOfClassifiers][n];
            labels[0] = origPredictions;
            for (int i = 1; i < numOfClassifiers; ++i)  {
                labels[i] = new double[n];
            }
            int numClasses = train.numClasses();
            double[] classValues = new double[numClasses];
            for (int i = 0; i < numClasses; ++i) {
                classValues[i] = i;
            }
            EnhancerSA sa = new EnhancerSA(labels, actual, classValues, 0.01, 0.95, 1, 10, step, 1);
            //MultiClassEnhancerSA sa = new MultiClassEnhancerSA(labels, actual, classValues, 4, 0.995, 0.01, 1, step);
            double[][] ensemble = sa.start();
/*            for (int i = 0; i < ensemble.length; ++i) {
                System.out.println(Arrays.toString(ensemble[i]));
            }
            System.out.println(Arrays.toString(actual));*/
            PearsonsCorrelation pc = new PearsonsCorrelation();
            RealMatrix rm = new Array2DRowRealMatrix(labels);
            RealMatrix corr = pc.computeCorrelationMatrix(rm.transpose());
            double sum = 0;
            double count = 0;
            for (int i = 0; i < numOfClassifiers; ++i) {
                for (int j = i+1; j < numOfClassifiers; ++j) {
                    double c = corr.getEntry(i, j);
                    if (!Double.isNaN(c)) {
                        sum += Math.abs(c);
                        count++;
                    }

                }
            }
            if (count != 0) {
                sum /= count;
            }
            corrStat.addValue(sum);
            //System.out.println("sum of correlations = " + sum);
            Classifier[] classifiers = new Classifier[numOfClassifiers];
            classifiers[0] = cls;
            for (int i = 1; i < numOfClassifiers; ++i) {
                Instances falseLabelled = new Instances(subTest);
                double[] falseLabels = ensemble[i];
                for (int j = 0; j < falseLabels.length; ++j) {
                    falseLabelled.instance(j).setClassValue(falseLabels[j]);
                }
                Classifier falseCls = new NaiveBayes();
                falseCls.buildClassifier(falseLabelled);
            /*eval = new Evaluation(test);
            eval.evaluateModel(falseCls, test);
            System.out.println(eval.toSummaryString("\nResults (" + i + ")\n======\n", false));*/
                classifiers[i] = falseCls;
            }
        /*eval = new Evaluation(subTrain);
        eval.evaluateModel(cls, test);

        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());*/

            int inst = test.numInstances();
            double error = 0;
            double tp = 0, fp = 0, tn = 0, fn = 0;
            double tp2 = 0, fp2 = 0, tn2 = 0, fn2 = 0;
            for (int i = 0; i < inst; ++i) {
                Instance instance = test.instance(i);
                double orig = instance.classValue();
                double vote = 0;
                //double vote2 = classifiers[0].classifyInstance(instance);
                for (Classifier c: classifiers) {
                    vote += c.classifyInstance(instance);
                }
                if (vote > step) {
                    if (orig != 1.0) {
                        fp++;
                    }
                    else {
                        tp++;
                    }
                }
                else {
                    if (orig != 0.0) {
                        fn++;
                    }
                    else {
                        tn++;
                    }
                }
                /*if (vote2 == 1.0) {
                    if (orig != 1.0) {
                        fp2++;
                    }
                    else {
                        tp2++;
                    }
                }
                else {
                    if (orig != 0.0) {
                        fn2++;
                    }
                    else {
                        tn2++;
                    }
                }*/
            }
            double perf = (tp+tn)/inst;
            correct.addValue(perf);
            //correct2.addValue((tp2+tn2)/inst);
            timeStat.addValue(System.currentTimeMillis() - time);
            /*
            System.out.println(tp2 + " " + fn2);
            System.out.println(fp2 + " " + tn2);
            error = fp2+fn2;
            System.out.println(1.0 - (error/inst));


            System.out.println(tp + " " + fn);
            System.out.println(fp + " " + tn);
            error = fp+fn;
            System.out.println(1.0 - (error/inst));  */
        }
        System.out.println(correct.getMean());
        System.out.println(correct.getStandardDeviation());
        System.out.println(corrStat.getMean());
        System.out.println(timeStat.getMean());
       // System.out.println("corrStat = " + corrStat.toString());
        //System.out.println("---");
        /*System.out.println("correct2 = " + correct2);
        System.out.println("---");*/
        //System.out.println("correct = " + correct);
        /*System.out.println("---");
        System.out.println("timeStat = " + timeStat);*/
    }
    public void enhanceOnTestCVBalanced(int numFolds, int numOfClassifiers) throws Exception {
        DescriptiveStatistics meanAccStat = new DescriptiveStatistics();
        DescriptiveStatistics meanCorrStat = new DescriptiveStatistics();
        DescriptiveStatistics correct = new DescriptiveStatistics();
        DescriptiveStatistics correct2 = new DescriptiveStatistics();
        DescriptiveStatistics corrStat = new DescriptiveStatistics();
        DescriptiveStatistics timeStat = new DescriptiveStatistics();
        //System.out.println("numOfClassifiers = " + numOfClassifiers);
        for (int k = 0; k < numFolds; ++k) {
            long time = System.currentTimeMillis();
            //System.out.println("k = " + k);
            dataset.randomize(new Random());
/*            Resample filter = new Resample();
            filter.setInputFormat(dataset);
            filter.setBiasToUniformClass(1);

            dataset = Filter.useFilter(dataset, filter);
            //dataset.stratify(2);*/
            train = dataset.trainCV(2, 0);

            //train = Filter.useFilter(train, filter);
            test = dataset.testCV(2, 0);
            //train.randomize(new Random());
            /*train = Filter.useFilter(train, filter);
            test = Filter.useFilter(test, filter);*/
            //System.out.println(test.toSummaryString());
            //train.stratify(2);
            Instances subTrain = train.trainCV(2, 0);
            //System.out.println(subTrain.toSummaryString());
            Instances subTest = train.testCV(2, 0);
            /*System.out.println(subTest.toSummaryString());
            System.out.println(subTrain.classAttribute().toString());*/
            cls = new NaiveBayes();
            cls.buildClassifier(subTrain);
            double[] classDistributions = new double[dataset.numClasses()];
            Evaluation eval = new Evaluation(subTrain);
            eval.evaluateModel(cls, subTest);
            FastVector fv = eval.predictions();
        /*System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());*/
            Enumeration en = fv.elements();
            int n = subTest.numInstances();
            double[] origPredictions = new double[n];
            actual = new double[n];
            int idx = 0;
            while (en.hasMoreElements()) {
                Prediction pred = (Prediction)en.nextElement();
                origPredictions[idx] = pred.predicted();
                double classValue = pred.actual();
                actual[idx++] = classValue;
                classDistributions[(int)classValue]++;
                //System.out.println(pred.predicted() + " " + pred.actual());
            }
            //int numOfClassifiers = 33;
            int step = numOfClassifiers/2;
            labels = new double[numOfClassifiers][n];
            labels[0] = origPredictions;
            for (int i = 1; i < numOfClassifiers; ++i)  {
                labels[i] = new double[n];
            }
            int numClasses = train.numClasses();
            double[] classValues = new double[numClasses];
            for (int i = 0; i < numClasses; ++i) {
                classValues[i] = i;
            }
            //EnhancerSA sa = new EnhancerSA(labels, actual, classValues, 0.01, 0.95, 1, 10, step);
            UnbalancedEnhancerSA sa = new UnbalancedEnhancerSA(labels, actual, classValues, classDistributions, 10, step, 0.01, 0.95, 1);
            double[][] ensemble = sa.start();
            /*for (int i = 0; i < ensemble.length; ++i) {
                System.out.println(Arrays.toString(ensemble[i]));
            }
            System.out.println(Arrays.toString(actual));*/
            PearsonsCorrelation pc = new PearsonsCorrelation();
            RealMatrix rm = new Array2DRowRealMatrix(labels);
            RealMatrix corr = pc.computeCorrelationMatrix(rm.transpose());
            double sum = 0;
            double count = 0;
            for (int i = 0; i < numOfClassifiers; ++i) {
                for (int j = i+1; j < numOfClassifiers; ++j) {
                    double c = corr.getEntry(i, j);
                    if (!Double.isNaN(c)) {
                        sum += Math.abs(c);
                        count++;
                    }

                }
            }
            if (count != 0) {
                sum /= count;
            }
            corrStat.addValue(sum);
            //System.out.println("sum of correlations = " + sum);
            Classifier[] classifiers = new Classifier[numOfClassifiers];
            classifiers[0] = cls;
            for (int i = 1; i < numOfClassifiers; ++i) {
                Instances falseLabelled = new Instances(subTest);
                double[] falseLabels = ensemble[i];
                for (int j = 0; j < falseLabels.length; ++j) {
                    falseLabelled.instance(j).setClassValue(falseLabels[j]);
                }
                Classifier falseCls = new NaiveBayes();
                falseCls.buildClassifier(falseLabelled);
            /*eval = new Evaluation(test);
            eval.evaluateModel(falseCls, test);
            System.out.println(eval.toSummaryString("\nResults (" + i + ")\n======\n", false));*/
                classifiers[i] = falseCls;
            }
        /*eval = new Evaluation(subTrain);
        eval.evaluateModel(cls, test);

        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());*/

            int inst = test.numInstances();
            double error = 0;
            double tp = 0, fp = 0, tn = 0, fn = 0;
            double tp2 = 0, fp2 = 0, tn2 = 0, fn2 = 0;
            for (int i = 0; i < inst; ++i) {
                Instance instance = test.instance(i);
                double orig = instance.classValue();
                double vote = 0;
                //double vote2 = classifiers[0].classifyInstance(instance);
                for (Classifier c: classifiers) {
                    vote += c.classifyInstance(instance);
                }
                if (vote > step) {
                    if (orig != 1.0) {
                        fp++;
                    }
                    else {
                        tp++;
                    }
                }
                else {
                    if (orig != 0.0) {
                        fn++;
                    }
                    else {
                        tn++;
                    }
                }
                /*if (vote2 == 1.0) {
                    if (orig != 1.0) {
                        fp2++;
                    }
                    else {
                        tp2++;
                    }
                }
                else {
                    if (orig != 0.0) {
                        fn2++;
                    }
                    else {
                        tn2++;
                    }
                }*/
            }
            double perf = (tp+tn)/inst;
            correct.addValue(perf);
            //correct2.addValue((tp2+tn2)/inst);
            timeStat.addValue(System.currentTimeMillis() - time);
            /*
            System.out.println(tp2 + " " + fn2);
            System.out.println(fp2 + " " + tn2);
            error = fp2+fn2;
            System.out.println(1.0 - (error/inst));


            System.out.println(tp + " " + fn);
            System.out.println(fp + " " + tn);
            error = fp+fn;
            System.out.println(1.0 - (error/inst));  */
        }
        System.out.println(correct.getMean());
        System.out.println(correct.getStandardDeviation());
        System.out.println(corrStat.getMean());
        System.out.println(timeStat.getMean());
        // System.out.println("corrStat = " + corrStat.toString());
        //System.out.println("---");
        /*System.out.println("correct2 = " + correct2);
        System.out.println("---");*/
        //System.out.println("correct = " + correct);
        /*System.out.println("---");
        System.out.println("timeStat = " + timeStat);*/
    }

    public void enhanceOnTrain() throws Exception {
        Resample filter = new Resample();
        filter.setInputFormat(dataset);
        filter.setBiasToUniformClass(1);
        dataset.randomize(new Random());
        dataset = Filter.useFilter(dataset, filter);
        //dataset.randomize(new Random());
        train = dataset.trainCV(2, 0);
        test = dataset.testCV(2, 0);
        //System.out.println(test.toSummaryString());
        Instances subTrain = train.trainCV(2, 0);
        //System.out.println(subTrain.toSummaryString());
        Instances subTest = train.testCV(2, 0);
        //System.out.println(subTest.toSummaryString());
        //System.out.println(subTrain.classAttribute().toString());
        //cls.buildClassifier(train);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(cls, test);
        FastVector fv = eval.predictions();
        /*System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());*/
        /*Enumeration en = fv.elements();*/
        int n = train.numInstances();
        actual = new double[n];
        int idx = 0;
        //double[] origPredictions = new double[n];
        for (int i = 0; i < n; ++i) {
            actual[idx++] = train.instance(i).classValue();
        }

        /*while (en.hasMoreElements()) {
            Prediction pred = (Prediction)en.nextElement();
            origPredictions[idx] = pred.predicted();
            actual[idx++] = pred.actual();
            //System.out.println(pred.predicted() + " " + pred.actual());
        }*/
        int numOfClassifiers = 5;
        int step = numOfClassifiers/2;
        labels = new double[numOfClassifiers][n];
        for (int i = 0; i < numOfClassifiers; ++i)  {
            labels[i] = new double[n];
        }
        int numClasses = train.numClasses();
        double[] classValues = new double[numClasses];
        for (int i = 0; i < numClasses; ++i) {
            classValues[i] = i;
        }
        EnhancerSA sa = new EnhancerSA(labels, actual, classValues, 0.01, 0.95, 4, 5, step, 1);
        double[][] ensemble = sa.start();
        /*for (int i = 0; i < ensemble.length; ++i) {
            System.out.println(Arrays.toString(ensemble[i]));
        }*/
        System.out.println(Arrays.toString(actual));
        PearsonsCorrelation pc = new PearsonsCorrelation();
        RealMatrix corr = pc.computeCorrelationMatrix(labels);
        double sum = 0;
        double count = 0;
        for (int i = 0; i < numOfClassifiers; ++i) {
            for (int j = i+1; j < numOfClassifiers; ++j) {
                sum += Math.abs(corr.getEntry(i, j));
                count++;
            }
        }
        sum /= count;
        System.out.println("sum of correlations = " + sum);
        Classifier[] classifiers = new Classifier[numOfClassifiers];
        //classifiers[0] = cls;
        for (int i = 0; i < numOfClassifiers; ++i) {
            Instances falseLabelled = new Instances(train);
            double[] falseLabels = ensemble[i];
            for (int j = 0; j < falseLabels.length; ++j) {
                falseLabelled.instance(j).setClassValue(falseLabels[j]);
            }
            Classifier falseCls = new MultilayerPerceptron();
            falseCls.buildClassifier(falseLabelled);
            /*eval = new Evaluation(test);
            eval.evaluateModel(falseCls, test);
            System.out.println(eval.toSummaryString("\nResults (" + i + ")\n======\n", false));*/
            classifiers[i] = falseCls;
        }
        /*eval = new Evaluation(subTrain);
        eval.evaluateModel(cls, test);

        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());*/
        int inst = test.numInstances();
        double error = 0;
        int tp = 0, fp = 0, tn = 0, fn = 0;
        int tp2 = 0, fp2 = 0, tn2 = 0, fn2 = 0;
        for (int i = 0; i < inst; ++i) {
            Instance instance = test.instance(i);
            double orig = instance.classValue();
            double vote = 0;
            double vote2 = classifiers[0].classifyInstance(instance);
            for (Classifier c: classifiers) {
                vote += c.classifyInstance(instance);
            }
            if (vote > step) {
                if (orig != 1.0) {
                    fp++;
                }
                else {
                    tp++;
                }
            }
            else {
                if (orig != 0.0) {
                    fn++;
                }
                else {
                    tn++;
                }
            }
            /*if (vote2 == 1.0) {
                if (orig != 1.0) {
                    fp2++;
                }
                else {
                    tp2++;
                }
            }
            else {
                if (orig != 0.0) {
                    fn2++;
                }
                else {
                    tn2++;
                }
            }*/
        }
        /*System.out.println(tp2 + " " + fn2);
        System.out.println(fp2 + " " + tn2);
        error = fp2+fn2;
        System.out.println(1.0 - (error/inst));*/

        System.out.println(tp + " " + fn);
        System.out.println(fp + " " + tn);
        error = fp+fn;
        System.out.println(1.0 - (error/inst));


    }

    public void enhanceOnTrainCV(int numFolds, int numOfClassifiers) throws Exception {
        DescriptiveStatistics meanAccStat = new DescriptiveStatistics();
        DescriptiveStatistics meanCorrStat = new DescriptiveStatistics();
        DescriptiveStatistics correct = new DescriptiveStatistics();
        DescriptiveStatistics correct2 = new DescriptiveStatistics();
        DescriptiveStatistics corrStat = new DescriptiveStatistics();
        DescriptiveStatistics timeStat = new DescriptiveStatistics();
        //System.out.println("numOfClassifiers = " + numOfClassifiers);

        for (int k = 0; k < numFolds; ++k) {
            long time = System.currentTimeMillis();
            //System.out.println("k = " + k);
            Resample filter = new Resample();
            filter.setInputFormat(dataset);
            filter.setBiasToUniformClass(1);
            dataset.randomize(new Random());
            dataset = Filter.useFilter(dataset, filter);
            //dataset.randomize(new Random());
            train = dataset.trainCV(2, 0);
            test = dataset.testCV(2, 0);
            //System.out.println(test.toSummaryString());
            //Instances subTrain = train.trainCV(2, 0);
            //System.out.println(subTrain.toSummaryString());
            //Instances subTest = train.testCV(2, 0);
            /*System.out.println(subTest.toSummaryString());
            System.out.println(subTrain.classAttribute().toString());*/
            /*cls = new NaiveBayes();
            cls.buildClassifier(subTrain);

            Evaluation eval = new Evaluation(subTrain);
            eval.evaluateModel(cls, subTest);
            FastVector fv = eval.predictions();*/
        /*System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());*/
            /*Enumeration en = fv.elements();
            int n = subTest.numInstances();
            double[] origPredictions = new double[n];
            actual = new double[n];
            int idx = 0;
            while (en.hasMoreElements()) {
                Prediction pred = (Prediction)en.nextElement();
                origPredictions[idx] = pred.predicted();
                actual[idx++] = pred.actual();
                //System.out.println(pred.predicted() + " " + pred.actual());
            }*/
            //int numOfClassifiers = 33;
            int n = train.numInstances();
            actual = new double[n];
            int idx = 0;
            //double[] origPredictions = new double[n];
            for (int i = 0; i < n; ++i) {
                actual[idx++] = train.instance(i).classValue();
            }
            int step = numOfClassifiers/2;
            labels = new double[numOfClassifiers][n];
            labels[0] = actual;
            for (int i = 1; i < numOfClassifiers; ++i)  {
                labels[i] = new double[n];
            }
            int numClasses = train.numClasses();
            double[] classValues = new double[numClasses];
            for (int i = 0; i < numClasses; ++i) {
                classValues[i] = i;
            }

            EnhancerSA sa = new EnhancerSA(labels, actual, classValues, 0.01, 0.95, 4, 1, step, 1);
            //Classifier updateable = new NNge();
            FastVector attributeList = new FastVector();
            attributeList.addElement(new Attribute("horizontalEnergy"));
            attributeList.addElement(new Attribute("verticalEnergy"));
            //attributeList.addElement(new Attribute("majorityEnergy"));
            //attributeList.addElement(new Attribute("smoothness"));
            //attributeList.addElement(new Attribute("classEnergy"));
            attributeList.addElement(new Attribute("class2Energy"));
            FastVector classValuesFV = new FastVector();
            classValuesFV.addElement("0");
            classValuesFV.addElement("1");
            attributeList.addElement(new Attribute("classValue", classValuesFV));
            Instances instances = new Instances("dec", attributeList, 100);
            instances.setClassIndex(instances.numAttributes()-1);
            //updateable.buildClassifier(instances);
            double[][] ensemble = sa.start();
            /*for (int i = 0; i < ensemble.length; ++i) {
                System.out.println(Arrays.toString(ensemble[i]));
            }
            System.out.println(Arrays.toString(actual));*/
            PearsonsCorrelation pc = new PearsonsCorrelation();
            RealMatrix rm = new Array2DRowRealMatrix(labels);
            RealMatrix corr = pc.computeCorrelationMatrix(rm.transpose());
            double sum = 0;
            double count = 0;
            for (int i = 0; i < numOfClassifiers; ++i) {
                for (int j = i+1; j < numOfClassifiers; ++j) {
                    double c = corr.getEntry(i, j);
                    if (!Double.isNaN(c)) {
                        sum += Math.abs(c);
                        count++;
                    }

                }
            }
            if (count != 0) {
                sum /= count;
            }
            corrStat.addValue(sum);
            //System.out.println("sum of correlations = " + sum);
            Classifier[] classifiers = new Classifier[numOfClassifiers];
            cls.buildClassifier(train);
            classifiers[0] = cls;
            for (int i = 1; i < numOfClassifiers; ++i) {
                Instances falseLabelled = new Instances(train);
                double[] falseLabels = ensemble[i];
                for (int j = 0; j < falseLabels.length; ++j) {
                    falseLabelled.instance(j).setClassValue(falseLabels[j]);
                }
                Classifier falseCls = new NaiveBayes();
                falseCls.buildClassifier(falseLabelled);
            /*eval = new Evaluation(test);
            eval.evaluateModel(falseCls, test);
            System.out.println(eval.toSummaryString("\nResults (" + i + ")\n======\n", false));*/
                classifiers[i] = falseCls;
            }
        /*eval = new Evaluation(subTrain);
        eval.evaluateModel(cls, test);

        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());*/

            int inst = test.numInstances();
            double error = 0;
            double tp = 0, fp = 0, tn = 0, fn = 0;
            double tp2 = 0, fp2 = 0, tn2 = 0, fn2 = 0;
            for (int i = 0; i < inst; ++i) {
                Instance instance = test.instance(i);
                double orig = instance.classValue();
                double vote = 0;
                //double vote2 = classifiers[0].classifyInstance(instance);
                for (Classifier c: classifiers) {
                    vote += c.classifyInstance(instance);
                }
                if (vote > step) {
                    if (orig != 1.0) {
                        fp++;
                    }
                    else {
                        tp++;
                    }
                }
                else {
                    if (orig != 0.0) {
                        fn++;
                    }
                    else {
                        tn++;
                    }
                }
                /*if (vote2 == 1.0) {
                    if (orig != 1.0) {
                        fp2++;
                    }
                    else {
                        tp2++;
                    }
                }
                else {
                    if (orig != 0.0) {
                        fn2++;
                    }
                    else {
                        tn2++;
                    }
                }*/
            }
            correct.addValue((tp+tn)/inst);
            //correct2.addValue((tp2+tn2)/inst);
            timeStat.addValue(System.currentTimeMillis() - time);
            /*
            System.out.println(tp2 + " " + fn2);
            System.out.println(fp2 + " " + tn2);
            error = fp2+fn2;
            System.out.println(1.0 - (error/inst));


            System.out.println(tp + " " + fn);
            System.out.println(fp + " " + tn);
            error = fp+fn;
            System.out.println(1.0 - (error/inst));  */
        }
        System.out.println(correct.getMean());
        System.out.println(correct.getStandardDeviation());
        System.out.println(corrStat.getMean());
        System.out.println(timeStat.getMean());
        /*System.out.println("corrStat = " + corrStat.toString());
        System.out.println("---");
        *//*System.out.println("correct2 = " + correct2);
        System.out.println("---");*//*
        System.out.println("correct = " + correct);*/
        /*System.out.println("---");
        System.out.println("timeStat = " + timeStat);*/
    }

    public static void main(String[] args) {
        //String train = "/home/antalb/datasets/UCI/breast-cancer.arff";

        //String train = "/home/antalb/datasets/UCI/diabetes.arff";
        //String test = "/home/antalb/datasetes/UCI/poker-hand-testing.arff";
        //String train = "/home/antalb/datasets/UCI/colic.arff";
        //String train = "/home/antalb/datasets/keng/4/DLBCL-Stanford.arff";
        String base = "/home/antalb/datasets/keng/2/";
        File dir = new File(base);
        String[] names = dir.list();
        //Classifier[] classifiers = {new NaiveBayes(), new AdaBoostM1(), new Bagging(), new RandomForest()};
        for (String train: names) {
            //train = base + train;
            System.out.println(train );
            Classifier cls = new NaiveBayes();
            try {
                Instances trainData = DatasetUtil.loadDataset(base + train);
                trainData.setClassIndex(trainData.numAttributes() - 1);
                System.out.println(trainData.numAttributes());
                //Enhancer enhancer = new Enhancer(cls, DatasetUtil.convertClassType(trainData));
                //System.out.println(trainData.toSummaryString());
                //Instances testData = DatasetUtil.loadDataset(test);
                //testData.setClassIndex(testData.numAttributes()-1);
                /*for (Classifier c: classifiers) {

                    int numFolds = 5;
                    enhancer.test(numFolds, c);

                }*/
                //System.out.println("\\\\");
                /*for (int i = 3; i < 17; i+=2) {
                    enhancer.enhanceOnTrainCV(100, i);
                }*/


            } catch (Exception e) {
            /*int step = trainData.numClasses();
            int start = step+1;

            int end = start + 10*step + 1;
            for (int i = start; i < end; i+=step) {
                System.out.println("i = " + i);
                enhancer.enhanceOnTestCV(numFolds, i);
            }*/
                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }
        }
        base = "/home/antalb/datasets/keng/3/";
        dir = new File(base);
        names = dir.list();
        for (String train: names) {
            train = base + train;
            System.out.println("train = " + train);
            Classifier cls = new NaiveBayes();
            try {
                Instances trainData = DatasetUtil.loadDataset(train);
                trainData.setClassIndex(trainData.numAttributes() - 1);
                System.out.println(trainData.numAttributes());
                //System.out.println(trainData.toSummaryString());
                //Instances testData = DatasetUtil.loadDataset(test);
                //testData.setClassIndex(testData.numAttributes()-1);

                /*Enhancer enhancer = new Enhancer(cls, DatasetUtil.convertClassType(trainData));
                *//*int numFolds = 5;
                enhancer.crossValidate(numFolds);*//*
                for (int i = 3; i < 17; i+=2) {
                    enhancer.enhanceOnTrainCV(10, i);
                }*/


            } catch (Exception e) {
                            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }
        }
    }
}
