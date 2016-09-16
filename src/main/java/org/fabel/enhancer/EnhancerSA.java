package org.fabel.enhancer;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.rules.NNge;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: antalb
 * Date: 7/10/13
 * Time: 2:48 PM
 * To change this template use File | Settings | File Templates.
 */
public class EnhancerSA {

    private int numClassifiers;
    private int numInstances;
    private int optimalVotes;
    private int numberOfVotes;
    private double temperature;
    private double tempChange;
    private double energyThreshold;
    private int numClasses;
    private double[][] labels;
    private double[] actual;
    private double[] classByIndex;
    private double penalty;
    private int step;
    private Random random;

    public EnhancerSA(double[][] labels, double[] actual, double[] classByIndex, double energyThreshold, double tempChange, double temperature) {
        this.labels = labels;
        this.actual = actual;
        this.classByIndex = classByIndex;
        this.energyThreshold = energyThreshold;
        this.tempChange = tempChange;
        this.temperature = temperature;
        this.numClassifiers = labels.length;
        this.numInstances = actual.length;
        this.optimalVotes = (numClassifiers+1)/2;
        this.numClasses = classByIndex.length;
        for (int i = 0; i < labels[0].length; ++i) {
            numberOfVotes += labels[0][i] != 0 ? 1 : 0;
        }
    }

    public EnhancerSA(double[][] labels, double[] actual, double[] classByIndex, double energyThreshold, double tempChange, double temperature, double penalty, int step, Random random) {
        this(labels, actual, classByIndex, energyThreshold, tempChange, temperature);
        this.penalty = penalty;
        this.step = step;
        this.random = random;
    }

    public double getHorizontalEnergy(int n) {
        double sum = 0;
        for (int i = 0; i < numClassifiers; ++i) {
            sum += labels[i][n];
        }
        if (actual[n] == 1) {
            return Math.abs(sum - optimalVotes)  / (double)optimalVotes;
        }
        return Math.abs(sum - numClassifiers + optimalVotes)  / (double)optimalVotes;
    }

    public double getVerticalEnergy(int m) {
        double sum = 0;
        for (int i = 0; i < numInstances; ++i) {
            sum += labels[m][i];
        }
        return Math.abs(sum - numberOfVotes) / (double)numberOfVotes;
    }

    public double getCorrelationError() {
        PearsonsCorrelation pc = new PearsonsCorrelation();
        RealMatrix corr = pc.computeCorrelationMatrix(labels);
        double sum = 0;
        for (int i = 0; i < numClassifiers; ++i) {
            for (int j = i+1; j < numClassifiers; ++j) {
                sum += corr.getEntry(i, j);
            }
        }
        return sum;
    }

    public double getClassificationError(int n) {
        double sum = 0;
        for (int i = 0; i < numClassifiers; ++i) {
            sum += labels[i][n];
        }

        if (sum >= optimalVotes) {
            sum = 1.0;
        }
        else {
            sum = 0.0;
        }
        return Math.abs(sum - actual[n]);
    }

    public double getClassificationError(int m, double target) {
        double sum = 0;
        for (int i = 0; i < numInstances; ++i) {
            sum += getClassificationError(m, i);
        }
        sum /= (double)numInstances;
        return sum - target;
    }

    public double getClassificationError(int m, int n) {
        return Math.abs(labels[m][n] - actual[n]);
    }

    public double getSmoothness(int m, int step) {
        double sum = 0;
        double count = 0;
        int start = Math.max(m-step, 0), end = Math.min(m+step+1, numInstances);
        for (int i = 0; i < numClassifiers; ++i) {

            for (int j = start; j < end; ++j) {
                double value = labels[m][j];
                if (i != m) {
                    count++;
                    if (value == labels[i][j]) {
                        sum += penalty;
                    }
                    else {
                        sum -= penalty;
                    }
                }
            }
        }
        return sum / count;
    }

    public double getMajorityEnergy(int m, int n, double energy) {
        double value = labels[m][n];
        if (value == actual[n]) {
            double sum = 0;
            for (int i = 0; i < numClassifiers; ++i) {
                if (i != m && labels[i][n] == actual[n]) {
                    sum++;
                }
            }
            if (sum < optimalVotes) {
                return -energy;
            }
        }
        return energy;
    }

    public double getDisagreement(int m, int n) {
        double value = labels[m][n];
        double disagreement = 0;
        for (int i = 0; i < numClassifiers; ++i) {
            if (i != m && labels[i][n] != value) {
                disagreement++;
            }
        }

        return -disagreement / (double)numClassifiers;
    }

    public double getEnergy(int m, int n) {
        return getHorizontalEnergy(n) + /*getVerticalEnergy(m) + getClassificationError(m, 0.1) + getSmoothness(m, 30)+ getClassificationError(m, n) /* + getMajorityEnergy(m, n, 1) + getClassificationError(n)*/getSmoothness(m, numClassifiers)+getDisagreement(m, n);
    }

    public double getEnergy(int m, int n, double target) {
        return getHorizontalEnergy(n) /*+ getVerticalEnergy(m)*/ + getClassificationError(m, n)  + getSmoothness(m, step) + getMajorityEnergy(m, n, 1) + getClassificationError(n);
    }

    public double getEnergy() {
        double sum = 0;
        for (int i = 0; i < numClassifiers; ++i) {
            for (int j = 0; j < numInstances; ++j) {
                sum += getEnergy(i, j);
            }
        }
        return sum;
    }

    public double getEnergy(double target) {
        double sum = 0;
        for (int i = 0; i < numClassifiers; ++i) {
            for (int j = 0; j < numInstances; ++j) {
                sum += getEnergy(i, j, target);
            }
        }
        return sum;
    }

    public double[][] start() {
        double energy = getEnergy();
        double delta = 0;
        int k = 0;
        double avgCorr = 0;
        do {
            //System.out.println(k + ": " + temperature + " " + energy + " " + delta);
            delta = 0;
            for (int i = 0; i < numClassifiers; ++i) {
                for (int j = 0; j < numInstances; ++j) {
                    double value = labels[i][j];

                    double oldEnergy = getEnergy(i, j);

                    if (numClasses == 2) {
                        labels[i][j] = 1.0 - labels[i][j];
                    }
                    else {
//                        Random random = new Random(seed);
                        double newValue = classByIndex[random.nextInt(numClasses)];
                        while (value == newValue) {
                            newValue = classByIndex[random.nextInt(numClasses)];
                        }
                        labels[i][j] = newValue;
                    }
                    double horizontalEnergy = getHorizontalEnergy(j);
                    double verticalEnergy = getVerticalEnergy(i);
                    double majorityEnergy = getMajorityEnergy(i, j, 1);
                    double smoothness = getSmoothness(i, 30);
                    double classEnergy = getClassificationError(i, j);
                    double newEnergy = getEnergy(i, j);
                    double diff = newEnergy - oldEnergy;
                    if (diff > 0) {
                        Random random = new Random();
                        double rnd = random.nextDouble();
                        double d = Math.exp(-diff / temperature);
                        if (rnd >= d) {
                            labels[i][j] = value;
                            continue;
                        }

                    }
                    delta += Math.abs(diff);
                    energy -= Math.abs(diff);
                }
            }
             k++;
             temperature *= tempChange;
            /*PearsonsCorrelation pc = new PearsonsCorrelation();
            RealMatrix rm = new Array2DRowRealMatrix(labels);
            RealMatrix corr = pc.computeCorrelationMatrix(rm.transpose());

            double count = 0;
            for (int i = 0; i < numClassifiers; ++i) {
                for (int j = i+1; j < numClassifiers; ++j) {
                    double c = corr.getEntry(i, j);
                    if (!Double.isNaN(c)) {
                        avgCorr += Math.abs(c);
                        count++;
                    }

                }
            }
            if (count != 0) {
                avgCorr /= count;
            }   */
        }
        while (delta > energyThreshold/* && avgCorr > 0.3*/);
        return labels;

    }

    public double[][] start(double target) {
        double energy = getEnergy(target);
        double delta = 0;
        int k = 0;
        do {
            System.out.println(k + ": " + temperature + " " + energy + " " + delta);
            delta = 0;
            for (int i = 1; i < numClassifiers; ++i) {
                for (int j = 0; j < numInstances; ++j) {
                    double value = labels[i][j];
                    double oldEnergy = getEnergy(i, j, target);
                    if (numClasses == 2) {
                        labels[i][j] = 1.0 - labels[i][j];
                    }
                    else {
//                        Random random = new Random(seed);
                        double newValue = classByIndex[random.nextInt(numClasses)];
                        while (value == newValue) {
                            newValue = classByIndex[random.nextInt(numClasses)];
                        }
                        labels[i][j] = newValue;
                    }
                    double horizontalEnergy = getHorizontalEnergy(j);
                    double verticalEnergy = getVerticalEnergy(i);
                    double majorityEnergy = getMajorityEnergy(i, j, 1);
                    double smoothness = getSmoothness(i, step);
                    double classEnergy = getClassificationError(i, j);
                    double class2Energy = getClassificationError(i,  target);
                    double newEnergy = getEnergy(i, j, target);
                    double diff = newEnergy - oldEnergy;
                    if (diff > 0) {
                        Random random = new Random();
                        double rnd = random.nextDouble();
                        double d = Math.exp(-diff / temperature);
                        if (rnd >= d) {
                            labels[i][j] = value;
                            continue;
                        }

                    }
                    delta += Math.abs(diff);
                    energy -= Math.abs(diff);
                }
            }
            k++;
            temperature *= tempChange;

        }
        while (delta > energyThreshold);
        return labels;

    }

    public double[][] start(double target, double corrTh) {
        double energy = getEnergy(target);
        double delta = 0;
        int k = 0;
        double avgCorr = 0;
        do {
            //System.out.println(k + ": " + temperature + " " + energy + " " + delta);
            delta = 0;
            for (int i = 1; i < numClassifiers; ++i) {
                for (int j = 0; j < numInstances; ++j) {
                    double value = labels[i][j];
                    double oldEnergy = getEnergy(i, j, target);
                    if (numClasses == 2) {
                        labels[i][j] = 1.0 - labels[i][j];
                    }
                    else {
//                        Random random = new Random();
                        double newValue = classByIndex[random.nextInt(numClasses)];
                        while (value == newValue) {
                            newValue = classByIndex[random.nextInt(numClasses)];
                        }
                        labels[i][j] = newValue;
                    }
                    double horizontalEnergy = getHorizontalEnergy(j);
                    double verticalEnergy = getVerticalEnergy(i);
                    double majorityEnergy = getMajorityEnergy(i, j, 1);
                    double smoothness = getSmoothness(i, step);
                    double classEnergy = getClassificationError(i, j);
                    double class2Energy = getClassificationError(i,  target);
                    double newEnergy = getEnergy(i, j, target);
                    double diff = newEnergy - oldEnergy;
                    if (diff > 0) {
                        Random random = new Random();
                        double rnd = random.nextDouble();
                        double d = Math.exp(-diff / temperature);
                        if (rnd >= d) {
                            labels[i][j] = value;
                            continue;
                        }

                    }
                    delta += Math.abs(diff);
                    energy -= Math.abs(diff);
                }
            }
            k++;
            temperature *= tempChange;
            PearsonsCorrelation pc = new PearsonsCorrelation();
            RealMatrix rm = new Array2DRowRealMatrix(labels);
            RealMatrix corr = pc.computeCorrelationMatrix(rm.transpose());
            double count = 0;
            for (int i = 0; i < numClassifiers; ++i) {
                for (int j = i+1; j < numClassifiers; ++j) {
                    double c = corr.getEntry(i, j);
                    if (!Double.isNaN(c)) {
                        avgCorr += Math.abs(c);
                        count++;
                    }

                }
            }
            if (count != 0) {
                avgCorr /= count;
            }
        }
        while (delta > energyThreshold && avgCorr > corrTh);
        return labels;

    }

    public double[][] startOnline(double target) throws Exception {
        //NaiveBayesUpdateable updateable = new NaiveBayesUpdateable();
        //RacedIncrementalLogitBoost updateable = new RacedIncrementalLogitBoost();
        NNge updateable = new NNge();
        FastVector attributeList = new FastVector();
        attributeList.addElement(new Attribute("horizontalEnergy"));
        attributeList.addElement(new Attribute("verticalEnergy"));
        //attributeList.addElement(new Attribute("majorityEnergy"));
        //attributeList.addElement(new Attribute("smoothness"));
        //attributeList.addElement(new Attribute("classEnergy"));
        attributeList.addElement(new Attribute("class2Energy"));
        FastVector classValues = new FastVector();
        classValues.addElement("0");
        classValues.addElement("1");
        attributeList.addElement(new Attribute("classValue", classValues));
        Instances instances = new Instances("dec", attributeList, 100);
        instances.setClassIndex(instances.numAttributes()-1);
        updateable.buildClassifier(instances);
        double energy = getEnergy(target);
        double delta = 0;
        int k = 0;
        double avgCorr = 0;
        do {
            //System.out.println(k + ": " + temperature + " " + energy + " " + delta);
            delta = 0;
            for (int i = 1; i < numClassifiers; ++i) {
                for (int j = 0; j < numInstances; ++j) {
                    double value = labels[i][j];
                    double oldEnergy = getEnergy(i, j, target);
                    if (numClasses == 2) {
                        labels[i][j] = 1.0 - labels[i][j];
                    }
                    else {
//                        Random random = new Random();
                        double newValue = classByIndex[random.nextInt(numClasses)];
                        while (value == newValue) {
                            newValue = classByIndex[random.nextInt(numClasses)];
                        }
                        labels[i][j] = newValue;
                    }
                    double horizontalEnergy = getHorizontalEnergy(j);
                    double verticalEnergy = getVerticalEnergy(i);
                    //double majorityEnergy = getMajorityEnergy(i, j, 1);
                    double smoothness = getSmoothness(i, step);
                    double classEnergy = getClassificationError(i, j);
                    double class2Energy = getClassificationError(i, target);
                    Instance instance = new Instance(1.0, new double[]{horizontalEnergy, verticalEnergy, /*majorityEnergy, smoothness, classEnergy,*/ class2Energy, actual[j]});
                    instance.setDataset(instances);
                    updateable.updateClassifier(instance);
                    labels[i][j] = updateable.classifyInstance(instance);
                    double newEnergy = getEnergy(i, j, target);
                    double diff = newEnergy - oldEnergy;
                    if (diff > 0) {
                        /*labels[i][j] = value;
                        continue;*/
                        Random random = new Random();
                        double rnd = random.nextDouble();
                        double d = Math.exp(-diff / temperature);
                        if (rnd >= d) {
                            labels[i][j] = value;
                            continue;
                        }

                    }
                    delta += Math.abs(diff);
                    energy -= Math.abs(diff);
                }
            }
            k++;
            temperature *= tempChange;
            /*PearsonsCorrelation pc = new PearsonsCorrelation();
            RealMatrix rm = new Array2DRowRealMatrix(labels);
            RealMatrix corr = pc.computeCorrelationMatrix(rm.transpose());

            double count = 0;
            for (int i = 0; i < numClassifiers; ++i) {
                for (int j = i+1; j < numClassifiers; ++j) {
                    double c = corr.getEntry(i, j);
                    if (!Double.isNaN(c)) {
                        avgCorr += Math.abs(c);
                        count++;
                    }

                }
            }
            if (count != 0) {
                avgCorr /= count;
            }*/
        }
        while (delta > energyThreshold);
        //System.out.println("updateable = " + updateable.toString());
        return labels;

    }

    public double[][] startOnline(double target, Classifier updateable, Instances instances) throws Exception {
        //NaiveBayesUpdateable updateable = new NaiveBayesUpdateable();
        //RacedIncrementalLogitBoost updateable = new RacedIncrementalLogitBoost();

        double energy = getEnergy(target);
        double delta = 0;
        int k = 0;
        double avgCorr = 0;
        do {
            //System.out.println(k + ": " + temperature + " " + energy + " " + delta);
            delta = 0;
            for (int i = 1; i < numClassifiers; ++i) {
                for (int j = 0; j < numInstances; ++j) {
                    double value = labels[i][j];
                    double oldEnergy = getEnergy(i, j, target);
                    if (numClasses == 2) {
                        labels[i][j] = 1.0 - labels[i][j];
                    }
                    else {
//                        Random random = new Random();
                        double newValue = classByIndex[random.nextInt(numClasses)];
                        while (value == newValue) {
                            newValue = classByIndex[random.nextInt(numClasses)];
                        }
                        labels[i][j] = newValue;
                    }
                    double horizontalEnergy = getHorizontalEnergy(j);
                    double verticalEnergy = getVerticalEnergy(i);
                    //double majorityEnergy = getMajorityEnergy(i, j, 1);
                    double smoothness = getSmoothness(i, step);
                    double classEnergy = getClassificationError(i, j);
                    double class2Energy = getClassificationError(i, target);
                    Instance instance = new Instance(1.0, new double[]{horizontalEnergy, verticalEnergy, /*majorityEnergy, smoothness, classEnergy,*/ class2Energy, actual[j]});
                    instance.setDataset(instances);
                    ((UpdateableClassifier)updateable).updateClassifier(instance);
                    labels[i][j] = updateable.classifyInstance(instance);
                    double newEnergy = getEnergy(i, j, target);
                    double diff = newEnergy - oldEnergy;
                    if (diff > 0) {
                        /*labels[i][j] = value;
                        continue;*/
                        Random random = new Random();
                        double rnd = random.nextDouble();
                        double d = Math.exp(-diff / temperature);
                        if (rnd >= d) {
                            labels[i][j] = value;
                            continue;
                        }

                    }
                    delta += Math.abs(diff);
                    energy -= Math.abs(diff);
                }
            }
            k++;
            temperature *= tempChange;
            PearsonsCorrelation pc = new PearsonsCorrelation();
            RealMatrix rm = new Array2DRowRealMatrix(labels);
            RealMatrix corr = pc.computeCorrelationMatrix(rm.transpose());

            double count = 0;
            for (int i = 0; i < numClassifiers; ++i) {
                for (int j = i+1; j < numClassifiers; ++j) {
                    double c = corr.getEntry(i, j);
                    if (!Double.isNaN(c)) {
                        avgCorr += Math.abs(c);
                        count++;
                    }

                }
            }
            if (count != 0) {
                avgCorr /= count;
            }
        }
        while (delta > energyThreshold);
        //System.out.println("updateable = " + updateable.toString());
        return labels;

    }


}
