package org.fabel.enhancer;

import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: antalb
 * Date: 7/24/13
 * Time: 1:17 PM
 * To change this template use File | Settings | File Templates.
 */
public class UnbalancedEnhancerSA {

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
    private double[] classDistributions;

    public UnbalancedEnhancerSA(double[][] labels, double[] actual, double[] classByIndex, double[] classDistributions, double penalty, int step,  double energyThreshold, double temperature, double tempChange) {
        this.labels = labels;
        this.actual = actual;
        this.classByIndex = classByIndex;
        this.penalty = penalty;
        this.step = step;
        this.classDistributions = classDistributions;
        this.energyThreshold = energyThreshold;
        this.temperature = temperature;
        this.tempChange = tempChange;
    }

    public UnbalancedEnhancerSA(double[][] labels, double[] actual, double[] classByIndex, double[] classDistributions, double temperature, double tempChange, double energyThreshold) {
        this.labels = labels;
        this.actual = actual;
        this.classByIndex = classByIndex;
        this.classDistributions = classDistributions;
        this.temperature = temperature;
        this.tempChange = tempChange;
        this.energyThreshold = energyThreshold;
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

    public double getSmoothness(int m, int step) {
        double sum = 0;
        double count = 0;
        int start = Math.max(m-step, 0), end = Math.min(m+step+1, numInstances);
        for (int i = 0; i < numClassifiers; ++i) {

            for (int j = start; j < end; ++j) {
                double value = labels[m][j];
                if (i != j) {
                    count++;
                    double beta = classDistributions[(int)labels[i][j]];
                    if (value == labels[i][j]) {
                        sum += beta*penalty;
                    }
                    else {
                        sum -= beta*penalty;
                    }
                }
            }
        }
        return sum / count;
    }

    public double getDisagreement(int m, int n) {
        double value = labels[m][n];
        double disagreement = 0;
        for (int i = 0; i < numClassifiers; ++i) {
            if (i != m && labels[i][n] != value) {
                disagreement+=classDistributions[(int)actual[n]];
            }
        }

        return -disagreement / (double)numClassifiers;
    }

    public double getEnergy(int m, int n) {
        return getHorizontalEnergy(n) + /*getVerticalEnergy(m) + getClassificationError(m, 0.1) + getSmoothness(m, 30)+ getClassificationError(m, n) /* + getMajorityEnergy(m, n, 1) + getClassificationError(n)*/getSmoothness(m, numClassifiers)+getDisagreement(m, n);
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
                        Random random = new Random();
                        double newValue = classByIndex[random.nextInt(numClasses)];
                        while (value == newValue) {
                            newValue = classByIndex[random.nextInt(numClasses)];
                        }
                        labels[i][j] = newValue;
                    }
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

        }
        while (delta > energyThreshold);
        return labels;

    }
}
