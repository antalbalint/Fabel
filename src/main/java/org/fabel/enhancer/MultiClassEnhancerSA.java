package org.fabel.enhancer;

import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: antalb
 * Date: 7/24/13
 * Time: 1:38 PM
 * To change this template use File | Settings | File Templates.
 */
public class MultiClassEnhancerSA {
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

    public MultiClassEnhancerSA(double[][] labels, double[] actual, double[] classByIndex, double temperature, double tempChange, double energyThreshold, double penalty, int step) {
        this.labels = labels;
        this.actual = actual;
        this.classByIndex = classByIndex;
        this.temperature = temperature;
        this.tempChange = tempChange;
        this.energyThreshold = energyThreshold;
        this.penalty = penalty;
        this.step = step;
        this.numClassifiers = labels.length;
        this.numInstances = actual.length;
        this.optimalVotes = (numClassifiers)/2+1;
        this.numClasses = classByIndex.length;
        /*for (int i = 0; i < labels[0].length; ++i) {
            numberOfVotes += labels[0][i] != 0 ? 1 : 0;
        }*/
    }

    public double getHorizontalEnergy(int n) {
        double sum = 0;
        for (int i = 0; i < numClassifiers; ++i) {
            double value = labels[i][n];
            if (value == actual[n])
                sum++;
        }
        return Math.abs(optimalVotes - sum)  / (double)optimalVotes;

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

    public double getDisagreement(int m, int n) {
        double value = labels[m][n];
        double disagreement = 0;
        for (int i = 0; i < numClassifiers; ++i) {
            if (i != m && labels[i][n] != value) {
                disagreement++;
            }
        }

        return (1.0-disagreement) / (double)numClassifiers;
    }

    public double getEnergy(int m, int n) {
        double hor = getHorizontalEnergy(n);
        double smooth = getSmoothness(m, numClassifiers);
        double dis = getDisagreement(m, n);
        return hor /*+ smooth + dis*/;
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

/*    int i, j;
    int r;
    double kszi = log(alpha);  // This is for MMD. When executing
    // Metropolis, kszi will be randomly generated.
    double summa_deltaE;

    TRandomMersenne rg(time(0));  // create instance of random number generator

    K = 0;
    T = T0;
    E_old = CalculateEnergy();

    do
    {
        summa_deltaE = 0.0;
        for (i=0; i<height; ++i)
            for (j=0; j<width; ++j)
            {
	    *//* Generate a new label different from the current one with
	     * uniform distribution.
	     *//*
                if (no_regions == 2)
                    r = 1 - classes[i][j];
                else
                    r = (classes[i][j] +
                            (int)(rg.Random()*(no_regions-1))+1) % no_regions;
                if (!mmd)  // Metropolis: kszi is a  uniform random number
                    kszi = log(rg.Random());
	    *//* Accept the new label according to Metropolis dynamics.
	     *//*
                if (kszi <= (LocalEnergy(i, j, classes[i][j]) -
                        LocalEnergy(i, j, r)) / T) {
                    summa_deltaE +=
                            fabs(LocalEnergy(i, j, r) - LocalEnergy(i, j, classes[i][j]));
                    E_old = E = E_old -
                            LocalEnergy(i, j, classes[i][j]) + LocalEnergy(i, j, r);
                    classes[i][j] = r;
                }
            }
        T *= c;         // decrease temperature
        ++K;	      // advance iteration counter
        CreateOutput(); // display current labeling
    } while (summa_deltaE > t); // stop when energy change is small*/


    public double[][] start() {
        double energy = getEnergy();
        double delta = 0;
        int k = 0;
        double avgCorr = 0;
        do {
            System.out.println(k + ": " + temperature + " " + energy + " " + delta);
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
                        double newValue;
                        do {
                            newValue = (value+classByIndex[random.nextInt(numClasses)])%numClasses;
                        }
                        while (value == newValue);
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
                    //labels[i][j] = (labels[i][j]+1)%numClasses;
                    /*if (numClasses == 2) {
                        labels[i][j] = 1.0 - labels[i][j];
                    }
                    else {
                        Random random = new Random();
                        double newValue = classByIndex[random.nextInt(numClasses)];
                        while (value == newValue) {
                            newValue = classByIndex[random.nextInt(numClasses)];
                        }
                        labels[i][j] = newValue;

                    }*/

                }
            }
            k++;
            temperature *= tempChange;

        }
        while (delta > energyThreshold);
        return labels;

    }
}
