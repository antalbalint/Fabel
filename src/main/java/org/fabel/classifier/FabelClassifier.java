package org.fabel.classifier;

import org.fabel.enhancer.FabelEnhancer;
import weka.classifiers.Classifier;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.meta.Vote;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;


import java.util.Random;


/**
 * Created by Balint on 2015-07-11.
 */
public class FabelClassifier extends RandomizableIteratedSingleClassifierEnhancer {

    private Vote vote;
    private FabelEnhancer enhancer;
    private Classifier[] classifiers;
    private double[][] labels;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        Classifier cls = getClassifier();
        int seed = getSeed();
        Random random = new Random(seed);
        instances.randomize(random);
        enhancer = new FabelEnhancer(cls, instances, random);
        classifiers = enhancer.createClassifiers();
        labels = enhancer.getLabels();
        vote = new Vote();
        vote.setCombinationRule(new SelectedTag(Vote.MAJORITY_VOTING_RULE, Vote.TAGS_RULES));
        vote.setClassifiers(classifiers);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return vote.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return vote.distributionForInstance(instance);
    }

    public double[][] getLabels() {
        return labels;
    }

    public Classifier[] getClassifiers() {
        return classifiers;
    }
}
