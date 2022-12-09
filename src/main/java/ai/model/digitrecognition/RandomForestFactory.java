package ai.model.digitrecognition;

import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;

public class RandomForestFactory implements EstimatorFactory {

    private RandomForestClassifier forest;

    @Override
    public Estimator getEstimator() {
        if(forest == null) {
            forest = new RandomForestClassifier();
        }
        return forest;
    }

    @Override
    public ParamMap[] getParamGrid() {
        ParamMap[] paramGrid = new ParamGridBuilder()
            .addGrid(forest.numTrees(), new int[] {50})
            .addGrid(forest.maxDepth(), new int[] {5, 10, 15, 20})
            .build();

        return paramGrid;
    }
    
}
