package ai.model.digitrecognition;

import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.param.ParamMap;

public interface EstimatorFactory {
    Estimator getEstimator();
    ParamMap[] getParamGrid();
}
