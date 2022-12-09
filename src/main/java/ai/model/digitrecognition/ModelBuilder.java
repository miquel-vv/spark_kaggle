package ai.model.digitrecognition;

import java.util.List;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class ModelBuilder {
    private Pipeline pipeline = new Pipeline();
    private CrossValidator crossValidator = new CrossValidator();
    private VectorAssembler assembler;
    private Estimator estimator;
    private Dataset<Row> data; 
    private EstimatorFactory factory;

    public CrossValidatorModel build() {
        buildPipeline();
        buildCrossValidator();
        CrossValidatorModel model = crossValidator.fit(data);

        try{
            model
                .write()
                .overwrite()
                .save(String.format("model_%s", AlgoType.RANDOM_FOREST.name()));
        } catch(IOException e){
            System.out.println("====> unable to save model.");
        }
        return model;
    }

    public ModelBuilder setAlgoType(AlgoType type) {
        switch (type) {
            case RANDOM_FOREST:
                factory = new RandomForestFactory();
                break;
        }
        return this;
    }

    public ModelBuilder addTrainData(Dataset<Row> train){
        data = train;
        return this;
    }

    private void buildPipeline() {
        createAssembler();
        createEstimator();
        pipeline.setStages(new PipelineStage[] {assembler, estimator});
    }

    private void buildCrossValidator() {
        crossValidator
            .setEstimator(pipeline)
            .setEvaluator(new MulticlassClassificationEvaluator())
            .setEstimatorParamMaps(factory.getParamGrid())
            .setNumFolds(3)
            .setParallelism(3);
    }

    private void createEstimator() {
        estimator = factory.getEstimator();
    }

    private void createAssembler() {
        List<String> pixelNames = new ArrayList<>();
        for(int i=0; i<784; i++) {
            pixelNames.add(String.format("pixel%s", i));
        }

        assembler = new VectorAssembler()
            .setInputCols(pixelNames.toArray(new String[pixelNames.size()]))
            .setOutputCol("features");
    }
}
