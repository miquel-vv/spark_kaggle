package ai.model.digitrecognition;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;


public class App {
    public static void main( String[] args ) {
        SparkSession spark = SparkSession.builder()
            .appName("testSpark").master("local[*]")
            .getOrCreate();
        
        Dataset<Row> df = getData(spark);
        
        List<String> columns = List.of(df.columns());
        List<String> pixelNames = columns.subList(1, columns.size());

        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(pixelNames.toArray(new String[pixelNames.size()]))
            .setOutputCol("features");
        
        //Dataset<Row> transformed = assembler.transform(df);

        Dataset<Row>[] splits = df.randomSplit(new double[] {0.9,0.1}, 1);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        RandomForestClassifier forest = new RandomForestClassifier();
        //RandomForestClassificationModel model = forest.fit(training);
        
        Pipeline pipeline = new Pipeline()
            .setStages(new PipelineStage[] {assembler, forest});

        ParamMap[] paramGrid = new ParamGridBuilder()
            .addGrid(forest.numTrees(), new int[] {25, 50, 100})
            .addGrid(forest.maxDepth(), new int[] {5, 10})
            .build();

        CrossValidator cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(new MulticlassClassificationEvaluator())
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(3)
            .setParallelism(3);

        CrossValidatorModel model = cv.fit(training);

        Dataset<Row> results = model.transform(test);
        Dataset<Row> predictionAndLabels = results.select("prediction", "label");
        MulticlassClassificationEvaluator eval = new MulticlassClassificationEvaluator()
            .setMetricName("accuracy");
        
        System.out.printf("test set accuracy: %s%n", eval.evaluate(predictionAndLabels));

        spark.stop();
    }

    private static Dataset<Row> getData(SparkSession spark){
        StructField labelType = DataTypes.createStructField("label", DataTypes.DoubleType, false);
        List<StructField> fields = new ArrayList<>();

        for(int i=0;i<784;i++){
            fields.add(DataTypes.createStructField(String.format("pixel%s", i), DataTypes.IntegerType, false));
        }

        List<StructField> trainSetFields = new ArrayList<>(fields);
        trainSetFields.add(0, labelType);

        StructType trainSetTypes = DataTypes.createStructType(trainSetFields);
            
        Dataset<Row> df = spark.read()
            .option("header", true)
            .schema(trainSetTypes)
            .csv("train.csv");

        return df;
    }
}
