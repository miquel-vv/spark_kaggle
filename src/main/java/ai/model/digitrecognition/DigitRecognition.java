package ai.model.digitrecognition;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import org.apache.spark.sql.functions;


public class DigitRecognition {

    private final SparkSession spark;

    public DigitRecognition(SparkSession spark) {
        this.spark = spark;
    }

    public void run() {
        Dataset<Row> df = getData();
        Dataset<Row>[] splits = df.randomSplit(new double[] {0.9,0.1}, 1);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        CrossValidatorModel model = new ModelBuilder()
            .setAlgoType(AlgoType.RANDOM_FOREST)
            .addTrainData(training)
            .build();

        System.out.println("====> Params used: ");
        System.out.println(model.avgMetrics());

        double accuracy = determineAccuracy(model, test); 
        System.out.printf("====> test set accuracy: %s%n", accuracy);
        createPredictions(model, AlgoType.RANDOM_FOREST.name()); 
    }

    private double determineAccuracy(Model model, Dataset<Row> test) {
        Dataset<Row> results = model.transform(test);
        Dataset<Row> predictionAndLabels = results.select("prediction", "label");
        MulticlassClassificationEvaluator eval = new MulticlassClassificationEvaluator()
            .setMetricName("accuracy");
        
        return eval.evaluate(predictionAndLabels);
    }

    private void createPredictions(Model model, String folderName) {
        Dataset<Row> test = getTestData();
        Dataset<Row> predictions = model.transform(test);
        Dataset<Row> submission = predictions
            .select("ImageId","prediction")
            .withColumn("Label", functions.expr("CAST(prediction as int)"))
            .drop("prediction");

        submission
            .coalesce(1)
            .write()
            .option("header", true)
            .mode("overwrite")
            .csv(String.format("data/digitrecognition/submissions_%s",folderName)); 
    }


    private Dataset<Row> getTestData(){
        List<StructField> fields = new ArrayList<>();
        for(int i=0;i<784;i++){
            fields.add(DataTypes.createStructField(String.format("pixel%s", i), DataTypes.IntegerType, false));
        }
        StructField labelType = DataTypes.createStructField("ImageId", DataTypes.IntegerType, false);

        List<StructField> trainSetFields = new ArrayList<>(fields);
        trainSetFields.add(0, labelType);

        StructType trainSetTypes = DataTypes.createStructType(trainSetFields);
            
        Dataset<Row> df = spark.read()
            .option("header", true)
            .schema(trainSetTypes)
            .csv("data/digitrecognition/test_with_id.csv");

        return df;
    }

    private Dataset<Row> getData(){
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
            .csv("data/digitrecognition/train.csv");

        return df;
    }
}
