package ai.model.digitrecognition;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.hadoop.shaded.org.apache.commons.net.nntp.Article;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.mllib.linalg.VectorUDT;


public class App {
    public static void main( String[] args ) {
        SparkSession spark = SparkSession.builder()
            .appName("testSpark").master("local[*]")
            .getOrCreate();

        StructField labelType = DataTypes.createStructField("label", DataTypes.DoubleType, false);
        List<StructField> fields = new ArrayList<>();

        List<String> pixelNames = new ArrayList();
        for(int i=0;i<784;i++){
            String colName = String.format("pixel%s", i);
            pixelNames.add(colName);
            fields.add(DataTypes.createStructField(colName, DataTypes.IntegerType, false));
        }

        List<StructField> trainSetFields = new ArrayList<>(fields);
        trainSetFields.add(0, labelType);

        StructType trainSetTypes = DataTypes.createStructType(trainSetFields);
            
        Dataset<Row> df = spark.read()
            .option("header", true)
            .schema(trainSetTypes)
            .csv("train.csv");

        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(pixelNames.toArray(new String[pixelNames.size()]))
            .setOutputCol("features");

        Dataset<Row> transformed = assembler.transform(df);

        Dataset<Row>[] splits = transformed.randomSplit(new double[] {0.9,0.1}, 1);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        RandomForestClassifier forest = new RandomForestClassifier();
        RandomForestClassificationModel model = forest.fit(training);

        Dataset<Row> results = model.transform(test);
        Dataset<Row> predictionAndLabels = results.select("prediction", "label");
        MulticlassClassificationEvaluator eval = new MulticlassClassificationEvaluator()
            .setMetricName("accuracy");
        
        System.out.printf("test set accuracy: %s%n", eval.evaluate(predictionAndLabels));

        spark.stop();
    }
}
