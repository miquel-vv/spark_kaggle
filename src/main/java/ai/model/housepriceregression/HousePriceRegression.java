package ai.model.housepriceregression;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.expr;
import static org.apache.spark.sql.functions.when;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.OneHotEncoderModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.collection.IterableOnce;

public class HousePriceRegression {
    
    private final SparkSession spark;

    public HousePriceRegression(SparkSession spark) {
        this.spark = spark;
    }

    public void run() {
        //Load data
        Dataset<Row> df = getData();
        
        //Transformations:
            //Outlier removal
        df = df.where(
            "not(GrLivArea > 4000 and SalePrice < 300000)"
        );

            //Filling missing values
        // missing lot frontages
        Dataset<Row> averageLotFrontage = df.select("LotFrontage", "Neighborhood")
            .where(df.col("LotFrontage").isNotNull())
            .groupBy("Neighborhood")
            .agg(avg(df.col("LotFrontage")).alias("AvgLotFrontage"));

        df = df.join(averageLotFrontage, "Neighborhood");
        df = df
            .withColumn("LotFrontage", when(df.col("LotFrontage").isNull(), df.col("AvgLotFrontage")).otherwise(df.col("LotFrontage")))
            .drop("AvgLotFrontage");

        // missing becomes 0
        String[] fields = new String[] {
            "GarageYrBlt",
            "GarageArea",
            "GarageCars",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "BsmtFullBath",
            "BsmtHalfBath",
            "MasVnrArea"
        };

        for(String field:fields) {
            df = df.withColumn(field, when(df.col(field).isNull(), 0).otherwise(df.col(field)));
        }
        
        //Set missing categorical with most common (value taken from notebook)
        df = df.withColumn("MSZoning", when(df.col("MSZoning").equalTo("NA"), "RL").otherwise(df.col("MSZoning")));
        df = df.withColumn("Electrical", when(df.col("Electrical").equalTo("NA"), "SBrkr").otherwise(df.col("Electrical")));
        df = df.withColumn("KitchenQual", when(df.col("KitchenQual").equalTo("NA"), "TA").otherwise(df.col("KitchenQual")));
        df = df.withColumn("SaleType", when(df.col("SaleType").equalTo("NA"), "WD").otherwise(df.col("SaleType")));
            //Label encode some categorical
            //Add total SF of house
        df = df.withColumn("TotalSF", expr("TotalBsmtSf + 1stFlrSF + 2ndFlrSF"));

        //Model:
            //Elastic Net regression

        df = transformCategoricalColumns(df);
        df.select("features").show(5);

        Dataset<Row>[] splits = df.randomSplit(new double[] {0.9,0.1}, 1);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        LinearRegression lr = new LinearRegression()
            .setFeaturesCol("features")
            .setLabelCol("SalePrice")
            .setMaxIter(5)
            .setRegParam(0.8)
            .setElasticNetParam(0.2);
        
        LinearRegressionModel model = lr.fit(training);

        LinearRegressionTrainingSummary summary = model.summary();

        Dataset<Row> results = model.transform(test);
        Dataset<Row> predictions = results.select("prediction", "SalePrice"); 

        spark.udf().register("logtransform", (Double p) -> {
            return Math.log(p);
        }, DataTypes.DoubleType);

        predictions = predictions
            .withColumn("prediction_log", callUDF("logtransform", predictions.col("prediction")))
            .withColumn("saleprice_log", callUDF("logtransform", predictions.col("SalePrice")));

        RegressionEvaluator evaluatorNorm = new RegressionEvaluator()
            .setPredictionCol("prediction")
            .setLabelCol("SalePrice")
            .setMetricName("rmse");

        RegressionEvaluator evaluatorLog = new RegressionEvaluator()
            .setPredictionCol("prediction_log")
            .setLabelCol("saleprice_log")
            .setMetricName("rmse");

        spark.udf().register("logtransform", (Double p) -> {
            return Math.log(p);
        }, DataTypes.DoubleType);
        
        Double rmseNorm = evaluatorNorm.evaluate(predictions);
        Double rmseLog = evaluatorLog.evaluate(predictions);

        System.out.printf("In sample rmse: %s.%n", summary.meanSquaredError());
        System.out.printf("Explanatory power: %s.%n", summary.r2());

        System.out.printf("Out of sample normal root mean squared error: %s%n", rmseNorm);
        System.out.printf("Out of sample log root mean squared error: %s%n", rmseLog);
        
    }

    private Dataset<Row> transformCategoricalColumns(Dataset<Row> df) {
        StructField[] fields = df.schema().fields();
        List<String> categoricalFields = Arrays.stream(fields)
            .filter(field -> {
                return field.dataType() == DataTypes.StringType;
            })
            .map(field -> field.name())
            .collect(Collectors.toList());

        List<String> categoricalFieldsIndexed = categoricalFields.stream()
            .map(name -> String.format("%s_indexed", name))
            .collect(Collectors.toList());

        List<String> categoricalFieldsFeature = categoricalFields.stream()
            .map(name -> String.format("%s_feature", name))
            .collect(Collectors.toList());

        List<String> featureFields = Arrays.stream(fields)
            .filter(field -> field.dataType() == DataTypes.FloatType) //Id and SalePrice are int and double, so excluded.
            .map(field -> field.name())
            .collect(Collectors.toList());

        String[] categoricalFieldsIndexedArray = categoricalFieldsIndexed.toArray(new String[categoricalFieldsIndexed.size()]); //Need this twice.

        StringIndexer indexer = new StringIndexer()
            .setInputCols(categoricalFields.toArray(new String[categoricalFields.size()]))
            .setOutputCols(categoricalFieldsIndexedArray);
        Dataset<Row> transformed = indexer.fit(df).transform(df);

        OneHotEncoder encoder = new OneHotEncoder()
            .setInputCols(categoricalFieldsIndexedArray) 
            .setOutputCols(categoricalFieldsFeature.toArray(new String[categoricalFieldsFeature.size()]));
        transformed = encoder.fit(transformed).transform(transformed);

        featureFields.addAll(categoricalFieldsFeature);

        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(featureFields.toArray(new String[featureFields.size()]))
            .setOutputCol("features");

        return assembler.transform(transformed);
    }

    private Dataset<Row> getData() {
        Dataset<Row> df = spark.read()
            .option("header", true)
            .schema(getSchema(true))
            .csv("data/houseregression/train.csv");
        return df;
    }

    private StructType getSchema(Boolean includeSalesPrice) {
        List<StructField> fields = DataTypeProvider.getDataTypes(includeSalesPrice);
        StructType schema = DataTypes.createStructType(fields); 
        return schema;
    }
}
