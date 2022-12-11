package ai.model.housepriceregression;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.expr;
import static org.apache.spark.sql.functions.when;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.OneHotEncoderModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class HousePriceRegression {
    
    private final SparkSession spark;
    private Dataset<Row> averageLotFrontage;
    private StringIndexerModel stringIndexer;
    private OneHotEncoderModel encoderModel;
    private VectorAssembler assembler;

    public HousePriceRegression(SparkSession spark) {
        this.spark = spark;
    }

    public void run() {
        Dataset<Row> df = getData();
        
        //Outlier removal
        df = df.where(
            "not(GrLivArea > 4000 and SalePrice < 300000)"
        );

        df = featureEngineering(df);
        df = transformCategoricalColumns(df);

        Dataset<Row>[] splits = df.randomSplit(new double[] {0.9,0.1}, 1);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        LinearRegressionModel model = createModel(training);
        testModel(model, test);
        test.select("features").show(5);
        createPredictions(model);
    }

    private void createPredictions(LinearRegressionModel model) {
        Dataset<Row> test = getTestData(); 
        test = featureEngineering(test);
        test = transformCategoricalColumns(test);
        test.select("features").show(5);
        
        Dataset<Row> predictions = model.transform(test);
        Dataset<Row> submission = predictions
            .select("Id", "prediction")
            .withColumnRenamed("prediction", "SalePrice");

        submission
            .coalesce(1)
            .write()
            .option("header", true)
            .mode("overwrite")
            .csv(String.format("data/houseregression/submissions"));
    }

    private void testModel(LinearRegressionModel model, Dataset<Row> outOfSample) {
        Dataset<Row> results = model.transform(outOfSample);
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

        LinearRegressionTrainingSummary summary = model.summary();

        System.out.printf("In sample rmse: %s.%n", summary.meanSquaredError());
        System.out.printf("Explanatory power: %s.%n", summary.r2());

        System.out.printf("Out of sample normal root mean squared error: %s%n", rmseNorm);
        System.out.printf("Out of sample log root mean squared error: %s%n", rmseLog);
    }

    private LinearRegressionModel createModel(Dataset<Row> df) {
        LinearRegression lr = new LinearRegression()
            .setFeaturesCol("features")
            .setLabelCol("SalePrice")
            .setMaxIter(5)
            .setRegParam(0.8)
            .setElasticNetParam(0.2);
        
        LinearRegressionModel model = lr.fit(df);
        try {
            model
                .write()
                .overwrite()
                .save("data/houseregression/regression_model");
        } catch(IOException e) {
            System.out.println("Couldn't save model.");
        }
        
        return model;
    }

    private Dataset<Row> featureEngineering(Dataset<Row> df) {

        if(averageLotFrontage == null) {
            averageLotFrontage = df.select("LotFrontage", "Neighborhood")
                .where(df.col("LotFrontage").isNotNull())
                .groupBy("Neighborhood")
                .agg(avg(df.col("LotFrontage")).alias("AvgLotFrontage"));
        }

        Dataset<Row> transformed = df.join(averageLotFrontage, "Neighborhood");
        transformed = transformed
            .withColumn("LotFrontage", when(transformed.col("LotFrontage").isNull(), transformed.col("AvgLotFrontage")).otherwise(transformed.col("LotFrontage")))
            .drop("AvgLotFrontage");

        //Drop utilities as it only has one value in training and a different value in test.
        transformed = transformed.drop("Utilities");

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
            transformed = transformed.withColumn(field, when(df.col(field).isNull(), 0).otherwise(transformed.col(field)));
        }
        
        //Set missing categorical with most common (value taken from notebook)
        transformed = transformed.withColumn("MSZoning", when(df.col("MSZoning").equalTo("NA"), "RL").otherwise(transformed.col("MSZoning")));
        transformed = transformed.withColumn("Electrical", when(df.col("Electrical").equalTo("NA"), "SBrkr").otherwise(transformed.col("Electrical")));
        transformed = transformed.withColumn("KitchenQual", when(df.col("KitchenQual").equalTo("NA"), "TA").otherwise(transformed.col("KitchenQual")));
        transformed = transformed.withColumn("SaleType", when(df.col("SaleType").equalTo("NA"), "WD").otherwise(transformed.col("SaleType")));
        transformed = transformed.withColumn("Exterior1st", when(df.col("Exterior1st").equalTo("NA"), "VinylSd").otherwise(transformed.col("Exterior1st")));
        transformed = transformed.withColumn("Exterior2nd", when(df.col("Exterior2nd").equalTo("NA"), "VinylSd").otherwise(transformed.col("Exterior2nd")));
        //Add total SF of house
        transformed = transformed.withColumn("TotalSF", expr("TotalBsmtSf + 1stFlrSF + 2ndFlrSF"));
        return transformed;
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
        
        if(stringIndexer == null) {
            StringIndexer indexer = new StringIndexer()
                .setHandleInvalid("keep")
                .setInputCols(categoricalFields.toArray(new String[categoricalFields.size()]))
                .setOutputCols(categoricalFieldsIndexedArray);
            stringIndexer = indexer.fit(df);
        }
        Dataset<Row> transformed = stringIndexer.transform(df);

        if(encoderModel == null) {
            OneHotEncoder encoder = new OneHotEncoder()
                .setInputCols(categoricalFieldsIndexedArray) 
                .setOutputCols(categoricalFieldsFeature.toArray(new String[categoricalFieldsFeature.size()]));
            encoderModel = encoder.fit(transformed);
        }
        transformed = encoderModel.transform(transformed);

        if(assembler == null) {
            featureFields.addAll(categoricalFieldsFeature);
            assembler = new VectorAssembler()
                .setInputCols(featureFields.toArray(new String[featureFields.size()]))
                .setOutputCol("features");
        }

        return assembler.transform(transformed);
    }

    private Dataset<Row> getTestData() {
        Dataset<Row> df = spark.read()
            .option("header", true)
            .schema(getSchema(false))
            .csv("data/houseregression/test.csv");
        return df;
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
