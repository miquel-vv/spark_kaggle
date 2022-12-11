package ai.model.housepriceregression;

import java.util.ArrayList;
import java.util.List;

import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class HousePriceRegression {
    
    private final SparkSession spark;

    public HousePriceRegression(SparkSession spark) {
        this.spark = spark;
    }

    public void run() {
        //Load data
        Dataset<Row> df = getData();
        Dataset<Row>[] splits = df.randomSplit(new double[] {0.9,0.1}, 1);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];
        
        //Transformations:
            //Outlier removal
        df = df.where(
            "not(GrLivArea > 4000 and SalePrice < 300000)"
        );
            //Target value normalization

        spark.udf().register("logtransform", (Double p) -> {
            return Math.log(p);
        }, DataTypes.DoubleType);

        df = df.withColumn("logSalePrice", callUDF("logtransform", df.col("SalePrice")));
        
            //Filling missing values
        // missing lot frontages
        Dataset<Row> averageLotFrontage = df.select("LotFrontage", "Neighborhood")
            .where(df.col("LotFrontage").isNotNull())
            .groupBy("Neighborhood")
            .agg(avg(df.col("LotFrontage")).alias("AvgLotFrontage"));

        df = df.join(averageLotFrontage, df.col("Neighborhood").equalTo(averageLotFrontage.col("Neighborhood")));
        df = df
            .withColumn("LotFrontage", when(df.col("LotFrontage").isNull(), df.col("AvgLotFrontage")).otherwise("LotFrontage"))
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
            //Box Cox transform skewed features => if extra time, not available out of the box. 

            //Dummy categorical
        //Model:
            //LASSO regression
            //Elastic Net regression
            //Kernel Ridge regression
            //Gradient boosting regression
            //xgboost
        //Stacking
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
