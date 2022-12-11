package ai.model.housepriceregression;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
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
        System.out.printf("Amount of lines: %s%n", df.count());
        
        //Transformations:
            //Outlier removal
            //Target value normalization
            //Filling missing values
            //Transform numerical into categorical
            //Label encode some categorical
            //Add total SF of house
            //Box Cox transform skewed features
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
            .csv("data/digitrecognition/train.csv");
        return df;
    }

    private StructType getSchema(Boolean includeSalesPrice) {
        List<StructField> fields = DataTypeProvider.getDataTypes(includeSalesPrice);
        StructType schema = DataTypes.createStructType(fields); 
        return schema;
    }
}
