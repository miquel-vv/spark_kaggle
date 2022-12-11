package ai.model.housepriceregression;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;

public class DataTypeProvider {
    public static List<StructField> getDataTypes(Boolean includeSalesPrice){
        List<StructField> fields = new ArrayList<>();

        fields.add(DataTypes.createStructField("Id", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("MSSubClass", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("MSZoning", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("LotFrontage", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("LotArea", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("Street", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("Alley", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("LotShape", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("LandContour", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("Utilities", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("LotConfig", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("LandSlope", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("Neighborhood", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("Condition1", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("Condition2", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("BldgType", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("HouseStyle", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("OverallQual", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("OverallCond", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("YearBuilt", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("YearRemodAdd", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("RoofStyle", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("RoofMatl", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("Exterior1st", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("Exterior2nd", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("MasVnrType", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("MasVnrArea", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("ExterQual", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("ExterCond", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("Foundation", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("BsmtQual", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("BsmtCond", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("BsmtExposure", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("BsmtFinType1", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("BsmtFinSF1", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("BsmtFinType2", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("BsmtFinSF2", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("BsmtUnfSF", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("TotalBsmtSF", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("Heating", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("HeatingQC", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("CentralAir", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("Electrical", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("1stFlrSF", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("2ndFlrSF", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("LowQualFinSF", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("GrLivArea", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("BsmtFullBath", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("BsmtHalfBath", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("FullBath", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("HalfBath", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("BedroomAbvGr", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("KitchenAbvGr", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("KitchenQual", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("TotRmsAbvGrd", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("Functional", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("Fireplaces", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("FireplaceQu", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("GarageType", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("GarageYrBlt", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("GarageFinish", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("GarageCars", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("GarageArea", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("GarageQual", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("GarageCond", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("PavedDrive", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("WoodDeckSF", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("OpenPorchSF", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("EnclosedPorch", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("3SsnPorch", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("ScreenPorch", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("PoolArea", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("PoolQC", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("Fence", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("MiscFeature", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("MiscVal", DataTypes.FloatType, true));
        fields.add(DataTypes.createStructField("MoSold", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("YrSold", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("SaleType", DataTypes.StringType, true));
        fields.add(DataTypes.createStructField("SaleCondition", DataTypes.StringType, true));
        if(includeSalesPrice)
            fields.add(DataTypes.createStructField("SalePrice", DataTypes.FloatType, true));

        return fields;
    }
 
}
