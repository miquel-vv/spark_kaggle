package ai.model;

import org.apache.spark.sql.SparkSession;
import ai.model.digitrecognition.DigitRecognition;
import ai.model.housepriceregression.HousePriceRegression;

public class App {

    private static SparkSession SPARK;

    public static void main( String[] args ) {
        SPARK = SparkSession.builder()
            .appName("testSpark").master("local[*]")
            .getOrCreate();
        
        housePriceRegression();

        SPARK.stop();
    }

    private static void housePriceRegression() {
        HousePriceRegression housePriceRegression = new HousePriceRegression(SPARK);
        housePriceRegression.run();
    }

    private static void digitRecoginition() {
        DigitRecognition digitRecognition = new DigitRecognition(SPARK);
        digitRecognition.run();
    }
}
