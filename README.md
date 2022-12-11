# Kaggle competitions with spark
## To run
1. Use java 11
2. Make sure maven has enough heap size, otherwise set the environment variable MAVEN_OPTS to '-Xms256m -Xmx8G'.
3. Run with 'mvn exec:java digits' for the digit_recognition model or 'mvn exec:java houses' for the house price regression.  