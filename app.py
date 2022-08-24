import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import corr, isnull, when, count, col
from pandas.plotting import scatter_matrix
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import GBTClassifier
from pyspark.mllib.stat import Statistics

spark = SparkSession.builder.appName('ml-students').getOrCreate()

def loadData(type) :
    if type == "local" :
        df_mat = spark.read.option("delimiter", ";").csv('file:///Users/mjinor/Desktop/ms.dehghan/student/student-mat.csv', header = True, inferSchema = True)
        df_por = spark.read.option("delimiter", ";").csv('file:///Users/mjinor/Desktop/ms.dehghan/student/student-por.csv', header = True, inferSchema = True)
        return df_mat.union(df_por)
    elif type == "hdfs" :
        df_mat = spark.read.option("delimiter", ";").csv('student-mat.csv', header = True, inferSchema = True)
        df_por = spark.read.option("delimiter", ";").csv('student-por.csv', header = True, inferSchema = True)
        return df_mat.union(df_por)

def makeFeaturesNumeric(df) :
    df = df.withColumn('sex_numeric', when(col("sex") == "F", 1).otherwise(2))
    df = df.withColumn('address_numeric', when(col("address") == "U", 1).otherwise(2))
    df = df.withColumn('school_numeric', when(col("school") == "GP", 1).otherwise(2))
    df = df.withColumn('famsize_numeric', when(col("famsize") == "LE3", 3).otherwise(4))
    df = df.withColumn('Pstatus_numeric', when(col("Pstatus") == "T", 1).otherwise(0))
    df = df.withColumn('Mjob_numeric', when(col("Mjob") == "teacher", 1).when(col("Mjob") == "health", 2).when(col("Mjob") == "services", 3).when(col("Mjob") == "at_home", 4).otherwise(0))
    df = df.withColumn('Fjob_numeric', when(col("Fjob") == "teacher", 1).when(col("Fjob") == "health", 2).when(col("Fjob") == "services", 3).when(col("Fjob") == "at_home", 4).otherwise(0))
    df = df.withColumn('reason_numeric', when(col("reason") == "home", 1).when(col("reason") == "reputation", 2).when(col("reason") == "course", 3).otherwise(4))
    df = df.withColumn('guardian_numeric', when(col("guardian") == "mother", 1).otherwise(2))
    df = df.withColumn('schoolsup_numeric', when(col("schoolsup") == "yes", 1).otherwise(0))
    df = df.withColumn('famsup_numeric', when(col("famsup") == "yes", 1).otherwise(0))
    df = df.withColumn('paid_numeric', when(col("paid") == "yes", 1).otherwise(0))
    df = df.withColumn('activities_numeric', when(col("activities") == "yes", 1).otherwise(0))
    df = df.withColumn('nursery_numeric', when(col("nursery") == "yes", 1).otherwise(0))
    df = df.withColumn('higher_numeric', when(col("higher") == "yes", 1).otherwise(0))
    df = df.withColumn('internet_numeric', when(col("internet") == "yes", 1).otherwise(0))
    df = df.withColumn('romantic_numeric', when(col("romantic") == "yes", 1).otherwise(0))
    df = df.withColumn('target', when(col('G3') >= '16', 1).otherwise(0))
    df = df.replace('null', None).dropna(how='any')
    df = df.drop("school")
    df = df.drop("sex")
    df = df.drop("address")
    df = df.drop("famsize")
    df = df.drop("Pstatus")
    df = df.drop("Mjob")
    df = df.drop("Fjob")
    df = df.drop("reason")
    df = df.drop("guardian")
    df = df.drop("schoolsup")
    df = df.drop("famsup")
    df = df.drop("paid")
    df = df.drop("activities")
    df = df.drop("nursery")
    df = df.drop("higher")
    df = df.drop("internet")
    df = df.drop("romantic")
    df = df.drop("G1")
    df = df.drop("G2")
    df = df.drop("G3")
    return df

def preProccess(df) :
    df = makeFeaturesNumeric(df)
    numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
    numeric_data = df.select(numeric_features).toPandas()
    df = df.select(numeric_features)
    n = len(numeric_data.columns)
    threshold = 0.8
    clmns = ["target"]
    for i in range(n):
        column_name = numeric_data.columns[i]
        mean_corr = 0.0
        Xs = df.select(column_name).rdd.map(lambda x : np.array(x)[0])
        counter = 0
        for j in range(n) :
            if j != i :
                counter = counter + 1
                Ys = df.select(numeric_data.columns[j]).rdd.map(lambda x : np.array(x)[0])
                corr = Statistics.corr(Xs,Ys,"pearson")
                mean_corr += corr
        if mean_corr > threshold :
            clmns.append(column_name)
    df = df.select(clmns)
    return df

def decisionTree(training_data,test_data) :
    dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'target')
    dtModel = dt.fit(training_data)
    dt_predictions = dtModel.transform(test_data)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'target', metricName = 'accuracy')
    print('Decision Tree Accuracy:', multi_evaluator.evaluate(dt_predictions))

def randomForest(training_data,test_data) :
    rf = RandomForestClassifier(labelCol='target', featuresCol='features')
    model = rf.fit(training_data)
    rf_predictions = model.transform(test_data)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'target', metricName = 'accuracy')
    print('Random Forest classifier Accuracy:', multi_evaluator.evaluate(rf_predictions))

def logisticRegression(training_data,test_data) :
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'target', maxIter=10)
    lrModel = lr.fit(training_data)
    lr_predictions = lrModel.transform(test_data)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'target', metricName = 'accuracy')
    print('Logistic Regression Accuracy:', multi_evaluator.evaluate(lr_predictions))

def gbtClassifier(training_data,test_data) :
    gb = GBTClassifier(labelCol = 'target', featuresCol = 'features')
    gbModel = gb.fit(training_data)
    gb_predictions = gbModel.transform(test_data)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'target', metricName = 'accuracy')
    print('Gradient-boosted Trees Accuracy:', multi_evaluator.evaluate(gb_predictions))

def main() :
    spark.sparkContext.setLogLevel("WARN")
    df = loadData("local")
    df = preProccess(df)
    required_features = df.columns
    required_features.remove("target")
    assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    transformed_data = assembler.transform(df)
    (training_data, test_data) = transformed_data.randomSplit([0.8,0.2])
    randomForest(training_data,test_data)

if __name__ == "__main__":
    main()