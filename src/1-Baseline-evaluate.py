#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import psutil
import math
from baseline import *
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from datetime import datetime


def main(spark, als_model_path, val_data_path):

    # Load ALS model & validation data
    als_model = ALSModel.load(als_model_path)
    val_data = spark.read.parquet(val_data_path)

    # Query with recommendForUserSubset() method & Evaluate using ranking metrics
    # Also calculate running time
    start_time = time.time()
    PK, MAP, NDCG = baseline_evaluate(als_model, val_data)
    total_time = time.time() - start_time

    # Print result
    annoyStr = f'NO Fast Search \n\
                 Evaluate using recommendForUserSubset() \n\
                 val data = {val_data_path} \n\
                 model = {als_model_path} \n'
    timeStr = f'Time: \n\
                Total query time: {total_time}\n'
    metricsStr = f'Metrics: \n\
                   Precision at 500: {PK} \n\
                   Mean Average Precision: {MAP} \n\
                   Normalized Discounted Cumulative Gain: {NDCG} \n'
    resultStr = annoyStr + '\n' + timeStr + '\n' + metricsStr + '\n'
    print('\n### FINAL RESULT:')
    print(resultStr)

    # Save result
    timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S-%f")
    fileName = '../results/baseline_result-' + timestampStr +'.txt'
    with open(fileName, 'w') as f:
        f.write(resultStr)


if __name__ == '__main__':

    # Input ALS model path & validation data path
    als_model_path = sys.argv[1]
    val_data_path = sys.argv[2]

    # Create Spark session
    # memory = f'{math.floor(psutil.virtual_memory()[1]*.9) >> 30}g'
    memory = '10g'
    spark = (SparkSession.builder
             .appName('evaluate_baseline')
             .master('yarn')
             .config('spark.executor.memory', memory)
             .config('spark.driver.memory', memory)
             .getOrCreate())

    # spark = SparkSession.builder.appName('test').getOrCreate()
    # spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

    main(spark, als_model_path, val_data_path)


