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


def main(spark, train_data_path, val_data_path, rank, reg_param, model_path=None):

    # Load data
    train_data = spark.read.parquet(train_data_path)
    val_data = spark.read.parquet(val_data_path)

    # Load ALS model / Train a new ALS model
    if model_path:
        model = ALSModel.load(model_path)
    else:
        model = train_als(train_data, rank, reg_param)
        

    # Evaluate model
    start_time = time.time()
    PK, MAP, NDCG = baseline_evaluate(model, val_data)
    total_time = time.time() - start_time

    # Print result
    paramsStr = f'Baseline \n\
                  Hyperparameters: \n\
                  train data = {train_path} \n\
                  val data = {val_path} \n\
                  rank = {rank} \n\
                  reg_param = {reg_param} \n'
    timeStr = f'Time: \n\
                Total query time: {total_time}\n'
    metricsStr = f'Metrics: \n\
                   Precision at 500: {PK} \n\
                   Mean Average Precision: {MAP} \n\
                   Normalized Discounted Cumulative Gain: {NDCG} \n'
    resultStr = paramsStr + '\n' + metricsStr + '\n'
    print('\n### FINAL RESULT:')
    print(resultStr)

    # Save result
    timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S-%f")
    fileName = '../results/baseline_result-' + timestampStr +'.txt'
    with open(fileName, 'w') as f:
        f.write(resultStr)


if __name__ == '__main__':

    # Input data path
    train_data_path = sys.argv[1]
    val_data_path = sys.argv[2]

    # Input model hyperparameters
    rank = int(sys.argv[3])
    reg_param = float(sys.argv[4])

    # Input a model path to load an existing model / Otherwise will train a new model
    try:
        model_path = sys.argv[5]
    except:
        model_path = None

    # Create Spark session
    # memory = f'{math.floor(psutil.virtual_memory()[1]*.9) >> 30}g'
    memory = '20g'
    spark = (SparkSession.builder
             .appName('baseline_pipeline')
             .master('yarn')
             .config('spark.executor.memory', memory)
             .config('spark.driver.memory', memory)
             .getOrCreate())

    main(spark, train_data_path, val_data_path, rank, reg_param, model_path=model_path)

