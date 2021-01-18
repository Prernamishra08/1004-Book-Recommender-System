#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import psutil
import math
from baseline import *
from pyspark.sql import SparkSession


def main(spark, train_data_path, rank, reg_param, model_path=None):
    
    # Load train data
    train_data = spark.read.parquet(train_data_path)
    
    # Train ALS model
    model = train_als(train_data, rank, reg_param)
    print('Model trained')

    # Save model
    if model_path:
        model.save(model_path)
        print('Model Saved at ' + model_path + '\n')
    else:
        print('Model not saved')


if __name__ == '__main__':

    # Path of training data
    train_data_path = sys.argv[1]

    # Hyperparameters of ALS model
    rank = int(sys.argv[2])
    reg_param = float(sys.argv[3])

    # Input a model path if want to save model
    try:
        model_path = sys.argv[4]
    except:
        model_path = None

    # Create Spark session
    # memory = f'{math.floor(psutil.virtual_memory()[1]*.9) >> 30}g'
    memory = '20g'
    spark = (SparkSession.builder
             .appName('train_als')
             .master('yarn')
             .config('spark.executor.memory', memory)
             .config('spark.driver.memory', memory)
             .getOrCreate())

    main(spark, train_data_path, rank, reg_param, model_path)

