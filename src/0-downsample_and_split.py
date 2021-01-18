#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import sys
import psutil
import math
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession


def downsample(spark, fraction):
    seed = 401

    # Load full data
    interaction = spark.read.parquet('hdfs:/user/lw2350/recommend/goodreads_interactions.parquet')

    # Downsample users
    sample_user = interaction.select('user_id').distinct().sample(False, fraction=fraction, seed=seed)

    # Get sample users' interactions
    sample_user_interaction = sample_user.join(interaction, on='user_id', how='inner')

    return sample_user_interaction


def split(spark, interaction):
    # Get user_id list
    all_user = interaction.select('user_id').distinct()

    # Split users into 0.6, 0.2, 0.2
    train_user, val_user, test_user = all_user.randomSplit(weights=[0.6,0.2,0.2], seed=401)

    train_split = interaction.join(train_user, on='user_id', how='inner')
    val_split = interaction.join(val_user, on='user_id', how='inner')
    test_split = interaction.join(test_user, on='user_id', how='inner')

    # Partition interaction for each user into half and half for validation and test
    window = Window.partitionBy('user_id').orderBy('book_id') 
    val_split = val_split.select('user_id','book_id','rating', F.row_number().over(window).alias("row_number"))
    test_split = test_split.select('user_id','book_id','rating', F.row_number().over(window).alias("row_number"))

    val_observed = val_split.filter(val_split.row_number % 2 == 1).drop('row_number')
    val_data = val_split.filter(val_split.row_number % 2 == 0).drop('row_number')

    test_observed = test_split.filter(test_split.row_number % 2 == 1).drop('row_number')
    test_data = test_split.filter(test_split.row_number % 2 == 0).drop('row_number')

    train_data = train_split.union(val_observed).union(test_observed)
    train_data = train_data.filter(train_data.rating > 0)

    return train_data, val_data, test_data


def main(spark, fraction, netid):
    # Downsample if fraction < 1
    if fraction < 1:
        interaction = downsample(spark, fraction)
    else:
        interaction = spark.read.parquet('hdfs:/user/lw2350/recommend/goodreads_interactions.parquet')

    # Train-Val-Test split
    train_data, val_data, test_data = split(spark, interaction)

    # write into parquet file
    train_data.write.parquet('hdfs:/user/{}/recommend/{}_train.parquet'.format(netid,fraction*100))
    val_data.write.parquet('hdfs:/user/{}/recommend/{}_val.parquet'.format(netid,fraction*100))
    test_data.write.parquet('hdfs:/user/{}/recommend/{}_test.parquet'.format(netid,fraction*100))


if __name__ == '__main__':

    netid = sys.argv[1]

    # Input a fraction if want to downsample
    try:
        fraction = float(sys.argv[2])
    except:
        fraction = 1
    
    # Create Spark session
    memory = f'{math.floor(psutil.virtual_memory()[1]*.9) >> 30}g'
    spark = (SparkSession.builder
             .appName('split')
             .master('yarn')
             .config('spark.executor.memory', memory)
             .config('spark.driver.memory', memory)
             .getOrCreate())

    main(spark, fraction, netid)

