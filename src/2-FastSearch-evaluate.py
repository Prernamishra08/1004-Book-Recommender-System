#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import psutil
import math
from annoy import AnnoyIndex
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from pyspark.mllib.evaluation import RankingMetrics



def fast_evaluate_with_annoy(n_candidates, als_model, val_data, annoy_tree_path, annoy_index_map):

    #--- 1. Loading ----------------------------------------------------------
    ##########################################################################
    K = 500
    rank = als_model.rank

    # Get val_user list
    val_user = val_data.select('user_id').distinct()
    # Get val user factors
    user_factors = als_model.userFactors.withColumnRenamed('id', 'user_id')
    val_user_factors = val_user.join(user_factors, on='user_id', how='inner')


    #--- 2. Build candidate set ----------------------------------------------
    ##########################################################################
    @udf(returnType=ArrayType(IntegerType()))
    def find_candidates_udf(u_factor):
        from annoy import AnnoyIndex
        seed = 401
        u = AnnoyIndex(rank, 'dot')
        u.set_seed(seed)
        u.load(SparkFiles.get(annoy_tree_path))
        return u.get_nns_by_vector(u_factor, n_candidates, search_k=-1, include_distances=False)

    # Retrieve candidates from annoy tree
    candidates = val_user_factors.select('user_id', find_candidates_udf(col('features')).alias('candidates'))

    # Reformat candidates DataFrame
    candidates = candidates.select("user_id",F.explode("candidates").alias('annoy_id'))
    # Convert annoy index to original book_id
    candidates = candidates.join(annoy_index_map, on='annoy_id', how='inner')
    candidates = candidates.select('user_id', col('id').alias('book_id'))

    #--- 3. Construct recommendation list ------------------------------------
    ##########################################################################
    # Calculate predicted rating scores for candidate books
    pred = als_model.transform(candidates)
    # Reformat
    pred = pred.select('user_id', F.struct('book_id', 'prediction').alias('pred'))
    pred = pred.groupBy('user_id').agg(F.collect_list("pred").alias('pred'))

    # Select top 500 books to recommend
    @udf(returnType=ArrayType(IntegerType()))
    def top_500_udf(l):
        res = sorted(l, key=lambda l:l[1], reverse=True)
        if len(l)<500:
            K = len(l)
        else:
            K = 500
        return [l[i][0] for i in range(K)]

    rec = pred.select('user_id', top_500_udf(col('pred')).alias('recommendations'))


    #--- 4. Evaluate ---------------------------------------------------------
    ##########################################################################
    # Get ground truth list
    ground_truth = val_data.groupBy("user_id") \
                  .agg(F.collect_set("book_id") \
                  .alias('true_books'))

    # Calculate ranking metrics
    prediction_and_labels = rec.join(ground_truth, on='user_id', how='inner').drop('user_id').rdd

    metrics = RankingMetrics(prediction_and_labels)
    PK = metrics.precisionAt(K)
    MAP = metrics.meanAveragePrecision
    NDCG = metrics.ndcgAt(K)

    return PK, MAP, NDCG
    


def main(spark, n_candidates, als_model_path, val_data_path, annoy_tree_path, annoy_index_map_path):

    # Load ALS model & val data & annoy index map
    als_model = ALSModel.load(als_model_path)
    val_data = spark.read.parquet(val_data_path)
    annoy_index_map = spark.read.parquet(annoy_index_map_path)

    # Query & Evaluate with annoy
    # Also calculate running time
    start_time = time.time()
    PK, MAP, NDCG = fast_evaluate_with_annoy(n_candidates, als_model, val_data, 
                                             annoy_tree_path, annoy_index_map)
    total_time = time.time() - start_time


    # Print result
    annoyStr = f'Fast Search with annoy \n\
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
    fileName = '../results/annoy_result-' + timestampStr +'.txt'
    with open(fileName, 'w') as f:
        f.write(resultStr)



if __name__ == '__main__':

    # Input path
    als_model_path = sys.argv[1]
    val_data_path = sys.argv[2]
    annoy_tree_path = sys.argv[3]
    annoy_index_map_path = sys.argv[4]
    
    # Number of candidates for each uer
    try:
        n_candidates = int(sys.argv[5])
    except:
        n_candidates = 1000

    # Create Spark session
    # memory = f'{math.floor(psutil.virtual_memory()[1]*.9) >> 30}g'
    memory = '5g'
    spark = (SparkSession.builder
             .appName('evaluate_with_annoy')
             .master('yarn')
             .config('spark.executor.memory', memory)
             .config('spark.driver.memory', memory)
             .getOrCreate())

    main(spark, n_candidates, als_model_path, val_data_path, annoy_tree_path, annoy_index_map_path)

