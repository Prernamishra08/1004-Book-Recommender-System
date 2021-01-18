#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import psutil
import math
import time
from annoy import AnnoyIndex
from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics
from datetime import datetime



def down_sample_items(interaction, fraction):
    if fraction < 1:
        sample_items = interaction.select('book_id').distinct().sample(False, fraction=fraction, seed=401)
    else:
        sample_items = interaction.select('book_id').distinct()
    sample_items_interaction = sample_items.join(interaction, on='book_id', how='inner')
    return sample_items_interaction


def split(interaction):
    # Get user_id list
    all_user = interaction.select('user_id').distinct()

    # Split users into 0.6, 0.2, 0.2
    train_user, val_user, test_user = all_user.randomSplit(weights=[0.6,0.2,0.2], seed=401)

    train_split = interaction.join(train_user, on='user_id', how='inner')
    val_split = interaction.join(val_user, on='user_id', how='inner')
    # test_split = interaction.join(test_user, on='user_id', how='inner')

    # Partition interaction for each user into half and half for validation and test
    window = Window.partitionBy('user_id').orderBy('book_id') 
    val_split = val_split.select('user_id','book_id','rating', F.row_number().over(window).alias("row_number"))
    # test_split = test_split.select('user_id','book_id','rating', F.row_number().over(window).alias("row_number"))

    val_observed = val_split.filter(val_split.row_number % 2 == 1).drop('row_number')
    val_data = val_split.filter(val_split.row_number % 2 == 0).drop('row_number')

    # test_observed = test_split.filter(test_split.row_number % 2 == 1).drop('row_number')
    # test_data = test_split.filter(test_split.row_number % 2 == 0).drop('row_number')

    # train_data = train_split.union(val_observed).union(test_observed)
    train_data = train_split.union(val_observed)
    train_data = train_data.filter(train_data.rating > 0)

    # return train_data, val_data, test_data
    return train_data, val_data


################################################################################################################
def train_als(train_data, rank, reg_param):
    '''
    ########################################
    Inputs:
        train_data: DataFrame with three columns ['user_id', 'book_id', 'rating']
        rank, reg_param:  Model hyperparameters
    
    Output:
        ALS model object
    ########################################
    '''

    # model hyperparameters
    rank = rank
    reg_param = reg_param
    non_negative = True

    # optimization hyperparameters
    seed = 401
    max_iter = 10
    user_blocks = 10
    item_blocks = 10
    checkpoint_interval = 10
    # sc.setCheckpointDir(f'hdfs:/user/{netid}/recommend/__checkpoint__/') # Need to set CheckpointDir if checkpoint_interval > 10

    als = ALS(rank=rank, maxIter=max_iter, regParam=reg_param, 
              numUserBlocks=user_blocks, numItemBlocks=item_blocks, 
              implicitPrefs=False, alpha=1.0, 
              userCol='user_id', itemCol='book_id', seed=seed, 
              ratingCol='rating', nonnegative=non_negative, 
              checkpointInterval=checkpoint_interval, 
              intermediateStorageLevel='MEMORY_AND_DISK', 
              finalStorageLevel='MEMORY_AND_DISK',
              coldStartStrategy='drop')
    model = als.fit(train_data)

    return model


def baseline_evaluate(model, val_data):
    '''
    ########################################
    Inputs:
        model: ALSmodel object
        val_data_path: Path of validation data

    Output:
        Tuple of three elements (PK, MAP, NDCG),
        which are the three ranking metrics of the input als_model
    ########################################
    '''
    
    K = 500

    # Get val_user list
    val_user = val_data.select('user_id').distinct()

    # Get ground truth list
    ground_truth = val_data.groupBy("user_id") \
                  .agg(F.collect_set("book_id") \
                  .alias('true_books'))

    # Construct recommendation list for val users
    rec = model.recommendForUserSubset(val_user, K)

    # Reformat recommendation list
    rec_udf = udf(lambda l: [i[0] for i in l], ArrayType(IntegerType()))
    rec = rec.select("user_id", rec_udf(col("recommendations")).alias('recommendations'))

    # Evaluate
    prediction_and_labels = rec.join(ground_truth, on='user_id', how='inner').drop('user_id').rdd

    metrics = RankingMetrics(prediction_and_labels)
    PK = metrics.precisionAt(K)
    MAP = metrics.meanAveragePrecision
    NDCG = metrics.ndcgAt(K)

    return PK, MAP, NDCG



######################################################################################################
def build_annoy_tree(als_model, n_trees=30, seed=401):

    #--- 1. Convert item factors to annoy index -------------------------------------------
    ##########################################################################
    item_factors = als_model.itemFactors

    window = Window.orderBy('id')
    item_factors = item_factors.withColumn('annoy_id', F.row_number().over(window))

    annoy_index_map = item_factors.select('id', 'annoy_id')     # original book_id vs annoy_index map
    item_factors = item_factors.select('annoy_id', 'features')  # item_factors with annoy index

    #--- 2. Build annoy tree -------------------------------------------
    ##########################################################################
    t = AnnoyIndex(als_model.rank, 'dot')

    for row in item_factors.collect():
        t.add_item(row.annoy_id, row.features)

    t.set_seed(seed)
    t.build(n_trees)

    return t, annoy_index_map



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


