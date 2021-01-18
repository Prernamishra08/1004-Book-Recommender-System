#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics


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
    # sc.setCheckpointDir(f'hdfs:/user/{netid}/recommend/__checkpoint__/')

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
        val_data: (DataFrame) Validation data

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

