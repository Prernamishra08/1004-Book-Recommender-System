#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from my_util import *
from pyspark.sql import SparkSession
from annoy import AnnoyIndex
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime


def main(spark, data_path):

    data = spark.read.parquet(data_path)

    rank = 10
    reg_param = 0.01
    frac_list = [0.1, 0.3, 0.5, 0.75]
    result = []

    # Create file to save result
    timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S-%f")
    fileName = 'compare_result-' + timestampStr +'.txt'
    with open(fileName, 'w') as f:
        f.write(timestampStr+'\n\n')
    
    for frac in frac_list:
        sample_data = down_sample_items(data, frac)
        train_data, val_data = split(sample_data)
        als_model = train_als(train_data, rank, reg_param)

        # Baseline evaluation
        start_time = time.time()
        bPK, bMAP, bNDCG = baseline_evaluate(als_model, val_data)
        baseline_time = time.time() - start_time

        # Annoy evaluation
        n_trees = 10
        start_time = time.time()
        annoy_tree, annoy_index_map = build_annoy_tree(als_model, n_trees=n_trees)
        annoy_train_time = time.time() - start_time

        annoy_tree_path = '/Users/lizhong/OneDrive/New York University/Spring 2020/1004/final_project/final-project-401/workplace/annoy_test/__annoycache__/'
        annoy_tree_path = annoy_tree_path + str(frac) + '.ann'
        annoy_tree.save(annoy_tree_path)

        n_candidates = 1000
        start_time = time.time()
        aPK, aMAP, aNDCG = fast_evaluate_with_annoy(n_candidates, als_model, val_data, annoy_tree_path, annoy_index_map)
        annoy_query_time = time.time() - start_time

        num_items = train_data.select('book_id').distinct().count()

        resultStr = f'Frac: {frac}; Num: {num_items}\n\
                      \n\
                      Baseline: PK {bPK}, MAP {bMAP}, NDCG{bNDCG}\n\
                      BaselineTime: {baseline_time}\n\
                      \n\
                      Annoy: PK {aPK}, MAP {aMAP}, NDCG{aNDCG}\n\
                      AnnoyTime: train_time {annoy_train_time}, query_time {annoy_query_time}\n\
                      \n'

        # Print & Save result
        print(resultStr)
        with open(fileName, 'a') as f:
            f.write(resultStr)

        result.append(resultStr)

    print('\n### Complete!')    
    return

if __name__ == '__main__':
    
    # We used History genre data
    data_path = sys.argv[1]
    
    spark = (SparkSession.builder
             .appName('annoy_performance_test')
             .master('local[4]')
             .getOrCreate())

    main(spark, data_path)

