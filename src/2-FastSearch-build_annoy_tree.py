#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import psutil
import math
from annoy import AnnoyIndex

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from pyspark.sql.window import Window


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


def main(spark, als_model_path, tree_path, index_map_path, n_trees):

    # Load ALS model
    als_model = ALSModel.load(als_model_path)

    # Build annoy tree
    t, annoy_index_map = build_annoy_tree(als_model, n_trees=n_trees)

    # Save annoy tree & annoy index map
    t.save(tree_path)
    print(f'Annoy tree saved at: {tree_path}')

    annoy_index_map.write.parquet(index_map_path)
    print(f'Annoy index map saved at: {index_map_path}')


if __name__ == '__main__':

    # Path to load ALS model
    als_model_path = sys.argv[1]

    # Path where annoy_tree & index_map will be saved
    tree_path = sys.argv[2]
    index_map_path = sys.argv[3]

    # Hyperparameter for annoy tree
    try:
        n_trees = int(sys.argv[4])
    except:
        n_trees = 10

    # Create Spark session
    # memory = f'{math.floor(psutil.virtual_memory()[1]*.9) >> 30}g'
    memory = '10g'
    spark = (SparkSession.builder
             .appName('build_annoy_tree')
             .master('yarn')
             .config('spark.executor.memory', memory)
             .config('spark.driver.memory', memory)
             .getOrCreate())
    # spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

    main(spark, als_model_path, tree_path, index_map_path, n_trees)

