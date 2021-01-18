import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.column import *
from pyspark.sql import window
from pyspark.ml.feature import *
from pyspark.ml.regression import *
from pyspark.ml.recommendation import ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from baseline import *


def main(K, val_data_path, als_model_path):
	val_data = spark.read.parquet(val_data_path)
	als_model = ALSModel.load(als_model_path)
	PK, MAP, NDCG = baseline_evaluate(als_model, val_data)
	print('PK: {}'.format(PK))
	print('MAP: {}'.format(MAP))
	print('NDCG: {}'.format(NDCG))


if __name__ == '__main__':
	print("Please make sure the selected ALS model's ItemFactor are replaced by the new ItemFactor generated in the training! ")
	# hfs -rm -r recommend/baseline_model/d0.1_K500_maxIter100_l10_reg0.01_cold_start_lr/itemFactors
	# hfs -cp recommend/cold_start/lr_itemfactor.parquet recommend/baseline_model/d0.1_K500_maxIter100_l10_reg0.01_cold_start_lr/itemFactors

	spark = SparkSession.builder.appName('cold_start_evaluate').getOrCreate()
	spark.conf.set('spark.sql.pivotMaxValues', '100000')
	spark.conf.set('spark.sql.broadcastTimeout', '36000')
	# if OutofMemory error occurs, uncomment the following line
	# spark.conf.set('spark.sql.autoBroadcastJoinThreshold','-1')

	K = sys.argv[1] # 500
	val_data_path = sys.argv[2] # 'hdfs:/user/fz477/recommend/10.0_val.parquet'
	als_model_path = sys.argv[3] # 'hdfs:/user/fz477/recommend/baseline_model/d0.1_K500_maxIter100_l10_reg0.01_cold_start_lr_lasso'

	main(K, val_data_path, als_model_path)

