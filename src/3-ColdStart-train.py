import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.column import *
from pyspark.sql import window
from pyspark.ml.feature import *
from pyspark.ml.regression import *

def load_and_split(model_path, ps_path, sb_path):
	"""
	model_path: path for selected ALS model's itemFactor matrix to train regression model
	ps_path: save path for final selected popular shelves
	sb_path: save path for latent factor generated from the similar books
	"""
	itemfactor = spark.read.parquet(model_path)
	train_item, test_item = itemfactor.select('id').randomSplit(weights=[0.7,0.3], seed=401)
	itemfactor_train = itemfactor.join(train_item, on='id', how='inner')
	itemfactor_test = itemfactor.join(test_item, on='id', how='inner')

	# load popular shelves pivot table
	ps = spark.read.parquet(ps_path)
	ps = ps.na.fill(0)

	assembler = VectorAssembler(inputCols=ps.schema.names[1:], outputCol='book_features')
	ps_new = assembler.transform(ps).select('book_id','book_features')
	model_train_data = itemfactor_train.alias('i').join(ps_new.alias('j'), on=itemfactor_train['id']==ps_new['book_id'], how='inner')
	model_test_data = itemfactor_test.alias('i').join(ps_new.alias('j'), on=itemfactor_test['id']==ps_new['book_id'], how='inner')

	latent_length = len(model_train_data.select('features').collect()[1].features)
	model_train_data = model_train_data.select([model_train_data.id, model_train_data.book_features]+[model_train_data.features[i].cast(DoubleType()).alias('feature'+str(i)) for i in range(latent_length)])
	model_test_data = model_test_data.select([model_test_data.id, model_test_data.book_features]+[model_test_data.features[i].cast(DoubleType()).alias('feature'+str(i)) for i in range(latent_length)])
	
	# load similar books
	similar_book = spark.read.parquet(sb_path)
	similar_book = similar_book.select([similar_book.book_id, similar_book.averages]+[similar_book.averages[i].alias('averages'+str(i)) for i in range(latent_length)])

	return itemfactor_train, itemfactor_test, latent_length, model_train_data, model_test_data, similar_book

# train the lasso regression
def train_lasso(latent_length, model_train_data, model_test_data, lasso_save_path, model_save_path):
	lr_lst = []
	for _ in range(latent_length):
		lr = LinearRegression(featuresCol='book_features',labelCol='feature'+str(_),elasticNetParam=1.0,regParam=0.01)
		lr_model = lr.fit(model_train_data)
		lr_model.save(lasso_save_path+'/lr_lasso'+str(_))
		lr_lst.append(lr_model)
		print('lr_lasso'+str(_)+' finished')

	for _ in range(latent_length):
		model_test_data = lr_lst[_].evaluate(model_test_data).predictions.withColumnRenamed('prediction','prediction'+str(_))

	model_test_data.write.parquet(model_save_path+'/lr_lasso_evaluation.parquet')

# train the regular linear regression
def train_lr(latent_length, model_train_data, model_test_data, lr_save_path, model_save_path):
	lr_lst = []
	for _ in range(latent_length):
		lr = LinearRegression(featuresCol='book_features',labelCol='feature'+str(_))
		lr_model = lr.fit(model_train_data)
		lr_model.save(lr_save_path+'/lr'+str(_))
		lr_lst.append(lr_model)
		print('lr'+str(_)+' finished')

	for _ in range(latent_length):
		model_test_data = lr_lst[_].evaluate(model_test_data).predictions.withColumnRenamed('prediction','prediction'+str(_))

	model_test_data.write.parquet(model_save_path+'/lr_evaluation.parquet')

# combine similar book with popular shelves to create new item factors
def new_itemfactor(itemfactor_train, itemfactor_test, model_test_data, similar_book, latent_length, new_itemfactor_save_path):
	predict_test = itemfactor_test.select('id').join(model_test_data, on='id', how='left')
	predict_test = predict_test.join(similar_book, on=predict_test['id']==similar_book['book_id'], how='left')
	predict_test = predict_test.select(['id']+['averages'+str(_) for _ in range(latent_length)]+['prediction'+str(_) for _ in range(latent_length)]).fillna(0)
	for _ in range(latent_length):
		predict_test = predict_test.withColumn('sum'+str(_),0.4*col('averages'+str(_))+0.6*col('prediction'+str(_)))

	itemfactor_predict = predict_test.select(col('id'),array(['sum'+str(_) for _ in range(latent_length)]).cast(ArrayType(FloatType())).alias('features'))
	itemfactor_train.union(itemfactor_predict).write.parquet(new_itemfactor_save_path)


def main(model_path, ps_path, sb_path, regression_save_path, model_save_path, new_itemfactor_save_path):
	itemfactor_train, itemfactor_test, latent_length, model_train_data, model_test_data, similar_book = load_and_split(model_path, ps_path, sb_path)
	train_lasso(latent_length, model_train_data, model_test_data, regression_save_path, model_save_path)
	train_lr(latent_length, model_train_data, model_test_data, regression_save_path, model_save_path)
	new_itemfactor(itemfactor_train, itemfactor_test, model_test_data, similar_book, latent_length, new_itemfactor_save_path)


if __name__ == '__main__':
	spark = SparkSession.builder.appName('cold_start_training').getOrCreate()
	spark.conf.set('spark.sql.pivotMaxValues', '100000')
	spark.conf.set('spark.sql.broadcastTimeout', '36000')
	# if OutofMemory error occurs, uncomment the following line
	# spark.conf.set('spark.sql.autoBroadcastJoinThreshold','-1')

	model_path = sys.argv[1] # 'hdfs:/user/fz477/recommend/baseline_model/d0.1_K500_maxIter100_l10_reg0.01_29-Apr-2020-12-20-28.936480/itemFactors'
	ps_path = sys.argv[2] # 'hdfs:/user/fz477/recommend/cold_start/popular_shelves_1000.parquet'
	sb_path = sys.argv[3] # 'hdfs:/user/fz477/recommend/cold_start/similar_book_l10.parquet'
	regression_save_path = sys.argv[4] # 'hdfs:/user/fz477/recommend/baseline_model'
	model_save_path = sys.argv[5] # 'hdfs:/user/fz477/recommend/cold_start'
	new_itemfactor_save_path = sys.argv[6] # 'hdfs:/user/fz477/recommend/cold_start/lr_lasso_itemfactor.parquet'

	main(model_path, ps_path, sb_path, regression_save_path, model_save_path, new_itemfactor_save_path)


