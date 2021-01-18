import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.column import *
from pyspark.sql import window
from pyspark.ml.feature import *
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from textblob import TextBlob, Word
from graphframes import *

# read supplemental book data
def load_book(book_path):
	books = spark.read.parquet(book_path)
	books = books.withColumn('ratings_count',books['ratings_count'].cast(IntegerType()))
	books = books.withColumn('average_rating',books['average_rating'].cast(FloatType()))
	return books

# extract books' popular shelves information from books' supplemental data
def extract_popular_shelves(book):
	books_popular_shelves = books.select(col('book_id'),explode(col('popular_shelves')))
	books_popular_shelves = books_popular_shelves.select('book_id','col.*')
	books_popular_shelves = books_popular_shelves.withColumn('count',books_popular_shelves['count'].cast(IntegerType()))
	return books_popular_shelves

# preprocess popular shelves phrases (grouping similar phrases together)
def clean_array_list(x):
	res = []
	for _ in x:
		if (_ != '') & (_ != 'book'):
			res.append(_)
	return res

def str_to_array(x):
	res = []
	if ',' in x[1:-1]:
		for _ in x[1:-1].split(','):
			if _.strip() != '':
				res.append(_.strip())
	else:
		res.append(x[1:-1].strip())
	return res

def calculate_jaccard_similarity(books_popular_shelves, save_stage, save_stage_path):
	bps_name = books_popular_shelves.withColumn('name_1',remove_chr('name'))
	bps_name = bps_name.withColumn('name_array',split(col('name_1'),'-'))
	bps_name = bps_name.withColumn('clean_name_array',token_lemma('name_array'))
	bps_name = bps_name.withColumn('clean_name_array',clean_array(col('clean_name_array')))
	bps_name_distinct = bps_name.select('clean_name_array').distinct()
	bps_name_distinct = bps_name_distinct.withColumn('id',monotonically_increasing_id())
	bps_name = bps_name.alias('i').join(bps_name_distinct.alias('j'), on='clean_name_array', how='left')
	
	if save_stage:
		bps_name.write.parquet(save_stage_path+'/bps_clean_name.parquet')
		bps_name_distinct.write.parquet(save_stage_path+'/bps_clean_name_array.parquet')

	bps_name_distinct_new = bps_name_distinct.select('id').alias('i').join(bps_name_distinct.select('id').alias('j'), col('i.id')<col('j.id'))
	bps_name_distinct_dict = bps_name_distinct.rdd.map(lambda x: (x[1],x[0])).collectAsMap()
	dict_fun = udf(lambda x: bps_name_distinct_dict[x])
	bps_name_distinct_new = bps_name_distinct_new.withColumn('i', dict_fun(col('i.id'))).withColumn('j', dict_fun(col('j.id')))
	bps_name_distinct_new = bps_name_distinct_new.withColumn('i',string_to_array(col('i'))).withColumn('j',string_to_array(col('j')))
	bps_name_distinct_new = bps_name_distinct_new.withColumn('intersect',array_intersect(col('i'),col('j'))).withColumn('union',array_union(col('i'),col('j')))
	bps_name_distinct_new = bps_name_distinct_new.withColumn('fraction',size(col('intersect'))/size(col('union')))
	bps_name_distinct_new = bps_name_distinct_new.select(col('i.id').alias('i_id'),col('j.id').alias('j_id'),col('i'),col('j'),col('intersect'),col('union'),col('fraction'))
	return bps_name_distinct, bps_name_distinct_new, bps_name

def find_similar_item(bps_name_distinct, bps_name_distinct_new, bps_name, sim_thres, count_thres, checkpoint_path, ps_path, save_stage, save_stage_path):
	'''
	Use graph network to find connected popular shelves based on Jaccard similarity (vertices: popular shelves; edge: Jaccard similarity)
	bps_name_distinct: vertices dataframe
	bps_name_distinct_new: edge dataframe
	sim_thres: threshold to group popular shelves together (above threshold)
	count_thres: threshold to keep popular shelves (popular shelves need to appear in at least count_thres different books to get kept)
	checkpoint_path: save path for checkpoint in graphframe network
	ps_path: save path for final selected popular shelves
	'''
	spark.sparkContext.setCheckpointDir(checkpoint_path)
	v = bps_name_distinct
	e = bps_name_distinct_new.select(col('i_id').alias('src'),col('j_id').alias('dst'),col('fraction')).where(col('fraction')>sim_thres)
	g = GraphFrame(v,e)
	result = g.connectedComponents()

	if save_stage:
		result.write.parquet(save_stage_path+'/connected_component.parquet')

	phrase_dict = result.rdd.map(lambda x: (x['clean_name_array'],x['component'])).collectAsMap()
	phrase_dict_fun = udf(lambda x: phrase_dict[x])
	bps_name = bps_name.withColumn('new_feature', phrase_dict_fun(col('clean_name_array')))
	bps_name = bps_name.join(bps_name.groupby('new_feature').count().where(col('count')>count_thres).select('new_feature'), on='new_feature', how='inner')
	bps_name.groupby('book_id').pivot('new_feature').sum('count').fillna(0).write.parquet(ps_path)

def preprocess_similar_books(books, model_path, sb_path):
	'''
	extract books' similar books information from books' supplemental data
	model_path: path for selected ALS model's itemFactor matrix to calculate average of similar books' latent factors
	sb_path: save path for latent factor generated from the similar books
	'''
	similar_book = books.select('book_id',explode('similar_books').alias('similar_book'))
	itemfactor = spark.read.parquet(model_path)
	similar_book = similar_book.alias('i').join(itemfactor.alias('j'), on=similar_book['similar_book']==itemfactor['id'], how='left')
	latent_length = len(similar_book.select('features').first()[0])
	similar_book = similar_book.groupBy('book_id').agg(array(*[avg(col('features')[i]) for i in range(latent_length)]).alias('averages'))
	similar_book.write.parquet(sb_path)


def main(book_path, checkpoint_path, ps_path, model_path, sb_path, sim_thres, count_thres, save_stage=False, save_stage_path=None)
	books = load_book(book_path)
	books_popular_shelves = extract_popular_shelves(books)

	stop_words = stopwords.words('english')
	porter = PorterStemmer()
	lemmatizer = WordNetLemmatizer()
	remove_chr = udf(lambda x: x.replace(' ','').lstrip('0'))
	token_lemma = udf(lambda x: [Word(porter.stem(w)).lemmatize() for w in x if not w in stop_words])
	clean_array = udf(lambda x: clean_array_list(x))
	string_to_array = udf(str_to_array, ArrayType(StringType()))

	bps_name_distinct, bps_name_distinct_jaccard, bps_name_clean = calculate_jaccard_similarity(books_popular_shelves, save_stage=save_stage, save_stage_path=save_stage_path)
	find_similar_item(bps_name_distinct, bps_name_distinct_jaccard, bps_name_clean, sim_thres, count_thres, checkpoint_path, ps_path, save_stage=save_stage, save_stage_path=save_stage_path)
	preprocess_similar_books(books, model_path, sb_path)

if __name__ == '__main__':
	print('please run following commented line to install corpora first')
	print('import nltk')
	print("nltk.download('wordnet')")
	print('python -m textblob.download_corpora')
	print('python -m nltk.downloader')

	spark = SparkSession.builder.appName('cold_start_preprocess').getOrCreate()
	spark.conf.set('spark.sql.pivotMaxValues', '100000')
	spark.conf.set('spark.sql.broadcastTimeout', '36000')
	# if OutofMemory error occurs, uncomment the following line
	# spark.conf.set('spark.sql.autoBroadcastJoinThreshold','-1')

	book_path = sys.argv[1] # 'hdfs:/user/fz477/recommend/books.parquet'
	checkpoint_path = sys.argv[2] # 'hdfs:/user/fz477/recommend/checkpoint'
	ps_path = sys.argv[3] # 'hdfs:/user/fz477/recommend/cold_start/popular_shelves_1000.parquet'
	model_path = sys.argv[4] # 'hdfs:/user/fz477/recommend/baseline_model/d0.1_K500_maxIter100_l10_reg0.01_29-Apr-2020-12-20-28.936480/itemFactors'
	sb_path = sys.argv[5] # 'hdfs:/user/fz477/recommend/cold_start/similar_book_l10.parquet'
	sim_thres = sys.argv[6] # 0.7
	count_thres = sys.argv[7] # 1000
	save_stage = sys.argv[8] # True
	save_stage_path = sys.argv[9] # 'hdfs:/user/fz477/recommend/cold_start'

	main(book_path, checkpoint_path, ps_path, model_path, sb_path, sim_thres, count_thres, save_stage, save_stage_path)



