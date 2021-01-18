from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    spark = SparkSession.builder.appName('preprocess').getOrCreate()

    # Read data
    df = spark.read.csv('hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv',schema = 'user_id INT,book_id INT,is_read INT,rating INT,is_reviewed INT',header=True)
    user_map = spark.read.csv('hdfs:/user/bm106/pub/goodreads/user_id_map.csv',schema='user_id INT, user_full_id STRING',header=True)
    book_map = spark.read.csv('hdfs:/user/bm106/pub/goodreads/book_id_map.csv',schema='book_id INT, book_full_id STRING',header=True)

    # Select columns of interest & drop items with no rating at all from any user
    df = df.select(['user_id', 'book_id', 'rating'])
    df = df.filter(df.rating > 0)

    # Disregard users with fewer than 10 interactions
    users_to_keep = df.groupBy('user_id').agg(F.count('rating').alias('num_rating')).filter(F.column('num_rating')>10).drop('num_rating')
    df = df.join(users_to_keep, on="user_id",how='inner')
    df = df.filter(df.rating > 0)  # Drop books with no rating score from any user

    # Update user_map & book_map
    user_map = user_map.join(users_to_keep, on="user_id",how='inner')
    book_map = book_map.join(df.select('book_id').distinct(), on="book_id",how='inner')

    # Save data into parquet files
    df.write.parquet('hdfs:/user/lw2350/recommend/goodreads_interactions.parquet')
    user_map.write.parquet('hdfs:/user/lw2350/recommend/user_id_map.parquet')
    book_map.write.parquet('hdfs:/user/lw2350/recommend/book_id_map.parquet')


if __name__ == '__main__':
    main()

