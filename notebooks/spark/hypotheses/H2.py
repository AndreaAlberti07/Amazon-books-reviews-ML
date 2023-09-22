# Import libraries
import nltk
from pyspark.ml import Pipeline

nltk.download('stopwords')

from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

# Initialize spark
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer

# Import necessary libraries
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder.master('local[*]').config("spark.driver.memory", "4g").appName("hypothesis_2").getOrCreate()

# Define the schema
ratings_schema = StructType([
    StructField("Title", StringType(), True),
    StructField("Price", FloatType(), True),
    StructField("User_id", IntegerType(), True),
    StructField("profileName", StringType(), True),
    StructField("review/score", FloatType(), True),
    StructField("review/time", IntegerType(), True),
    StructField("review/summary", StringType(), True),
    StructField("review/text", StringType(), True),
    StructField("N_helpful", IntegerType(), True),
    StructField("Tot_votes", IntegerType(), True)
])

# Load your DataFrame (assuming you have it in a variable df)
# Load the data
df_ratings = spark.read.csv('hdfs://localhost:9900/user/book_reviews/books_rating_cleaned.csv', header=True, schema=ratings_schema, sep='\t')
df_ratings.show(5)

#random select 5000 rows of df_ratings
#df_ratings = df_ratings.sample(withReplacement = False, fraction = 5000/df_ratings.count(), seed = 42)

# Filter out the data
df_ratings_filtered = df_ratings.filter(df_ratings['review/text'].isNotNull())
df_ratings_filtered = df_ratings_filtered.filter(df_ratings_filtered['review/score'] != 3)
df_ratings_filtered = df_ratings_filtered.filter(df_ratings_filtered['Tot_votes'] != 0)

# Remove punctuation and convert to lowercase the review/text column
df_ratings_filtered = df_ratings_filtered.withColumn('review/text', lower(regexp_replace('review/text', r'[!"#$%&\'()*+,-./:;<=>?@\\^_`{|}~]', ' ')))
df_ratings_filtered.show(5)

# remove words with length less than 2
df_ratings_filtered = df_ratings_filtered.withColumn('review/text', regexp_replace('review/text', r'\b\w{1,2}\b', ' '))


# Add the helpfulness ratio column
df_ratings_filtered = df_ratings_filtered.withColumn('helpfulness_ratio', df_ratings_filtered['N_helpful']/df_ratings_filtered['Tot_votes']*sqrt(df_ratings_filtered['Tot_votes']))

# Add the class column
df_ratings_filtered = df_ratings_filtered.withColumn('class', when(df_ratings_filtered['review/score'] >= 4, 1).otherwise(0))

# Retain only the required columns
df_ratings_selected = df_ratings_filtered.select('review/text', 'helpfulness_ratio', 'class')
df_ratings_selected.show(5)

# Select relevant columns and handle missing values
df = df_ratings_selected.select("class", "helpfulness_ratio", "review/text").na.drop()
df.show(5)







# Tokenize the 'review/text' column
tokenizer = Tokenizer(inputCol="review/text", outputCol="words")

# Remove stopwords
stop_words_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")



# Convert words to a BoW feature vector
vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="features")

# Create a Naive Bayes model
nb = NaiveBayes(labelCol="class", featuresCol="features", predictionCol="prediction")

# Create a pipeline
pipeline = Pipeline(stages=[tokenizer, stop_words_remover, vectorizer, nb])

# Fit the pipeline on your data
model = pipeline.fit(df)

# save the model
#model.save("hdfs://localhost:9900/user/book_reviews/model")

# load the model
#from pyspark.ml import PipelineModel
#model = PipelineModel.load("hdfs://localhost:9900/user/book_reviews/model")

# Fit the Multinomial Naive Bayes model on the training data
nb_model = model.stages[-1]

# Get the vocabulary
vocabulary = model.stages[2].vocabulary

# Get the word probabilities for class 1
class_1_probs = nb_model.theta.toArray()[1]

# Create a DataFrame of words and probabilities
import pandas as pd
results = pd.DataFrame({'word': vocabulary, 'prob': class_1_probs})




# Calculate the difference in word probabilities between class 1 and class 0
class_0_probs = nb_model.theta.toArray()[0]
class_1_probs = nb_model.theta.toArray()[1]
pos_neg_ratio = class_1_probs - class_0_probs

# Create a DataFrame with words and their positive-to-negative probability ratio
results = pd.DataFrame({'word': vocabulary, 'pos_neg_ratio': pos_neg_ratio})

# Sort the DataFrame in descending order of 'pos_neg_ratio'
positive_words = results.sort_values(by='pos_neg_ratio', ascending=False)

# Get the top N most positive words (e.g., top 10)
top_positive_words = positive_words.head(800)  # Change 10 to the desired number of words you want to see

# Display the top positive words
print(top_positive_words)

# Get the top N most negative words (e.g., top 10)
top_negative_words = positive_words.tail(800)  # Change 10 to the desired number of words you want to see

# Display the top negative words
print(top_negative_words)










'''
# Sort the DataFrame by descending word probabilities and take top 2000
results = results.sort_values(by='prob', ascending=False).head(2000)
results
'''

# Create a list of the top 20 words
top_positive_words = results['word'].tolist()[:800]

# Tokenize the review/text column into words
words_df = df.withColumn("words", split(col("review/text"), "\\s+"))
words_df.show(5)

#create a pyspark list with the occurrences of the top 20 words in each review
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

def count_words(words):
    return len([word for word in words if word in top_positive_words])

count_words_udf = udf(count_words, IntegerType())
words_df = words_df.withColumn("count", count_words_udf(col("words")))
words_df.show(5)

# calculate and print the correlation between the count and the helpfulness ratio
from pyspark.sql.functions import corr
words_df.select(corr("count", "helpfulness_ratio")).show()



#random select 3000 rows of words_df and convert it to pandas dataframe
df_camp = words_df.sample(withReplacement = False, fraction = 3000/words_df.count(), seed = 42)
df_camp.show(5)

# words_df to pandas dataframe
words_df_pd = df_camp.toPandas()

#scatter plot of count vs helpfulness ratio
import matplotlib.pyplot as plt
plt.scatter(words_df_pd['count'], words_df_pd['helpfulness_ratio'], alpha=0.2)
plt.xlabel('Number of positive words')
plt.ylabel('Helpful score')
plt.show()

# Create 5 bins of positive_words
groups = [0, 10, 20, 50, 75, 100, 125, 150, 175, 200]
words_df_pd['length_bin'] = pd.cut(words_df_pd['count'], bins=groups, labels=[group for group in groups[1:]])

import seaborn as sns
import scipy
# Plot the distribution of positive_words with respect to helpfulness rate
plt.figure(figsize=(15, 10))
sns.boxplot(x='length_bin', y='helpfulness_ratio', data=words_df_pd, palette='rainbow')
plt.title('Review Length Range vs Helpfulness Rate')

for el in groups[1:]:
    dataframe = words_df_pd[words_df_pd['length_bin'] == el]
    corr, pval = scipy.stats.kendalltau(dataframe['count'], dataframe['helpfulness_ratio'])
    print(f'Group number: {el}\nCorrelation Coefficient: {corr}\nP-value: {pval}\n')
    plt.figure(figsize=(15, 10))
    dataframe.plot(kind='scatter', x='count', y='helpfulness_ratio', figsize=(15, 10), title=f'Review Length vs Helpfulness Rate in Group {el}')
    plt.show()



#close spark
spark.stop()
