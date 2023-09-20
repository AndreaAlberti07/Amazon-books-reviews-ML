# # Implementation of hypothesis 2 using pyspark

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

# Load the data
df_ratings = spark.read.csv('hdfs://localhost:9900/user/book_reviews/books_rating_cleaned.csv', header=True, schema=ratings_schema, sep='\t')
df_ratings.show(5)

# Filter out the data
df_ratings_filtered = df_ratings.filter(df_ratings['review/text'].isNotNull())
df_ratings_filtered = df_ratings_filtered.filter(df_ratings_filtered['review/score'] != 3)
df_ratings_filtered = df_ratings_filtered.filter(df_ratings_filtered['Tot_votes'] != 0)

# Add the helpfulness ratio column
df_ratings_filtered = df_ratings_filtered.withColumn('helpfulness_ratio', df_ratings_filtered['N_helpful']/df_ratings_filtered['Tot_votes']*sqrt(df_ratings_filtered['Tot_votes']))

# Add the class column
df_ratings_filtered = df_ratings_filtered.withColumn('class', when(df_ratings_filtered['review/score'] >= 4, 1).otherwise(0))

# Retain only the required columns
df_ratings_selected = df_ratings_filtered.select('review/text', 'helpfulness_ratio', 'class')
df_ratings_selected.show(5)

# Remove punctuation
df_ratings_selected = df_ratings_selected.withColumn('review/text', lower(regexp_replace('review/text', r'[!"#$%&\'()*+,-./:;<=>?@\\^_`{|}~]', ' ')))
df_ratings_selected.show(5)

# Remove stopwords
tokenizer = Tokenizer(inputCol='review/text', outputCol='review/text_tokenized')
df_ratings_selected = tokenizer.transform(df_ratings_selected)
remover = StopWordsRemover(inputCol='review/text_tokenized', outputCol='review/text_tokenized_filtered', stopWords=stopwords)
df_ratings_selected = remover.transform(df_ratings_selected)
df_ratings_selected.show(5)

# Remove words with length less than 2
#df_ratings_selected = df_ratings_selected.withColumn('review/text_tokenized_filtered', regexp_replace('review/text_tokenized_filtered', r'\b\w{1,2}\b', ''))
#df_ratings_selected.show(5)

# Remove empty rows
#df_ratings_selected = df_ratings_selected.filter(df_ratings_selected['review/text_tokenized_filtered'] != '')
#df_ratings_selected.show(5)


from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

vectorizer = CountVectorizer(inputCol='review/text_tokenized_filtered', outputCol='features', vocabSize=2000)
df_ratings_selected = vectorizer.fit(df_ratings_selected).transform(df_ratings_selected)
df_ratings_selected.show(5)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = df_ratings_selected.randomSplit([0.7, 0.3], seed=100)

# Train a Naive Bayes model.
nb = NaiveBayes(smoothing=1.0, modelType='multinomial', labelCol='class', featuresCol='features')
model = nb.fit(trainingData)


# Save the model
#model.save('hdfs://localhost:9900/user/book_reviews/model_nb')

# Load the model
#from pyspark.ml.classification import NaiveBayesModel
#model = NaiveBayesModel.load('hdfs://localhost:9900/user/book_reviews/model_nb')


# Select example rows to display.
predictions = model.transform(testData)
predictions.select('class', 'prediction', 'probability').show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol='class', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print('Test set accuracy = ' + str(accuracy))

#voc = model.stage[2].vocabulary
#print(voc)

