# **WORKFLOW**

## 1. DISCOVERY

### 1.1. Team

The team is composed by three members:

- **Andrea Alberti** ([GitHub](https://github.com/AndreaAlberti07))
- **Davide Ligari** ([GitHub](https://github.com/DavideLigari01))
- **Cristian Andreoli** ([GitHub](https://github.com/CristianAndreoli94))

### 1.2. Tools

To deal with the Big Data the following tools will be used:

- [Hadoop](https://hadoop.apache.org/)
- [Spark](https://spark.apache.org/)
- [MongoDB](https://www.mongodb.com/)

To analyze the data the following tools will be used together with Python:

- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Jupyter Notebook](https://jupyter.org/)

### 1.3. Framing

Goal of the project is to develop a scalable solution to analyze a dataset of reviews of books from Amazon and fit a ML model on them. Possible goals are:

- Sentiment Analysis
- Review Helpfulness Prediction (Chosen by us)
- Topic Modeling
- Recommendation System
- Common Issues Analysis
- Category Prediction

### 1.4. Data Collection

The chosen dataset is [Amazon Books Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews).
3 million reviews spanning May 1996 - July 2014. The dataset is composed by 2 tables:

### - Books Ratings table

| Column                 | Description                                 |
| ---------------------- | ------------------------------------------- |
| **Id**                 | The ID of the book                          |
| **Title**              | The title of the book                       |
| **Price**              | The price of the book                       |
| **User_id**            | The ID of the user who rated the book       |
| **profileName**        | The name of the user who rated the book     |
| **review/helpfulness** | Helpfulness rating of the review (e.g. 2/3) |
| **review/score**       | Rating from 0 to 5 for the book             |
| **review/time**        | Time the review was given                   |
| **review/summary**     | The summary of the text review              |
| **review/text**        | The full text of the review                 |

### - Books Info table

| Column            | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| **Title**         | Book title                                                  |
| **description**   | Description of the book                                     |
| **authors**       | Name of book authors                                        |
| **image**         | URL for book cover                                          |
| **previewLink**   | Link to access this book on Google Books                    |
| **publisher**     | Name of the publisher                                       |
| **publishedDate** | The date of publish                                         |
| **infoLink**      | Link to get more information about the book on Google Books |
| **categories**    | Genres of books                                             |
| **ratingsCount**  | Number of ratings and reviews for the book                  |

### 1.5. Hypotheses Formulation

1. **_Hypothesis_**: Reviews with longer text have higher helpfulness ratings.

   - **Metric**: Correlation coefficient (e.g., Pearson's correlation) between review length and helpfulness ratings. Plot the correlation coefficient as a function of the review length.

- **Missing Values**:

  - `review/text`: set missing values as empty string
  - `review/helpfulness`: remove the entire sample

- **Data Transformation**:

  - `review/text`: ...
  - `review/helpfulness`: $helpfulness = \frac{x}{y} \sqrt(y)$

  ***

2. **_Hypothesis_**: Reviews with more positive sentiment words receive higher helpfulness ratings.

   - **Metric**: Mean helpfulness ratings for positive and negative words.

   - **Model**: Multinomial Naive Bayes.

   - **Description**:

     - Use NBC as a classifier to predict the sentiment of a review.
     - Extract the most useful words from the classifier.
     - Compute the number of positive words in each review.

- **Missing Values**:

  - `review/score`: remove the entire sample
  - `review/text`: remove the entire sample
  - `review/helpfulness`: remove the entire sample

- **Data Transformation**:

  - `review/score`: Assign 1 to score (4, 5), 0 to score (1, 2).
  - `review/text`: Create the BoW for the text. Fit a MNBC and count the number of positive words. Graphical Plot.
  - `review/helpfulness`: $helpfulness = \frac{x}{y} \sqrt(y)$

    ***

3. **_Hypothesis_**: Reviews with higher average book ratings have higher helpfulness ratings.

   - **Metric**: Correlation between book ratings and helpfulness ratings.

- **Missing Values**:

  - `review/score`: remove the entire sample
  - `review/helpfulness`: remove the entire sample

- **Data Transformation**:

  - `review/score`: groupBy book title and calculate the average score.
  - `review/helpfulness`: $helpfulness = \frac{x}{y} \sqrt(y)$

- **Positive Bias**: people are more prone to vote a positive review, indeed there is a correlation between the number of votes and the book rating. Specifically, the number of positive reviews getting a single vote is very high and this might lead to a bias in the helpfulness score, since it is computed by this formula: 

$helpfulness = \frac{x}{y} \sqrt(y)$   where $x$ is the number of positive votes and $y$ is the total number of votes.

To face this problem, we decided to filter the data and consider only the reviews with a number of votes greater than 20.

  ***

4. **_Hypothesis_**:  
   The rating score is influenced by individual users, whose unique personalities and personal preferences may lead them to either overestimate or underestimate a book's quality. In addition, the Anonymous tends to overrate the books

   - **Metric**: ANOVA test, kolmogorov-smirnov

- **Missing Values**:

  - `profileName`: Missing values are set as "Anonymous"
  - `review/score`: The entire sample is removed.

- **Hypotheses**:
  - **H0**: The rating score is not related to the `profileName`, as all rating scores originate from the same distribution. If we consider the rating score of each user, they have the same mean and variance.

  - **H1**: The rating score is affected by the user, meaning the rating scores of each user follow a different distribution.

For the sake of consistency in this analysis, users with fewer than 20 reviews are excluded. This is because a lower number of reviews is insufficient to significantly estimate statistical measures.

  ***

5. **_Hypothesis_**:
The review/score is influenced by the category of the book
   - **Metric**: ANOVA test

- **Hypotheses**:
  - **H0**: The rating score is not related to the `categories`, as all rating scores originate from the same distribution. If we consider the rating score of each category, they have the same mean and variance.

  - **H1**: The rating score is affected by the category, meaning the rating scores of each category follow a different distribution.

For the sake of consistency in this analysis, categories with fewer than 20 reviews are excluded. This is because a lower number of reviews is insufficient to significantly estimate statistical measures.

***

6. **_Hypothesis_**: The larger the number of books published for a category, the higher the review score. (marketing strategy, the publishers tend to publish books of the most liked category). The larger the number of books published by publishers, the higher the review score (books published by the most famous publishers are preferred)

   - **Metric**: correlation coefficient

- **Missing Values**:

  - `publisher`: remove the entire sample
  - `review/score`: remove the entire sample
  - `categories`: remove the entire sample

- **Data Transformation**:

  - `categories`: GroupBy categories.
  - `publisher`: GroupBy publisher.
  - `review/score`: Compute the average review/score for each publisher and category.

---

## 2. DATA PREPARATION

### 2.1. Data Loading on HDFS

```bash
# Replace these paths with your actual paths
LOCAL_PATH="/path/to/local/files"
HDFS_PATH="/path/in/hdfs"

# Create HDFS directories if they don't exist
hdfs dfs -mkdir -p "$HDFS_PATH/ratings"
hdfs dfs -mkdir -p "$HDFS_PATH/books_info"

# Copy local files to HDFS
hdfs dfs -copyFromLocal "$LOCAL_PATH/ratings.csv" "$HDFS_PATH/ratings/"
hdfs dfs -copyFromLocal "$LOCAL_PATH/books_info.csv" "$HDFS_PATH/books_info/"
```

### 2.2. Data Cleaning with Spark

- Schema definition to better control over data

```python
# Books Ratings table schema
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType

ratings_schema = StructType([
    StructField("Id", IntegerType(), True),
    StructField("Title", StringType(), True),
    StructField("Price", FloatType(), True),
    StructField("User_id", IntegerType(), True),
    StructField("profileName", StringType(), True),
    StructField("review/helpfulness", StringType(), True),
    StructField("review/score", FloatType(), True),
    StructField("review/time", IntegerType(), True),
    StructField("review/summary", StringType(), True),
    StructField("review/text", StringType(), True)
])

# Books Info table schema
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType

info_schema = StructType([
    StructField("Title", StringType(), True),
    StructField("description", StringType(), True),
    StructField("authors", StringType(), True),
    StructField("image", StringType(), True),
    StructField("previewLink", StringType(), True),
    StructField("publisher", StringType(), True),
    StructField("publishedDate", StringType(), True),
    StructField("infoLink", StringType(), True),
    StructField("categories", StringType(), True),
    StructField("ratingsCount", IntegerType(), True)
])

# Execution
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("LoadDataWithSchema") \
    .getOrCreate()

ratings_df = spark.read.csv("path_to_ratings.csv", header=True, schema=ratings_schema)
info_df = spark.read.csv("path_to_books_info.csv", header=True, schema=info_schema)

# Continue with data processing or analysis...

spark.stop()

```

- Remove duplicates
- Deal with missing values
- Splitting the `review/helpfulness` column into two columns: `helpfulness_numerator` and `helpfulness_denominator`


### 2.3. Data Aggregation

- Program a MapReduce job to join the data by book title

- Run the MapReduce job on Hadoop

```bash
# Run a Hadoop Streaming job using the Hadoop Streaming JAR file

# Specify the Hadoop Streaming JAR file
HADOOP_STREAMING_JAR="$HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-3.3.6.jar"

# Specify the Mapper and Reducer scripts
MAPPER_SCRIPT="join_mapper.py"
REDUCER_SCRIPT="join_reducer.py"

# Specify the input data sources (CSV files to be processed)
INPUT_FILES="hdfs://localhost:9900/user/book_reviews/books_data_cleaned.csv,hdfs://localhost:9900/user/book_reviews/books_rating_cleaned.csv"

# Specify the output directory where the job results will be stored
OUTPUT_DIR="hdfs://localhost:9900/user/book_reviews/joined_tables"

# Run the Hadoop Streaming job
hadoop jar $HADOOP_STREAMING_JAR \
-D stream.num.map.output.key.fields=2 \
-D mapreduce.partition.keycomparator.options='-k1,1 -k2,2' \
-files $MAPPER_SCRIPT,$REDUCER_SCRIPT \
-mapper "$MAPPER_SCRIPT" \
-reducer "$REDUCER_SCRIPT" \
-input $INPUT_FILES \
-output $OUTPUT_DIR
```

### 2.5. Load into MongoDB

```bash
pip install pymongo pyspark
```
- Look at `import_mongoDB.ipynb`

### 2.6. Complex MongoDB query

They are two and can be found inside `hypothesis_6.ipynb`

```python
# Deal with missing values
pipeline_missing = {'$match': {
    'review/score': {'$exists': True, '$ne': 0},
    'publisher': {'$exists': True, '$ne': None},
    'categories': {'$exists': True},
}
}

# Compute average rating for each tuple category, publisher
pipeline_average_rating = {'$group': {
    '_id': {
        'category': '$categories',
        'publisher': '$publisher',
    },
    'avg_score': {'$avg': '$review/score'},
    'count': {'$sum': 1}
}
}

# Show average rating for category for each publisher
pipeline_publisher = {'$group': {
    '_id': '$_id.category',
    'avg_score/publisher': {
        '$push': {
            'publisher': '$_id.publisher',
            'avg_score': '$avg_score',
            'count': '$count'
        }
    }
}
}

# Unwind the list of categories
pipeline_unwind = {'$unwind': '$avg_score/publisher'}

# Remove categories or publisher with less than 'threshold' reviews
threshold = 0
pipeline_remove = {'$match': {
    'avg_score/publisher.count': {'$gte': threshold}
}
}

# Count the number of categories with average rating > 4.5
pipeline_counts = {'$project': {
    'category': '$_id',
    '_id': 0,
    'publisher': '$avg_score/publisher.publisher',
    'count': {
        '$sum': {
            '$cond': {

                'if': {'$gt': ['$avg_score/publisher.avg_score', 4.5]},
                'then': 1,
                'else': 0
            }
        }
    }
}
}

# Sum the results for each publisher. If Total > 10, then the hypothesis is False
pipeline_sum = {'$group': {
    '_id': '$category',
    'total': {'$sum': '$count'}
}
}

pipeline_sort = {'$sort': {
    'total': -1
}
}

results = books.aggregate([pipeline_missing, pipeline_average_rating, pipeline_publisher,
                          pipeline_unwind, pipeline_remove, pipeline_counts, pipeline_sum, pipeline_sort])

df_results = pd.DataFrame(list(results))
```

### 2.7. Features Extraction

To provide the model with the capabilities of understanding the context and detect similar words, instead of using a simple bag of words representation, we opted for a word embedding approach. In particular, we used the Word2Vec model provided by the Gensim library. The model is trained on the train set and then used to transform the reviews in a vector representation.
The model specification is:
- **vector_size**: 30 and 150
- **window**: 5
- **min_count**: 2

### 2.8. Local analysis with Python Libraries
The analysis are located inside 'notebooks/hypothesis_testing' folder.

## 3. MODEL CHOICE
Since we want to predict the helpfulness starting from the review text, we are dealing with a regression problem.
### 3.1.2. Regression
The model we compared are:
- Random Forest Regressor
- Support Vector Regressor
- Multilayer Perceptron Regressor

## 4. MODEL BUILDING
The model have been trained using the Scikit-learn library. The training set is composed by 80% of the data, while the remaining 20% is used as test set.
To test the models with different hyperparameters we used the `GridSearchCV` class provided by the Scikit-learn library.

## 5. EVALUATION
To evaluate the model we used the cross validation technique, provided by the Scikit-learn library in `GridSearchCV` class. The evaluation metric used is the Mean Squared Error (MSE). Furthermore, we defined an 'interpretation plot' to translate the results into a more understandable form, associating to each score a possible
combination of helpfulness numerator and helpfulness denominator leading to a score of ~0.8.
## 6. REPORTING
