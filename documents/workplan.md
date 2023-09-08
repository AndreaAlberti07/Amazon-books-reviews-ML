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

Goal of the project is to develop a scalable solution to analyze a dataset of reviews of books from Amazon. Possible analysis are:

- Sentiment Analysis
- Review Helpfulness Prediction
- Topic Modeling
- Recommendation System
- Common Issues Analysis
- Category Prediction

### 1.4. Data Collection

The chosen dataset is [Amazon Books Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews).
142.8 million reviews spanning May 1996 - July 2014. The dataset is composed by 2 tables:

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

   - **Model**: Linear Regression.

   - **Description**:
     - Add a column for each review's length.
     - Use the review length as the predictor variable and the helpfulness rating as the target variable. (i.e. activate feature only above a given threshold)
     - Train a linear regression model to predict helpfulness ratings based on review length.
     - The correlation coefficient can also be calculated as a post-processing step.

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
     - Compute the mean helpfulness ratings for the most useful words.

- **Missing Values**:
  - `review/score`: remove the entire sample
  - `review/text`: remove the entire sample
  - `review/helpfulness`: remove the entire sample

- **Data Transformation**:
  - `review/score`: Assign 1 to score (4, 5), 0 to score (1, 2). 
  - `review/text`: Create the BoW for the text. Fit a MNBC and count the number of positive and negative words. Graphical Plot.
  - `review/helpfulness`: $helpfulness = \frac{x}{y} \sqrt(y)$

   ***

3. **_Hypothesis_**: Reviews with higher average book ratings have higher helpfulness ratings.

   - **Metric**: Correlation between average book ratings and helpfulness ratings.

   - **Model**: Linear Regression

   - **Description**:

     - Use the average book rating as the predictor variable and the helpfulness rating as the target variable.
     - Train a linear regression model to predict helpfulness ratings based on average book ratings.

- **Missing Values**:

  - `review/score`: remove the entire sample
  - `review/helpfulness`: remove the entire sample

- **Data Transformation**:
  - `review/score`: groupBy book title and calculate the average score.
  - `review/helpfulness`: $helpfulness = \frac{x}{y} \sqrt(y)$

   ***

4. **_Hypothesis_**: Reviews with more specific and descriptive summaries are perceived as more helpful.

   - **Metric**: Compare the mean helpfulness ratings of reviews with detailed summaries against those with vague summaries.

   - **Model**: NLP methods.

   - **Description**:

     - Convert the review summaries into numerical features using techniques like TF-IDF or word embeddings.
     - Train a classification model to classify reviews as having detailed or vague summaries.
     - Compare the mean helpfulness ratings for reviews predicted as detailed vs. vague.

- **Missing Values**:

  - `review/summary`: set missing values as empty string
  - `review/helpfulness`: remove the entire sample

- **Data Transformation**:
  - `review/summary`: ...
  - `review/helpfulness`: $helpfulness = \frac{x}{y} \sqrt(y)$

   ***

5. **_Hypothesis_**: Reviews written by users with a history of providing helpful reviews receive higher helpfulness ratings.

   - **Metric**: Compare the mean helpfulness ratings for reviews by users with a history of high helpfulness ratings against those without such a history.

   - **Model**: User-based Features and Classification.

   - **Description**:

     - Aggregate user-level statistics (e.g., average helpfulness ratings of their previous reviews).
     - Train a classification model to predict whether a review will be helpful based on user-related features.
     - Compare the mean helpfulness ratings for reviews predicted as helpful by users with a history of high helpfulness vs. others.

- **Missing Values**:

  - `profileName`: Make a separate group for 'Anonymous' users.
  - `review/helpfulness`: remove the entire sample

- **Data Transformation**:
  - `profileName`: Transform the helpfulness. GroupBy profileName and average the helpfulness of the reviews.
  - `review/helpfulness`: $helpfulness = \frac{x}{y} \sqrt(y)$

   ***

6. **_Hypothesis_**: There is a relationship between the number of reviews of a specific user and the helpfulness of his reviews.

   - **Metric**: Correlation between the number of reviews of a specific user and the helpfulness of his reviews.
   - **Model**: Linear Regression.

   - **Description**:

     - Add a column for each user's number of reviews.
     - Use the number of reviews as the predictor variable and the helpfulness rating as the target variable.
     - Train a linear regression model to predict helpfulness ratings based on the number of reviews.

- **Missing Values**:

  - `profileName`: Make a separate group for 'Anonymous' users.
  - `review/helpfulness`: remove the entire sample

- **Data Transformation**:
  - `profileName`: GroupBy profileName and count the reviews.
  - `review/helpfulness`: $helpfulness = \frac{x}{y} \sqrt(y)$

7. **_Hypothesis_**: There are some publisher getting higher average review/score.

   - **Metric**: Compare the mean review/score of the books published by the same publisher.

   - **Model**: Linear Regression.

   - **Description**:
   
     - Add a column for each publisher's average review/score.
     - Use the average review/score as the predictor variable and the helpfulness rating as the target variable.
     - Train a linear regression model to predict helpfulness ratings based on the average review/score.

- **Missing Values**:
  
    - `publisher`: set missing values as empty string
    - `review/score`: remove the entire sample

- **Data Transformation**:
    - `publisher`: GroupBy publisher.
    - `review/score`: Compute the average review/score for each publisher.

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

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("DuplicateRemovalAndMissingDataHandling") \
    .getOrCreate()

# Load data from CSV files into DataFrames
ratings_df = spark.read.csv("path_to_ratings.csv", header=True, inferSchema=True)
info_df = spark.read.csv("path_to_books_info.csv", header=True, inferSchema=True)

# Drop duplicates from ratings DataFrame based on ID
ratings_df = ratings_df.dropDuplicates(subset=["Id"])

# Fill missing values in ratings DataFrame with default values
ratings_df = ratings_df.fillna({"Price": 0.0, "review/score": 0.0})

# Drop duplicates from info DataFrame based on Title
info_df = info_df.dropDuplicates(subset=["Title"])

# Fill missing values in info DataFrame with default values
info_df = info_df.fillna({"authors": "Unknown", "categories": "Unknown"})

# Perform the join operation on the Title column
joined_df = ratings_df.join(info_df, on="Title", how="inner")

# Select desired columns from the joined DataFrame
selected_columns = [
    "Title", "Price", "User_id", "profileName",
    "review/helpfulness", "review/score",
    "authors", "categories"
]
result_df = joined_df.select(selected_columns)

# Show the resulting DataFrame
result_df.show()

# Stop the Spark session
spark.stop()
```

### 2.3. Data Aggregation

- Program a MapReduce job to join the data by book title

```python
# Mapper function for Books Ratings table
def map_ratings(line):
    fields = line.split("\t")  # Assuming fields are tab-separated
    book_title = fields[1]
    # Emit (book_title, ("ratings", fields)) as key-value pair
    emit(book_title, ("ratings", fields))

# Mapper function for Books Info table
def map_info(line):
    fields = line.split("\t")  # Assuming fields are tab-separated
    book_title = fields[0]
    # Emit (book_title, ("info", fields)) as key-value pair
    emit(book_title, ("info", fields))

# Reducer function
def reduce_join(key, values):
    ratings = []
    info = []

    # Separate values into ratings and info lists based on their source
    for value_type, fields in values:
        if value_type == "ratings":
            ratings.append(fields)
        elif value_type == "info":
            info.append(fields)

    # Perform the join operation
    for rating_fields in ratings:
        for info_fields in info:
            # Emit joined information as key-value pair
            emit(None, (rating_fields + info_fields))

# Input: Read lines from both tables
for line in BooksRatingsTable:
    map_ratings(line)

for line in BooksInfoTable:
    map_info(line)

# Sort and shuffle step
sort_and_shuffle()

# Process sorted and shuffled data
for key, values in shuffled_data:
    reduce_join(key, values)
```

- Run the MapReduce job on Hadoop

```bash
# Run a Hadoop Streaming job using the Hadoop Streaming JAR file

# Specify the files to be distributed to Hadoop nodes (Mapper and Reducer scripts)
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
-files join_mapper.py,join_reducer.py \

# Specify the Mapper script to be used for the job (join_mapper.py)
-mapper join_mapper.py \

# Specify the Reducer script to be used for the job (join_reducer.py)
-reducer join_reducer.py \

# Specify the input data sources (CSV files to be processed)
-input /path/to/books_ratings.csv,/path/to/books_info.csv \

# Specify the output directory where the job results will be stored
-output /path/to/output_directory
```

### 2.4. Data Transformation

### 2.5. Load into MongoDB

```bash
pip install pymongo[tls] pyspark
```

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql import DataFrameWriter

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("LoadDataToMongoDB") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/mydb.collection") \
    .getOrCreate()

# Define schema for the transformed data
transformed_schema = StructType([
    StructField("Title", StringType(), True),
    StructField("Price", FloatType(), True),
    StructField("User_id", IntegerType(), True),
    # ... define other transformed fields
])

# Load, filter, and transform data
ratings_df = spark.read.csv("path_to_ratings.csv", header=True, inferSchema=True)

filtered_ratings_df = ratings_df.filter(col("review/score") >= 4)  # Example filter

transformed_df = filtered_ratings_df.select("Title", "Price", "User_id")

# Write the transformed data to MongoDB
transformed_df.write \
    .format("mongo") \
    .mode("overwrite") \
    .save()

# Stop the Spark session
spark.stop()
```

### 2.6. Complex MongoDB query

### 2.7. Feature Extraction

### 2.8. Local analysis with Python Libraries

## 3. MODEL CHOICE

### 3.1. Classification

### 3.1.2. Regression

### 3.1.3. Clustering

### 3.1.4. Dimensionality Reduction

## 4. MODEL BUILDING

## 5. EVALUATION

### 5.1. Accuracy

### 5.1.2. PR

### 5.1.3. ROC

### 5.1.4. MSE

## 6. REPORTING
