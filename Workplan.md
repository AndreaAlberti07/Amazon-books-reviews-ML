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
It contains 2.4 million reviews of books from Amazon. The dataset is composed by 2 tables:

### - Books Ratings table

| Column              | Description                           |
|---------------------|---------------------------------------|
| **Id**                  | The ID of the book                    |
| **Title**               | The title of the book                 |
| **Price**               | The price of the book                 |
| **User_id**             | The ID of the user who rated the book |
| **profileName**         | The name of the user who rated the book|
| **review/helpfulness** | Helpfulness rating of the review (e.g. 2/3)|
| **review/score**        | Rating from 0 to 5 for the book       |
| **review/time**         | Time the review was given             |
| **review/summary**     | The summary of the text review        |
| **review/text**         | The full text of the review           |

### - Books Info table

| Column           | Description                                     |
|------------------|-------------------------------------------------|
| **Title**        | Book title                                      |
| **description**  | Description of the book                        |
| **authors**      | Name of book authors                           |
| **image**        | URL for book cover                             |
| **previewLink**  | Link to access this book on Google Books       |
| **publisher**    | Name of the publisher                          |
| **publishedDate**| The date of publish                            |
| **infoLink**     | Link to get more information about the book on Google Books |
| **categories**   | Genres of books                                |
| **ratingsCount** | Number of ratings and reviews for the book     |


### 1.5. Hypotheses Formulation

## 2. DATA PREPARATION
### 2.0. Data Aggregation
### 2.1. Data Cleaning
### 2.2. Data Transformation
### 2.3. Feature Extraction
### 2.4. Metric Definition


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

