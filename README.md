# Amazon Book Reviews Analysis

This repository contains code and resources for analyzing Amazon book reviews. The project aims to develop scalable solutions for various analyses, including sentiment analysis, review helpfulness prediction, topic modeling, and more.

## Team

- **Andrea Alberti** ([GitHub](https://github.com/AndreaAlberti07))
- **Davide Ligari** ([GitHub](https://github.com/DavideLigari01))
- **Cristian Andreoli** ([GitHub](https://github.com/CristianAndreoli94))

## Tools

- **Big Data**: Hadoop, Spark, MongoDB
- **Data Analysis**: Pandas, Scikit-learn, Seaborn, Matplotlib, Jupyter Notebook

## Data

We're using the [Amazon Books Reviews dataset](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews), containing 142.8 million reviews. The dataset comprises two tables: Books Ratings and Books Info.

## Hypotheses

We've formulated several hypotheses, including:

1. Reviews with longer text have higher helpfulness ratings.
2. Reviews with more positive sentiment words receive higher helpfulness ratings.
3. Reviews with higher average book ratings have higher helpfulness ratings.
4. The rating score is influenced by individual users.
5. The review/score is influenced by the category of the book.
6. The number of books published for a category affects the review score.

## Workflow

1. **Data Preparation**:

   - Load data into HDFS.
   - Clean data with Spark.
   - Perform data aggregation and transformations.
   - Load transformed data into MongoDB.

2. **Modeling**:

   - Choose appropriate models (classification, regression, clustering, dimensionality reduction).

3. **Evaluation**:

   - Evaluate models using relevant metrics.

4. **Reporting**:
   - Generate reports and insights.
