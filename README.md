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

## Repository Structure

This repository is organized into the following folders, each serving a specific purpose:

- **Documents :**
  Contains general documents used for synchronizing activities among team members. You can find meeting notes, project timelines, and any other relevant files here.

- **MapReduce Join :**
  Contains the scripts for MapReduce operations, including mapper and reducer scripts. These scripts are used for data processing tasks within the project.

- **Notebooks :**
  This folder is further organized into subfolders, each dedicated to a specific aspect of the project:

  - **Hypotheses Testing :**
    Contains Jupyter notebooks used for testing and analyzing hypotheses related to the project. You'll find code and documentation for hypothesis testing here.

  - **Model :**
    Contains Jupyter notebooks used for feature extraction, model training, and evaluating the predictive capabilities of our models. This is where the core data analysis and machine learning work happens.

  - **MongoDB :**
    Holds Jupyter notebooks related to exporting a subset of data to MongoDB. This may include data migration and integration tasks.

  - **Spark :**
    It is dedicated to Jupyter notebooks for preliminary data analysis, data cleaning, and testing various hypotheses on the complete dataset using Apache Spark.

- **Report :**
  Contains LaTeX files for creating the project report. This is where you can find the documentation and presentation materials summarizing our project's goals, methodology, findings, and conclusions.

- **Presentation :**
  Contains template and images used in the PowerPoint presentation
## Trained models
The models trained during the project execution are available at the following [Google Drive folder](https://drive.google.com/drive/folders/1kgpb66SaGIKC7ud7nEyKdYnnswlh9VRY?usp=sharing)
## Contact

If you have any questions or need further information about this project, please feel free to contact us

Thank you for your interest in this project!
