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
