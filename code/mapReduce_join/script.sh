# Specify the Hadoop Streaming JAR file
HADOOP_STREAMING_JAR="$HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-3.3.6.jar"

# Specify the Mapper and Reducer scripts
MAPPER_SCRIPT="join_mapper.py"
REDUCER_SCRIPT="join_reducer.py"

# Specify the input data sources (CSV files to be processed)
INPUT_FILES="hdfs://localhost:9900/user/book_reviews/books_data.csv,hdfs://localhost:9900/user/book_reviews/books_rating.csv"

# Specify the output directory where the job results will be stored
OUTPUT_DIR="hdfs://localhost:9900/user/book_reviews/output"

# Run the Hadoop Streaming job
hadoop jar $HADOOP_STREAMING_JAR \
-files $MAPPER_SCRIPT,$REDUCER_SCRIPT \
-mapper "$MAPPER_SCRIPT" \
-reducer "$REDUCER_SCRIPT" \
-input $INPUT_FILES \
-output $OUTPUT_DIR

