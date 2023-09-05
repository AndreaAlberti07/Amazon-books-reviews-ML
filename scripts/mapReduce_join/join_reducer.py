#!/usr/bin/python3

import sys

# Create a dictionary for 'Books Data' table
data_table = {}

# Create a dictionary for 'Books Ratings' table
rating_table = {}

# Print header for the output CSV
print('Title,description,authors,publisher,publishedDate,categories,Price,User_id,profileName,review/helpfulness,review/score,review/time,review/summary,review/text\n')

# Loop through lines from standard input (input from a MapReduce job)
for line in sys.stdin:
    # Split the input line into columns
    columns = line.strip().split('\t')

    # Extract the source table identifier
    source = columns[-1]
    values = source.split(',')
    key = columns[0]  # Assuming the book title is the join key

    # Skip the header line
    if key == 'Title':
        continue

    if len(values) == 5:
        # If the source is 'Books Data', store the data in the 'Books Data' dictionary
        book_data = ','.join(values)
        data_table[key] = book_data

        # Check if the book title is in the 'Books Rating' dictionary
        if key in rating_table:
            rows = rating_table[key]
            for rating_data in rows:
                # Join 'Books Data' and 'Books Ratings' data
                joined_data = ','.join(book_data + rating_data)
                print(f"{key},{joined_data}")
    else:
        # If the source is 'Books Ratings', remove the table identifier
        rating_data = ','.join(values)

        # Add rating data to the 'Books Ratings' dictionary, grouped by book title
        if key in rating_table:
            rating_table[key].append(rating_data)
        else:
            rating_table[key] = [rating_data]

        # Check if the book title is in the 'Books Data' dictionary
        if key in data_table:
            data = data_table[key]

            # Join 'Books Data' and 'Books Ratings' data
            joined_data = ','.join(data + rating_data)
            print(f"{key},{joined_data}")
