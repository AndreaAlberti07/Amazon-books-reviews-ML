#!/usr/bin/python3

import sys

data_table = {}  # Create a dictionary for 'Books Data' table
rating_table = {}  # Create a dictionary for 'Books Ratings' table

print('Title,description,authors,image,previewLink,publisher,publishedDate,infoLink,categories,ratingsCount,Price,User_id,profileName,review/helpfulness,review/score,review/time,review/summary,review/text\n')
for line in sys.stdin:
    columns = line.strip().split('\t')
    source = columns[-1]
    key = columns[0]  # Assuming the book title is the join key
    if key == 'Title':
        continue

    if source == 'D':
        book_data = columns[1:-1]
        data_table[key] = book_data  # Store data in the dictionary

        if key in rating_table:
            rating_data = rating_table[key]
            # Join 'Books Data' and 'Books Ratings' data
            joined_data = ','.join(book_data + rating_data)
            print(f"{key},{joined_data}")
    else:
        rating_data = columns[1:-1]  # Remove the table identifier
        rating_table[key] = rating_data  # Store data in the dictionary
        # Check if the book title is in the 'Books Data' dictionary
        if key in data_table:
            data = data_table[key]
            # Join 'Books Data' and 'Books Ratings' data
            joined_data = ','.join(data + rating_data)
            print(f"{key},{joined_data}")
