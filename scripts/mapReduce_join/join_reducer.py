#!/usr/bin/python3

import sys

data_table = {}  # Create a dictionary for 'Books Data' table
rating_table = {}  # Create a dictionary for 'Books Ratings' table

print('Title,description,authors,image,previewLink,publisher,publishedDate,infoLink,categories,ratingsCount,Price,User_id,profileName,review/helpfulness,review/score,review/time,review/summary,review/text\n')
for line in sys.stdin:
    columns = line.strip().split('\t')
    source = columns[-1]
    values = source.split(',')
    table = values[-1]
    values = values[:-1]
    key = columns[0]  # Assuming the book title is the join key
    if key == 'Title':
        continue

    if table == 'D':
        book_data = values
        data_table[key] = book_data  # Store data in the dictionary

        # Check if the book title is in the 'Books Rating' dictionary
        if key in rating_table:
            rows = rating_table[key]
            for rating_data in rows:
                # Join 'Books Data' and 'Books Ratings' data
                joined_data = ','.join(data + rating_data)
                print(f"{key},{joined_data}")
    else:
        rating_data = values  # Remove the table identifier
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
