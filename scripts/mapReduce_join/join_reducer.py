#!/usr/bin/python3

import sys

# Create a dictionary for 'Books Data' table
data_table = {}

# Create a dictionary for 'Books Ratings' table
rating_table = {}

# Print header for the output CSV
print('Title,description,authors,publisher,publishedDate,categories,Price,User_id,profileName,review/helpfulness,review/score,review/time,review/summary,review/text\n')
with open('/home/davide/github/data-science-project/tmp_reducer_columns.txt', 'a') as var_file:
    # Loop through lines from standard input (input from a MapReduce job)
    for line in sys.stdin:
        # Split the input line into columns
        var_file.write(f"line\n")
        var_file.write(f"{line}\n")
        title, source = line.split('\t', 1)
        var_file.write(f"n_columns\n")
        var_file.write(f"{len(columns)}\n")
        var_file.write(f"columns\n")
        var_file.write(f"{columns}\n")
        # Extract the source table identifier
        source = columns[-1]
        var_file.write(f"Source\n")
        var_file.write(f"{source}\n")
        n_columns = len(source.split(','))
        var_file.write(f"n_columns\n")
        var_file.write(f"{n_columns}\n")
        title = columns[0]  # Assuming the book title is the join title
        var_file.write(f"title\n")
        var_file.write(f"{title}\n")
        # Skip the header line
        if title == 'Title':
            continue

        if n_columns == 5:
            # If the source is 'Books Data', store the data in the 'Books Data' dictionary
            book_data = source
            data_table[title] = book_data

            # Check if the book title is in the 'Books Rating' dictionary
            if title in rating_table:
                rows = rating_table[title]
                for rating_data in rows:
                    # Join 'Books Data' and 'Books Ratings' data
                    joined_data = ','.join(book_data + rating_data)
                    print('%s,%s' % (title, joined_data))
        else:
            # If the source is 'Books Ratings', remove the table identifier
            rating_data = source

            # Add rating data to the 'Books Ratings' dictionary, grouped by book title
            if title in rating_table:
                rating_table[title].append(rating_data)
            else:
                rating_table[title] = [rating_data]

            # Check if the book title is in the 'Books Data' dictionary
            if title in data_table:
                data = data_table[title]

                # Join 'Books Data' and 'Books Ratings' data
                joined_data = ','.join(data + rating_data)
                print('%s,%s' % (title, joined_data))
        break
