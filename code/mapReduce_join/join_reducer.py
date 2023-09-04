#!/usr/bin/env python

import sys

data_table = []
rating_table = []

for line in sys.stdin:
    columns = line.strip().split('\t')

    # Check the source of the data and prepare two separate lists
    source = columns[-1]
    if source == 'D':
        data_table.append(columns[:-1])
    else:
        rating_table.append(columns[:-1])

# Iterate through the two lists and join the data
for data in data_table:
    for rating in rating_table:
        if data[0] == rating[0]:  # Check if the book titles match
            other_data = '\t'.join(data[1:] + rating[1:])
            print(f"{data[0]}{other_data}")

# # ---------------------------------------------------------------------------------------------
# # ---------------------------------[ OPTION 3 ]------------------------------------------------
# # ---------------------------------------------------------------------------------------------

# '''Even more efficient: exploit the sorting mechanism embedded in the Hadoop Map-Reduce framework'''

# import sys

# data_table = {}  # Create a dictionary for 'Books Data' table

# for line in sys.stdin:
#     columns = line.strip().split('\t')
#     source = columns[-1]
#     key = columns[0]  # Assuming the book title is the join key

#     if source == 'D':
#         data_table[key] = columns[1:-1]  # Store data in the dictionary
#     else:
#         rating_data = columns[1:-1]  # Remove the table identifier

#         # Check if the book title is in the 'Books Data' dictionary
#         if key in data_table:
#             data = data_table[key]
#             # Join 'Books Data' and 'Books Ratings' data
#             joined_data = '\t'.join(data + rating_data)
#             print(f"{key}{joined_data}")
