# ---------------------------------------------------------------------------------------------
# ---------------------------------[ OPTION 1 ]------------------------------------------------
# ---------------------------------------------------------------------------------------------

import sys

# Initialize variables to keep track of the current book title and data lists
current_title = None
ratings_data = []
info_data = []

# Process input from the mapper
for line in sys.stdin:
    line = line.strip()
    # Split the input line into components: title, data, source, and other_data
    title, data, source, other_data = line.split('\t', 3)

    if current_title == title:
        # If the current book title matches the one in the input line
        if source == 'R':
            # If the source is 'R' (ratings), append the data to the ratings_data list
            ratings_data.append(data + '\t' + other_data)
        elif source == 'I':
            # If the source is 'I' (info), append the data to the info_data list
            info_data.append(data)
    else:
        # If the current book title doesn't match the one in the input line
        if current_title:
            # If there was a previous title, it means we've finished processing its data
            # Join ratings and info data for the same book title and print the result
            for r_data in ratings_data:
                for i_data in info_data:
                    print(f"{current_title}\t{r_data}\t{i_data}")
        # Update the current_title and reset the data lists for the new book title
        current_title = title
        if source == 'R':
            ratings_data = [data + '\t' + other_data]
            info_data = []
        elif source == 'I':
            ratings_data = []
            info_data = [data]

# Output the final joined data
if current_title:
    for r_data in ratings_data:
        for i_data in info_data:
            print(f"{current_title}\t{r_data}\t{i_data}")
            
# ---------------------------------------------------------------------------------------------
# ---------------------------------[ OPTION 2 ]------------------------------------------------
# ---------------------------------------------------------------------------------------------
            
'''Alternative option'''

# Path: code/mapReduce_join/join_reducer.py

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
        if data[0] == rating[0]: # Check if the book titles match
            other_data = '\t'.join(data[1:] + rating[1:])
            print(f"{data[0]}{other_data}")
            
# ---------------------------------------------------------------------------------------------
# ---------------------------------[ OPTION 3 ]------------------------------------------------
# ---------------------------------------------------------------------------------------------
        
'''Even more efficient: exploit the sorting mechanism embedded in the Hadoop Map-Reduce framework'''

import sys

data_table = {}  # Create a dictionary for 'Books Data' table

for line in sys.stdin:
    columns = line.strip().split('\t')
    source = columns[-1]
    key = columns[0]  # Assuming the book title is the join key

    if source == 'D':
        data_table[key] = columns[1:-1]  # Store data in the dictionary
    else:
        rating_data = columns[1:-1]  # Remove the table identifier

        # Check if the book title is in the 'Books Data' dictionary
        if key in data_table:
            data = data_table[key]
            # Join 'Books Data' and 'Books Ratings' data
            joined_data = '\t'.join(data + rating_data)
            print(f"{key}{joined_data}")