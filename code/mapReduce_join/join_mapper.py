# ---------------------------------------------------------------------------------------------
# ---------------------------------[ OPTION 1 ]------------------------------------------------
# ---------------------------------------------------------------------------------------------

import sys

# This function reads the 'Books Ratings' table and emits ('title', ('rating', 'other_data')) for each record.
for line in sys.stdin:
    line = line.strip()
    columns = line.split(',')

    '''Valid if the Id columns is deleted from the 'Books Ratings' table'''
    # Check if there are 9 columns (Assuming 9 columns in the 'Books Ratings' CSV)
    if len(columns) == 9:
        book_title = columns[0].strip()  # Extract the book title
        rating = columns[5].strip()      # Extract the rating
        # Join other data columns
        other_data = '\t'.join(columns[0:5] + columns[6:])
        # Emit the data in the format 'title    rating    R    other_data'
        print(f"{book_title}\t{rating}\tR\t{other_data}")

# This function reads the 'Books Info' table and emits ('title', ('info', 'other_data')) for each record.
for line in sys.stdin:
    line = line.strip()
    columns = line.split(',')

    # Check if there are 10 columns (Assuming 10 columns in the 'Books Info' CSV)
    if len(columns) == 10:
        book_title = columns[0].strip()  # Extract the book title
        info_data = '\t'.join(columns[1:])  # Join all other info data columns
        # Emit the data in the format 'title    info_data    I    -' where '-' represents no additional data
        print(f"{book_title}{info_data}\tI\t-")
        
# ---------------------------------------------------------------------------------------------
# ---------------------------------[ OPTION 2 ]------------------------------------------------
# ---------------------------------------------------------------------------------------------

'''Alternative option'''

# Path: code/mapReduce_join/join_mapper.py

import sys

for line in sys.stdin:
    columns = line.strip().split(',')
    
    # Check which table is being read
    if len(columns)==9: # 'Books Ratings' table
        title = columns[1] # Extract the book title
        other_data = '\t'.join(columns[0]+columns[2:]) # Join all other data columns
        print(f"{title}\t{other_data}\tR")
    
    else: # 'Books Data' table
        title = columns[0] # Extract the book title
        other_data = '\t'.join(columns[1:]) # Join all other data columns
        print(f"{title}\t{other_data}\tD")
    
