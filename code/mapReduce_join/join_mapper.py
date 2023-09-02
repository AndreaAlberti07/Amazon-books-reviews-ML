import sys

# This function reads the 'Books Ratings' table and emits ('title', ('rating', 'other_data')) for each record.
for line in sys.stdin:
    line = line.strip()
    columns = line.split(',')

    # Check if there are 8 columns (Assuming 8 columns in the 'Books Ratings' CSV)
    if len(columns) == 8:
        book_title = columns[1].strip()  # Extract the book title
        rating = columns[6].strip()      # Extract the rating
        # Join other data columns
        other_data = '\t'.join(columns[0:6] + columns[7:])
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
        print(f"{book_title}\t{info_data}\tI\t-")
