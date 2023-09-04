#! /usr/bin/python2
import sys

for line in sys.stdin:
    columns = line.strip().split(',')
    print(len(columns))
    if (len(columns) > 10):
        break
    # Check which table is being read
    if len(columns) == 9:  # 'Books Ratings' table
        title = columns[1]  # Extract the book title
        # Join all other data columns
        other_data = '\t'.join(columns[0]+columns[2:])
        print(f"{title}\t{other_data}\tR")

    else:  # 'Books Data' table
        title = columns[0]  # Extract the book title
        other_data = '\t'.join(columns[1:])  # Join all other data columns
        print(f"{title}\t{other_data}\tD")
