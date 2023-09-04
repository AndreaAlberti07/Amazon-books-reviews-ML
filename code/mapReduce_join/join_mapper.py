#!/usr/bin/env python

import sys

for line in sys.stdin:
    columns = line.strip().split(',')
    # Check which table is being read
    if len(columns) == 10:  # 'Books Ratings' table
        title = columns[0]  # Extract the book title
        other_data = ','.join(columns[1:])  # Join all other data columns
        with open('out.csv', 'a') as out:
            out.write(f"{title}\t{other_data}\tD\n")

    else:  # 'Books Rating' table
        title = columns[0]  # Extract the book title
        # Join all other data columns
        other_data = ','.join(columns[1:])
        with open('out.csv', 'a') as out:
            out.write(f"{title}\t{other_data}\tR\n")
