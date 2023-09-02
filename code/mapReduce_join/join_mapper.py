#!/usr/bin/env python

import sys

# This function reads the 'Books Ratings' table and emits ('title', ('rating', 'other_data')) for each record.
for line in sys.stdin:
    line = line.strip()
    columns = line.split(',')
    if len(columns) == 8:  # Assuming 8 columns in the 'Books Ratings' CSV
        book_title = columns[1].strip()
        rating = columns[6].strip()
        other_data = '\t'.join(columns[0:6] + columns[7:])
        print(f"{book_title}\t{rating}\tR\t{other_data}")

# This function reads the 'Books Info' table and emits ('title', ('info', 'other_data')) for each record.
for line in sys.stdin:
    line = line.strip()
    columns = line.split(',')
    if len(columns) == 10:  # Assuming 10 columns in the 'Books Info' CSV
        book_title = columns[0].strip()
        info_data = '\t'.join(columns[1:])
        print(f"{book_title}\t{info_data}\tI\t-")
