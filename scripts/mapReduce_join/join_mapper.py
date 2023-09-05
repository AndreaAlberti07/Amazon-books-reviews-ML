#!/usr/bin/python3

import sys

for line in sys.stdin:
    columns = line.strip().split(',')
    title = columns[0]  # Extract the book title
    other_data = ','.join(columns[1:])
    # Join all other data columns
    print('%s\t%s' % (title, other_data))
