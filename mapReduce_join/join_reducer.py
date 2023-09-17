#!/usr/bin/python3
"""reducer.py"""

import sys

# Initialize variables to keep track of the current key, data values, and rating values.
current_key = None
data_values = []
rating_values = []

for line in sys.stdin:
    line = line.strip()

    fields = line.split('\t')
    key = fields[0]

    # Check if the record belongs to the data table (fields[1] == '-').
    if fields[1] == '-':
        data_values = fields[2:]
    else:
        # Rating table record
        rating_values = fields[2:]

    # Check if the key has changed (new record).
    if current_key != key:
        current_key = key
    else:
        # Emit the joined record when both data and rating values are available.
        if data_values and rating_values:
            # Combine the current key, data values, and rating values with tab separators.
            output = current_key + '\t' + \
                '\t'.join(data_values) + '\t' + '\t'.join(rating_values)

            # Print the joined record as the output.
            print(output)
