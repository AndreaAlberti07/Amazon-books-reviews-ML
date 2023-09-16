#!/usr/bin/python3
"""reducer.py"""

import sys

current_key = None
data_values = []
rating_values = []

for line in sys.stdin:
    # Remove leading and trailing whitespace
    line = line.strip()
    # Parse the input from mapper.py
    fields = line.split('\t')

    key = fields[0]
    if fields[1] == '-':
        # Data table record
        data_values = fields[2:]
    else:
        # Rating table record
        rating_values = fields[2:]

    # Check if the key has changed (new record)
    if current_key != key:
        current_key = key
    else:
        # Emit the joined record when both data and rating values are available
        if data_values and rating_values:
            output = current_key + '\t' + \
                '\t'.join(data_values) + '\t' + '\t'.join(rating_values)
            print(output)
