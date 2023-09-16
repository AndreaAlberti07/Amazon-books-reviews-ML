#!/usr/bin/python3

import sys

for line in sys.stdin:
    # Remove leading and trailing whitespace
    line = line.strip()
    # Get the fields
    fields = line.split('\t')

    if len(fields) == 6:
        # Data table
        output = fields[0] + '\t-\t' + '\t'.join(fields[1:])
    elif len(fields) == 10:
        # Rating table
        output = fields[0] + '\t' + 'www' + '\t' + '\t'.join(fields[1:])
    else:
        continue

    print(output)
