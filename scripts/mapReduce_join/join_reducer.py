#!/usr/bin/python3
"""reducer.py"""

import sys

title = None
values = None

for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # parse the input from mapper.py
    fields = line.split('\t')

    if fields[1] == '-':
        # Save the title and genre of a movie
        title = fields[0]
        values = fields[10:]
    else:
        if fields[0] == title:
            # Title, description, authors, publisher, publishedDate, categories, Price, User_id, profileName,
            # review/score, review/time, review/summary, review/text, N_helpful, Tot_votes
            output = title + '\t' + \
                '\t'.join(values) + '\t'+'\t'.join(fields[2:11])
            print(output)
