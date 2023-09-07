#!/usr/bin/python3
"""reducer.py"""

import sys

title = None
values = None

for line in sys.stdin:
	# remove leading and trailing whitespace
	line = line.strip()
	# parse the input from mapper.py
	fields = [el.strip() for el in line.split('\t')]
 
	if fields[1] == '-':
		# Save the title and genre of a movie
		title = fields[0]
		values = fields[9:]
	else:
		if fields[0] == title:
            values = values[2:]
            # Title, description, authors, publisher, publishedDate, categories, Price, User_id, profileName, review/helpfulness, review/score, review/time, review/summary, review/text
            output = title + '\t' + '\t'.join(values) + '\t'+'\t'.join(fields[1:9])
			print(output)
