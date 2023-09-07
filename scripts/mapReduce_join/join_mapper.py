#!/usr/bin/python3

import sys

for line in sys.stdin:
	# Remove leading and trailing whitespace
	line = line.strip()
	# Get the fields
	fields = line.split(',')
	
	if len(fields) == 6:
	    # Data table
		output = fields[0] + '\t-'*8 + '\t' + '\t'.join(fields[1:])
	    # output = fields[0] + '\t-\t' + fields[1] + '\t-' * 8
	if len(fields) == 9:
    	# Rating table
		output = fields[0] + '\t' + '\t'.join(fields[1:]) + '\t-' * 5
		# output = fields[0] + '\t' + fields[1] + '\t-\t' + '\t'.join(fields[2:])
	
	print(output)
