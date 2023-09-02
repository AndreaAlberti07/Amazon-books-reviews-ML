#!/usr/bin/env python

import sys

current_title = None
ratings_data = []
info_data = []

# Process input from the mapper
for line in sys.stdin:
    line = line.strip()
    title, data, source, other_data = line.split('\t', 3)

    if current_title == title:
        if source == 'R':
            ratings_data.append(data + '\t' + other_data)
        elif source == 'I':
            info_data.append(data)
    else:
        if current_title:
            # Join ratings and info data for the same book title
            for r_data in ratings_data:
                for i_data in info_data:
                    print(f"{current_title}\t{r_data}\t{i_data}")
        current_title = title
        if source == 'R':
            ratings_data = [data + '\t' + other_data]
            info_data = []
        elif source == 'I':
            ratings_data = []
            info_data = [data]

# Output the final joined data
if current_title:
    for r_data in ratings_data:
        for i_data in info_data:
            print(f"{current_title}\t{r_data}\t{i_data}")
