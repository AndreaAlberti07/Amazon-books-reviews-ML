import sys

# Initialize variables to keep track of the current book title and data lists
current_title = None
ratings_data = []
info_data = []

# Process input from the mapper
for line in sys.stdin:
    line = line.strip()
    # Split the input line into components: title, data, source, and other_data
    title, data, source, other_data = line.split('\t', 3)

    if current_title == title:
        # If the current book title matches the one in the input line
        if source == 'R':
            # If the source is 'R' (ratings), append the data to the ratings_data list
            ratings_data.append(data + '\t' + other_data)
        elif source == 'I':
            # If the source is 'I' (info), append the data to the info_data list
            info_data.append(data)
    else:
        # If the current book title doesn't match the one in the input line
        if current_title:
            # If there was a previous title, it means we've finished processing its data
            # Join ratings and info data for the same book title and print the result
            for r_data in ratings_data:
                for i_data in info_data:
                    print(f"{current_title}\t{r_data}\t{i_data}")
        # Update the current_title and reset the data lists for the new book title
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
