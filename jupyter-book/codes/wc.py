def linecount(file_name):
    count = 0
    with open(file_name) as f:
        for line in f:
            count += 1
    return count
