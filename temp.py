with open("yelp_academic_dataset_review.json") as f:
    line = f.readline()
    i = 1
    train = open("raw-data/training.json", "w")
    while i < 2148054 and line:
        i = i + 1
        train.write(line)
        line = f.readline()
    print("train " + str(i))
    train.close()
    i = 1
    test = open("raw-data/test.json", "w")
    while line:
        i = i + 1
        test.write(line)
        line = f.readline()
    print("test " + str(i))
    test.close()