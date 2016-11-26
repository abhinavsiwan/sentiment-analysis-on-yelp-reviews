import json
import numpy as np
from sklearn import metrics, model_selection, tree
from sklearn.tree import DecisionTreeClassifier


def train(train_vecs, train_tags):
    clf = DecisionTreeClassifier(max_depth=15, criterion="entropy")  # construct a decision tree
    clf.fit(train_vecs, train_tags)
    print("Classifier Trained...")
    return clf


def classify(clf, train_vecs, train_tags):
    predicted = clf.predict(train_vecs)
    print("Cross Fold Validation Done...")
    print("accuracy score: ", metrics.accuracy_score(train_tags, predicted))
    print("precision score: ", metrics.precision_score(train_tags, predicted, pos_label=None, average='weighted'))
    print("recall score: ", metrics.recall_score(train_tags, predicted, pos_label=None, average='weighted'))
    print("classification_report: \n ", metrics.classification_report(train_tags, predicted))
    print("confusion_matrix:\n ", metrics.confusion_matrix(train_tags, predicted))
    return


def main():
    with open("processed-data/training-dataset.json", "r") as f:
        data = json.loads(f.read())
        map_sentiment = data[0]

    print("Classification without any processing")
    print("#" * 70)
    with open("processed-data/tf-idf-matrix.json", "r") as f:
        tf_vector = json.loads(f.read())

    tags = map_sentiment.values()
    train_vecs = np.array(list(tf_vector.values()))
    train_tags = np.array(list(tags))
    clf = train(train_vecs, train_tags)
    classify(clf, train_vecs, train_tags)
    print("#" * 70)

    # print("Classification after removing stop words")
    # print("#" * 70)
    # with open("processed-data/tf-idf-matrix-stopwords.json", "r") as f:
    #     tf_vector = json.loads(f.read())
    # print("TF Matrix Created...")
    # train_vecs = np.array(list(tf_vector.values()))
    # train_tags = np.array(list(map_sentiment.values()))
    # clf = train(train_vecs, train_tags)
    # classify(clf, train_vecs, train_tags)
    # print("#" * 70)

    # print("Classification into 5 Classes")
    # print("#" * 70)
    # tags = map_star.values()
    # train_tags = np.array(list(tags))
    # clf = train(train_vecs, train_tags)
    # classify(clf, train_vecs, train_tags)

if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    main()
    end = timeit.default_timer()
    print("\n Time taken: " + str(end - start))