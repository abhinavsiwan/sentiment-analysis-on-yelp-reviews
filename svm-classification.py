import json
import numpy as np
from sklearn import metrics, model_selection
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from utils import create_tags


def classification(train_vecs, train_tags):
    clf = OneVsRestClassifier(SVC(C=1, kernel='linear', gamma=1, verbose=False, probability=False))
    clf.fit(train_vecs, train_tags)
    print("Classifier Trained...")
    predicted = model_selection.cross_val_predict(clf, train_vecs, train_tags, cv=5)
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
        map_star = data[1]

    print("Classification without any processing")
    print("#" * 70)
    with open("processed-data/tf-idf-matrix.json", "r") as f:
        data = json.loads(f.read())
        tf_vector = data[1]

    tags = create_tags(map_sentiment)
    train_vecs = np.array(list(tf_vector.values()))
    train_tags = np.array(list(tags))
    classification(train_vecs, train_tags)
    print("#" * 70)

    print("Classification after removing stop words")
    print("#" * 70)
    with open("processed-data/tf-idf-matrix-stopwords.json", "r") as f:
        data = json.loads(f.read())
        tf_vector = data[1]
    print("TF Matrix Created...")
    train_vecs = np.array(list(tf_vector.values()))
    train_tags = np.array(list(map_sentiment.values()))
    classification(train_vecs, train_tags)
    print("#" * 70)

    print("Classification into 5 Classes")
    print("#" * 70)
    tags = create_tags(map_star)
    train_tags = np.array(list(tags))
    classification(train_vecs, train_tags)


if __name__ == "__main__":
    main()