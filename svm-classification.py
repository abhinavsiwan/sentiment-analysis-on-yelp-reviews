import json
import numpy as np
from sklearn import metrics, svm
from sklearn.multiclass import OneVsRestClassifier
import pickle
file_prefix = "processed-data/1000/"


def train(train_vecs, train_tags):
    clf = OneVsRestClassifier(svm.SVC(C=1, kernel='linear', gamma=1, verbose=False, probability=False))
    clf.fit(train_vecs, train_tags)
    with open("intermediate_dt_trained_dumps.bin", 'wb') as fs:
        fs.write(pickle.dumps(clf))
    print("Classifier Trained...")
    return clf


def classify(clf, vecs, tags):
    predicted = clf.predict(vecs)
    print("accuracy score: ", metrics.accuracy_score(tags, predicted))
    print("precision score: ", metrics.precision_score(tags, predicted, pos_label=None, average='weighted'))
    print("recall score: ", metrics.recall_score(tags, predicted, pos_label=None, average='weighted'))
    print("classification_report: \n ", metrics.classification_report(tags, predicted))
    # print("confusion_matrix:\n ", metrics.confusion_matrix(train_tags, predicted))
    return


def main():
    with open(file_prefix + "training-dataset.json", "r") as f:
        map_sentiment_train = json.loads(f.read())

    print("Training.....")
    print("#" * 70)
    with open(file_prefix + "tf-idf-matrix.json", "r") as f:
        tf_vector_train = json.loads(f.read())

    tags = map_sentiment_train.values()
    train_vecs = np.array(list(tf_vector_train.values()))
    train_tags = np.array(list(tags))
    clf = train(train_vecs, train_tags)
    print("#" * 70)
    print("Training completed\n\n")

    with open(file_prefix + "dev-dataset.json", "r") as f:
        map_sentiment_dev = json.loads(f.read())
    print("Classifying.....")
    print("#" * 70)
    with open(file_prefix + "dev-tf-idf-matrix.json", "r") as f:
        tf_vector_dev = json.loads(f.read())
    tags = map_sentiment_dev.values()
    dev_vecs = np.array(list(tf_vector_dev.values()))
    dev_tags = np.array(list(tags))
    print(len(dev_vecs))
    print(len(dev_tags))
    classify(clf, dev_vecs, dev_tags)

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


if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    main()
    end = timeit.default_timer()
    print("\n Time taken: " + str(end - start))



# import json
# import numpy as np
# from sklearn import metrics, model_selection
# from sklearn.svm import SVC
# from sklearn.multiclass import OneVsRestClassifier
#
#
# def classification(train_vecs, train_tags):
#     clf = OneVsRestClassifier(SVC(C=1, kernel='linear', gamma=1, verbose=False, probability=False))
#     clf.fit(train_vecs, train_tags)
#     print("Classifier Trained...")
#     predicted = model_selection.cross_val_predict(clf, train_vecs, train_tags, cv=5)
#     print("Cross Fold Validation Done...")
#     print("accuracy score: ", metrics.accuracy_score(train_tags, predicted))
#     print("precision score: ", metrics.precision_score(train_tags, predicted, pos_label=None, average='weighted'))
#     print("recall score: ", metrics.recall_score(train_tags, predicted, pos_label=None, average='weighted'))
#     print("classification_report: \n ", metrics.classification_report(train_tags, predicted))
#     print("confusion_matrix:\n ", metrics.confusion_matrix(train_tags, predicted))
#     return
#
#
# def main():
#     with open("processed-data/training-dataset.json", "r") as f:
#         data = json.loads(f.read())
#         map_sentiment = data[0]
#         map_star = data[1]
#
#     print("Classification without any processing")
#     print("#" * 70)
#     with open("processed-data/tf-idf-matrix.json", "r") as f:
#         tf_vector = json.loads(f.read())
#
#     tags = map_sentiment.values()
#     train_vecs = np.array(list(tf_vector.values()))
#     train_tags = np.array(list(tags))
#     classification(train_vecs, train_tags)
#     print("#" * 70)
#
#     # print("Classification after removing stop words")
#     # print("#" * 70)
#     # with open("processed-data/tf-idf-matrix-stopwords.json", "r") as f:
#     #     tf_vector = json.loads(f.read())
#     #
#     # print("TF Matrix Created...")
#     # train_vecs = np.array(list(tf_vector.values()))
#     # train_tags = np.array(list(map_sentiment.values()))
#     # classification(train_vecs, train_tags)
#     # print("#" * 70)
#
#
# if __name__ == "__main__":
#     main()