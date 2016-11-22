import json
import numpy as np
from sklearn import metrics, model_selection
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from utils import create_tags, create_tf_idf_matrix, remove_stopwords, tokenize_review


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
    f = open("yelp_academic_dataset_review.json")
    line = f.readline()
    map_sentiment = {}
    map_star = {}
    review_list = []
    i = 1
    while line:
        try:
            line = f.readline()
            review = json.loads(line)
            index = i
            star = review["stars"]
            text = review["text"]
            if star > 3:
                sentiment = '1.0'  # positive
            else:
                sentiment = '0.0'  # negative
            map_sentiment[index] = sentiment
            map_star[index] = star
            review_list.append([index, text])
            i += 1
            if i == 100:
                break
        except:
            print(line)
    f.close()
    print("Dataset Loaded...")
    tokenized_words = tokenize_review(review_list)
    print("Reviews Tokenized...")
    print("Classification without any processing")
    print("#" * 70)
    lexicon, tf_vector = create_tf_idf_matrix(tokenized_words)
    print("TF Matrix Created...")
    print("length of vector : ", len(tf_vector[1]))
    tags = create_tags(map_sentiment)
    train_vecs = np.array(list(tf_vector.values()))
    train_tags = np.array(list(tags))
    classification(train_vecs, train_tags)
    print("#" * 70)
    print("Classification after removing stop words")
    print("#" * 70)
    tokenized_words = remove_stopwords(tokenized_words)
    lexicon, tf_vector = create_tf_idf_matrix(tokenized_words)
    print("TF Matrix Created...")
    print("length of vector : ", len(tf_vector[1]))
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