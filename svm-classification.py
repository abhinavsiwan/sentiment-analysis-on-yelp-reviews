import json
import numpy as np
from sklearn import metrics, model_selection
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from utils import create_tags, create_tf_idf_matrix, remove_stopwords, tokenize_review
from nltk import word_tokenize


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
    with open("raw-data/training.json") as f:
        line = f.readline()
        map_sentiment = {}
        map_star = {}
        tokenized_words = {}
        i = 0
        while line:
            try:
                review = json.loads(line)
                line = f.readline()
                star = review["stars"]
                text = review["text"]
                if star > 3:
                    sentiment = '1.0'  # positive
                else:
                    sentiment = '0.0'  # negative
                map_sentiment[i] = sentiment
                map_star[i] = star
                tokenized_words[i] = word_tokenize(text)
                i += 1
            except:
                print(line)
    training_dataset = [map_sentiment, map_star, tokenized_words]
    with open("processed-data/training-dataset.json", "w") as f:
        f.write(json.dumps(training_dataset))
    # print("Classification without any processing")
    # print("#" * 70)
    lexicon, tf_vector = create_tf_idf_matrix(tokenized_words)
    with open("processed-data/tf-idf-matrix.json", "w") as f:
        f.write(json.dumps([list(lexicon), tf_vector]))
    # print("TF Matrix Created...")
    # print("length of vector : ", len(tf_vector[1]))
    # tags = create_tags(map_sentiment)
    # train_vecs = np.array(list(tf_vector.values()))
    # train_tags = np.array(list(tags))
    # classification(train_vecs, train_tags)
    # print("#" * 70)


    # print("Classification after removing stop words")
    # print("#" * 70)
    tokenized_words = remove_stopwords(tokenized_words)
    lexicon, tf_vector = create_tf_idf_matrix(tokenized_words)
    with open("processed-data/tf-idf-matrix-stopwords.json", "w") as f:
        f.write(json.dumps([list(lexicon), tf_vector]))
    # print("TF Matrix Created...")
    # print("length of vector : ", len(tf_vector[1]))
    # train_vecs = np.array(list(tf_vector.values()))
    # train_tags = np.array(list(map_sentiment.values()))
    # classification(train_vecs, train_tags)
    # print("#" * 70)

    # print("Classification into 5 Classes")
    # print("#" * 70)
    # tags = create_tags(map_star)
    # train_tags = np.array(list(tags))
    # classification(train_vecs, train_tags)


if __name__ == "__main__":
    main()