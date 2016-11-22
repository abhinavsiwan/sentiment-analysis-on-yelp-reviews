import json
from utils import create_tf_idf_matrix, remove_stopwords
from nltk import word_tokenize


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
                # if i == 100:
                #     break
            except:
                print(line)

    training_dataset = [map_sentiment, map_star, tokenized_words]
    with open("processed-data/training-dataset.json", "w") as f:
        f.write(json.dumps(training_dataset))

    lexicon, tf_vector = create_tf_idf_matrix(tokenized_words)
    with open("processed-data/tf-idf-matrix.json", "w") as f:
        f.write(json.dumps([list(lexicon), tf_vector]))

    tokenized_words = remove_stopwords(tokenized_words)
    lexicon, tf_vector = create_tf_idf_matrix(tokenized_words)
    with open("processed-data/tf-idf-matrix-stopwords.json", "w") as f:
        f.write(json.dumps([list(lexicon), tf_vector]))


if __name__ == "__main__":
    main()