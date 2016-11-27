import json
from utils import create_tf_idf_matrix, build_count_dict


def main(type):
    file = "raw-data/train.json" if type == "train" else "raw-data/dev.json"
    with open(file, 'r') as f:
        data = json.loads(f.read())
        map_sentiment = {}
        tokens_count = {}
        i = 0
        for record in data:
            try:
                sentiment = record["sentiment"]
                tokens = record["tokens"]
                map_sentiment[i] = sentiment
                tokens_count[i] = build_count_dict(tokens)
                i += 1
                # if i == 1000:
                #     break
            except:
                print(record)

    file = "processed-data/training-dataset.json" if type == "train" else "processed-data/dev-dataset.json"
    with open(file, "w") as f:
        f.write(json.dumps(map_sentiment))

    lexicon, tf_vector = create_tf_idf_matrix(tokens_count)
    file = "processed-data/tf-idf-matrix.json" if type == "train" else "processed-data/dev_tf-idf-matrix.json"
    with open(file, "w") as f:
        f.write(json.dumps(tf_vector))


if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    main("train")
    end = timeit.default_timer()
    print("\n Time taken: " + str(end - start))