from nltk import word_tokenize
from nltk.corpus import stopwords


def remove_stopwords(tokenized_words):
    for i in range(len(tokenized_words)):
        filtered_words = [word for word in tokenized_words[i] if word not in stopwords.words('english')]
        tokenized_words[i] = filtered_words
    return tokenized_words


def tokenize_review(review_list):
    tokenized_words = {}
    for review in review_list:
        tokenized_words[review[0]] = word_tokenize(review[1])
    return tokenized_words


def build_lexicon(tokenized_word):
    lexicon = set()
    for i in range(len(tokenized_word)):
        lexicon.update(tokenized_word[i])
    return lexicon


def tf(word, tokenized_words):
    return tokenized_words.count(word)


def create_tf_idf_matrix(tokenized_words):
    lexicon = build_lexicon(tokenized_words)
    tf_vector = {}
    for i in range(len(tokenized_words)):
        tf_vector[i] = [tf(word, tokenized_words[i]) for word in lexicon]
    return lexicon, tf_vector


def create_tags(map_sentiment):
    return map_sentiment.values()
