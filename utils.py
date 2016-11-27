import nltk
import re
import pickle


def build_lexicon(tokenized_word):
    lexicon = set()
    for i in range(len(tokenized_word)):
        lexicon.update(tokenized_word[i].keys())
    return lexicon


def create_tf_idf_matrix(tokenized_words, ftype):
    if ftype == "train":
        lexicon = build_lexicon(tokenized_words)
        with open("intermediate/lexicon.bin", 'wb') as fs:
            fs.write(pickle.dumps(lexicon))
    else:
        with open("intermediate/lexicon.bin", 'rb') as fs:
            lexicon = pickle.loads(fs.read())
    tf_vector = {}
    for i in range(len(tokenized_words)):
        tf_vector[i] = [tokenized_words[i][word] if word in tokenized_words[i] else 0 for word in lexicon]
    return lexicon, tf_vector


def tokenize(text):
    required_tags = {'NNP': 1, 'NN': 1, 'NNS': 1, 'NNPS': 1, 'JJ': 1, 'JJR': 1, 'JJS': 1, 'VBZ': 1}
    delimiters = ";|,|\*|\n|\.|\?|\)|\("
    pos_tags = nltk.pos_tag(text.split())
    ret_list = []
    for pos_tag in pos_tags:
        if pos_tag[1] in required_tags:
            split = re.split(delimiters, pos_tag[0])
            if len(split) > 1:
                for s in split:
                    if len(s) > 0:
                        ret_list.append(s)
            else:
                ret_list.append(pos_tag[0])
    return ret_list


def build_count_dict(tokens):
    map_text_count = {}
    for token in tokens:
        if token in map_text_count:
            map_text_count[token] += 1
        else:
            map_text_count[token] = 1
    return map_text_count
