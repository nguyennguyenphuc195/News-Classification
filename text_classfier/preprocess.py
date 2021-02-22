from collections import Counter
import random
import numpy as np

def preprocess_for_train(dataset, labels, n_features, maxlen, embedding_dim):
    all_tokens = [token for data in dataset for token in data]
    word_counts = Counter(all_tokens)

    word2idx = {"<pad>" : 0, "<start>" : 1, "<end>" : 2, "<unk>" : 3}
    n_features = n_features - 4

    idx = 4
    for word, count in word_counts.most_common(n_features):
        word2idx[word] = idx
        idx += 1

    data_indexed = []

    #Turn word to index
    for data in dataset:
        text = [1]
        for token in data:
            if len(text) == maxlen - 1:
                break
            text.append(word2idx.get(token, 3))
        
        text.append(2)
        while len(text) < maxlen:
            text.append(0)
        data_indexed.append(text)

    #shuffle data
    x_train = np.array(data_indexed)
    y_train = np.array(labels).reshape((-1, 1))

    random_ids = random.sample(range(x_train.shape[0]), x_train.shape[0])

    x_train = x_train[random_ids]
    y_train = y_train[random_ids]

    return {
        "X" : x_train,
        "y" : y_train,
        "Word2Idx" : word2idx,
    }

def preprocess_for_test(dataset, word2idx, maxlen):
    data_indexed = []
    for data in dataset:
        text = [1]
        for token in data:
            if len(text) == maxlen - 1:
                break
            text.append(word2idx.get(token, 3))
        
        text.append(2)
        while len(text) < maxlen:
            text.append(0)
        data_indexed.append(text)
    return np.array(data_indexed)