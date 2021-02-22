from text_classfier.dataset import load_dataset
from text_classfier.preprocess import preprocess_for_test
from text_classfier.model import create_model
import json
import os
import pickle
import numpy as np

with open("ClassifierConfig.json") as f:
    config = json.load(f)

data_dir      = config["DATA_DIR"]
test_dir      = config["TEST_DATA_DIR"]
ntoken        = config["N_TOKENS"]
maxlen        = config["MAXLEN"]
embedding_dim = config["EMBEDDING_DIM"] 
n_epochs      = config["N_EPOCHS"]
batch_size    = config["BATCH_SIZE"]
model_path    = config["MODEL_PATH"]

if __name__ == "__main__":
    with open(os.path.join(model_path, "word2idx.pkl"), "rb") as f:
        word2idx = pickle.load(f)

    with open(os.path.join(model_path, "cate2idx.pkl"), "rb") as f:
        cate2idx = pickle.load(f)

    nclass = len(cate2idx)
    model = create_model(ntoken, embedding_dim, nclass)    
    model.load_weights(os.path.join(model_path, "model"))    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    dataset = load_dataset(test_dir)

    data     = dataset["data"]
    labels   = dataset["label"]

    x_train = preprocess_for_test(data, word2idx, maxlen)
    y_train = np.array(labels).reshape((-1, 1))

    output = model.evaluate(x_train, y_train, verbose=0)
    print("*" * 100)
    print("Loss:     ", output[0])
    print("Accuracy: ", output[1])
    print("*" * 100)








