from text_classfier.dataset import load_dataset
from text_classfier.preprocess import preprocess_for_train
from text_classfier.model import create_model
import json
import os
import pickle

with open("ClassifierConfig.json") as f:
    config = json.load(f)

data_dir      = config["DATA_DIR"]
ntoken        = config["N_TOKENS"]
maxlen        = config["MAXLEN"]
embedding_dim = config["EMBEDDING_DIM"] 
n_epochs      = config["N_EPOCHS"]
batch_size    = config["BATCH_SIZE"]
model_path    = config["MODEL_PATH"]

if __name__ == "__main__":
    dataset = load_dataset(data_dir)

    data     = dataset["data"]
    labels   = dataset["label"]
    cate2idx = dataset["Category2Index"]

    nclass = len(cate2idx)

    preprocessed = preprocess_for_train(data, labels, ntoken, maxlen, embedding_dim)

    x_train = preprocessed["X"]
    y_train = preprocessed["y"]
    word2idx= preprocessed["Word2Idx"]
    
    model = create_model(ntoken, embedding_dim, nclass) 
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
   
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_split=0.2)

    #save model
    model.save_weights(os.path.join(model_path, "model"))
    with open(os.path.join(model_path, "word2idx.pkl"), "wb") as f:
        pickle.dump(word2idx, f)

    with open(os.path.join(model_path, "cate2idx.pkl"), "wb") as f:
        pickle.dump(cate2idx, f)
        
