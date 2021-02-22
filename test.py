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
    with open(os.path.join(model_path, "word2idx.pkl"), "rb") as f:
        word2idx = pickle.load(f)

    with open(os.path.join(model_path, "cate2idx.pkl"), "rb") as f:
        cate2idx = pickle.load(f)

    nclass = len(cate2idx)
    model = create_model(ntoken, embedding_dim, nclass)    
    model.load_weights(os.path.join(model_path, "model"))    

    model.summary()