import os
import random
    
def load_dataset(dir, shuffle=True):
    dataset = []
    labels  = []
    categories = sorted(os.listdir(dir))
    for i, cate in enumerate(categories):
        for fn in os.listdir(os.path.join(dir, cate)):
            path = os.path.join(dir, cate, fn)
            data = []
            with open(path) as f:
                for line in f:
                    data.append(line.replace("\n", ""))
            dataset.append(data)
            labels.append(i)

    cate2idx = dict([(cate, idx) for idx, cate in enumerate(categories)])

    return {
        "data" : dataset,
        "label": labels,
        "Category2Index" : cate2idx,
    }