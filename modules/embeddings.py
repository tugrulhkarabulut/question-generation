import numpy as np

def load_word_embeddings(path):
    word_embeddings = {}

    with open(path) as f:
        for line in f:
            splitted = line.split()
            word = splitted[0]
            embedding = np.array([float(val) for val in splitted[1:]])
            word_embeddings[word] = embedding
    
    return word_embeddings