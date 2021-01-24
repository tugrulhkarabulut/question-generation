import numpy as np
import gensim
from tqdm import tqdm

def load_word_embeddings(path):
    word_embeddings = {}

    with open(path) as f:
        for line in f:
            splitted = line.split()
            word = splitted[0]
            embedding = np.array([float(val) for val in splitted[1:]], dtype=np.float32)
            word_embeddings[word] = embedding
    
    return word_embeddings

def convert_to_word2vec(filename, vocab, dimension):
    vocab_size = len(vocab)
    
    with open(filename, 'wb') as f:
        dims_utf8 = gensim.utils.to_utf8("{} {}\n".format(vocab_size, dimension))
        f.write(dims_utf8)
        
        for word, row in tqdm(vocab.items()):
            row = row.astype(np.float32)
            f.write(gensim.utils.to_utf8(word) + b" " + row.tostring())