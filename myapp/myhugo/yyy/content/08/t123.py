from typing import List

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

sentence = "The New York Rangers beat the Pittsburgh Penguins last night"

words = sentence.lower().split()

int2word = {k: v for (k, v) in enumerate(set(words))}
word2int = {v: k for (k, v) in int2word.items()}

vocab_size = len(word2int)

embedding_dim = 3
embedding_layer = np.random.randn(vocab_size, embedding_dim)

tokens = [word2int[w] for w in words]

embeddings = np.asarray([embedding_layer[idx] for idx in tokens])

print(sentence)
print(tokens)
print(embeddings)