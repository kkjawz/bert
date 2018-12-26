import pickle
import os

vocab_all = []
vocab_train = []

with open(os.path.expandvars('$DATA/CBTest/data/cbtest_NE_vocab_all.txt')) as f:
    for l in f:
        w = l.strip()
        if w:
            vocab_all.append(w)

with open(os.path.expandvars('$DATA/CBTest/data/cbtest_NE_vocab.txt')) as f:
    for l in f:
        w = l.strip()
        if w:
            vocab_train.append(w)

vocab_train = set(vocab_train)

out_of_vocab = []
for i, w in enumerate(vocab_all):
    if w not in vocab_train:
        out_of_vocab.append(i)

with open(os.path.expandvars('$DATA/CBTest/data/out_of_vocab.pkl'), 'wb') as f:
    pickle.dump(out_of_vocab, f)
