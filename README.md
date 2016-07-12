## Contents

### 1) train_word2vec
Learns word embeddings from the ukwac corpus using word2vec toolkit (https://radimrehurek.com/gensim/install.html)

### 2) hypernym_classification
Implements binary classification ( hypernym relation or not ) using two methods of feature representation :
1. word2vec hyponym vectors
2. modified hyponym vectors (obtained by applying lexical function F on the word2vec vectors), learnt using partial least Squares (PLS) regression


