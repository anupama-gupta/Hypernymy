## About

Implements a Binary CLassification Task (hypernym relation or not) using linear SVM. 

The following feature vector representation methods are used:

1)  Word2Vec hypnonym vectors 

2)  Modified hypnonym vectors (modifying lexical function function is learnt using regression)

## Usage: 

Method 1)  

    $ python hypernym_classification.py naive_svm file1 file2 file3  

Method 2)
          
    $ python hypernym_classification.py lex_function file1 file2 file3

where :

[file1] (https://github.com/anupama-gupta/Hypernymy/tree/master/dataset) - dictionary(pickle file) of true hypnonyms,  eg: pos_dict['animal'] = ['cat', 'dog', 'goat']

[file2] (https://github.com/anupama-gupta/Hypernymy/tree/master/dataset) - dictionary(pickle file) of false hypnonyms, eg: neg_dict['animal'] = ['grass', 'prey', 'bone', 'ocean' ]

[file3] (https://github.com/anupama-gupta/Hypernymy/tree/master/dataset) - word2vec model 

## Tools

### 1) train_word2vec

Learns word2vec embeddings from the ukwac corpus using gensim [word2vec] (https://radimrehurek.com/gensim/install.html) toolkit.

#### Usage :




