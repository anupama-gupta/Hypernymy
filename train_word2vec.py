import gensim
import logging
import os
import string
import nltk.data
import re
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

class MySentences(object):
     def __init__(self, dirname):
         self.dirname = dirname
 
     def __iter__(self):
         for fname in os.listdir( self.dirname):
             
             fp = open(os.path.join(self.dirname, fname))          
	     flag = 0

	     for line in fp:

		line = line.lower()
		
        	if(  "<s>" in line ) :
			sent = ""
            		flag = 1
            		continue

		if(  "</s>" in line ) :
	    		flag = 0
			yield sent.split()	    		 
	    		continue

        	if( flag == 1 ) :
			wordlist = re.sub("[^\w]", " ",  line).split()	
			if( not len(wordlist) == 3 ) :
				continue
			sent += (wordlist[2] + " ")   

#python train_unigrams.py  /home/anupama/tensor/corpus/ukwac/tagged/          

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()	
	
	parser.add_argument("corpus", help="Corpus filepath" )

	args = parser.parse_args()	
                 
	#sentences = MySentences('/home/anupama/tensor/corpus/ukwac/tagged/')
	#model.save('/home/anupama/tensor/models/word2vec/word2vec_ukwac_unigrams20_size300.pkl')

	sentences = MySentences(args.corpus) 
	model = gensim.models.Word2Vec(sentences, min_count = 20, workers = 8, size=300)	
	model.save(os.getcwd()+'/word2vec_ukwac_unigrams20_size300.pkl')
	


