# Code to perform binary classification (Linear SVM) of hyponym features classwise (40 classes) using two methods :
# 1. State-of-the-art -> Using Normalized Word2Vec hypnonym vectors as Features
# 2. Our method -> Using mapped hypnonym vectors as Features ( by learning a lexical function using positive hypnonym instances )

import pickle
import collections
import gensim
import sklearn.linear_model
import numpy
import operator
import argparse
from scipy import linalg, mat, dot
from sklearn.metrics.pairwise import euclidean_distances
from composes.utils.regression_learner import LstsqRegressionLearner
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import r2_score

# Return the list of features corresponding to the list of hypnonym names
def extract_features ( train_words, class_name ) :

	features_list = []
	for t in train_words  :
		a = numpy.asarray(model[t])	

		# Normalize the vector
		a = a / numpy.linalg.norm(a)		
 		features_list.append(a.tolist())

		#b = numpy.asarray(model[class_name])
		#b = b / numpy.linalg.norm(b)
		#features_list.append((b-a).tolist())

	return features_list


# Input : Two sets of dictionaries : 
#1. Key - Hypernym , Values - True hyponymns
#2. Key - Hypernym , Values - False hyponyms

#Output : 3 sets dictionaries :
# train set = First 70% ( by default ) of positive instances + First 70% of negative instances 
# test set = First 25% ( by default ) of positive instances + First 25% of negative instances

def split_train_val_test ( train=0.75 ) : 

	train_dataset = collections.defaultdict(list)
	val_dataset = collections.defaultdict(list)
	test_dataset = collections.defaultdict(list)
	
	for class_name, hypo_list in pos_dict.items() :
		
		index = int(train* len(pos_dict[class_name]))
		pos_train_words = pos_dict[class_name] [0:index+1]
		pos_train_features = extract_features ( pos_train_words, class_name )		
		pos_test_words = pos_dict[class_name] [index+1:]
		pos_test_features = extract_features ( pos_test_words, class_name)
		
		index = int(train* len(neg_dict[class_name]))
		neg_train_words = neg_dict[class_name] [0:index+1]
		neg_train_features = extract_features ( neg_train_words, class_name )
		neg_test_words = neg_dict[class_name] [index+1:]
		neg_test_features = extract_features ( neg_test_words, class_name )
 
		train_dataset[class_name] = [ pos_train_features + neg_train_features,  [1]*len(pos_train_words) + [0]*len(neg_train_words), 				                                        pos_train_words  + neg_train_words ]

		test_dataset[class_name] = [ pos_test_features + neg_test_features,  [1]*len(pos_test_words) + [0]*len(neg_test_words), 				                                        pos_test_words  + neg_test_words ]


	return train_dataset, test_dataset

		

def extract_postive_data ( dataset, labels ) :
	
	pos_dataset = []
	for i , x in enumerate ( dataset ) :
		if( labels[i] == 1 ) :
			pos_dataset.append(numpy.asarray(x))
	return pos_dataset


def extract_numpy_features ( dataset ) :
	
	numpy_dataset = []
	for i , x in enumerate ( dataset ) :
		numpy_dataset.append(numpy.asarray(x))
	return numpy_dataset
	

def lex_function_classifier_training( class_name, reg_model, hyper_vec) :
	
	hypo_vectors = extract_numpy_features (train_dataset[class_name][0])
	labels = train_dataset[class_name][1]
	hypo_name = train_dataset[class_name][2]

	mapped_features = []
		
	for i, vec in enumerate(hypo_vectors) :

		sub = numpy.asarray(hyper_vec) - vec
				
		Y_pred = reg_model.predict(vec.reshape(1, -1))
		mapped_features.append(Y_pred[0].tolist())	
				

	clf = sklearn.svm.SVC(kernel="linear", probability=True)
	#clf = sklearn.svm.SVC(kernel="rbf", probability=True)
	#clf = sklearn.svm.SVC()
	#print len(mapped_features[0])
	
	clf.fit(mapped_features, train_dataset[class_name][1])
	train_acc = sklearn.metrics.accuracy_score( train_dataset[class_name][1], clf.predict(mapped_features))

	return clf, train_acc

# Learn lex function F such that : F(Hyponym_vector) = Hypernym_vector - Hyponym_vector
def lex_function_learning( class_name,  hyper_vec ) :

		#pls2 = KernelRidge( kernel = "rbf", gamma= 100)
		#pls2 = KernelRidge( )
		pls2 = PLSRegression(n_components=50, max_iter=10000)

		X = extract_postive_data ( train_dataset[class_name][0], train_dataset[class_name][1] )			

		Y = []

		for hypo_vec in X :
			sub = hyper_vec-hypo_vec
			Y.append(sub) # Target = difference vector ( Hypernym_vector - Hyponym_vector )
			#Y.append(hyper_vec) # Target = Hypernym vector 

		pls2.fit( X, Y)	
		train_acc = pls2.score(X, Y)
		print "class = ", class_name, "train len = ", len(X)
		
		return pls2, train_acc, len(X)


def lex_function_test( class_name, reg_model, hyper_vec, clf) :
	      
		hypo_vectors = extract_numpy_features ( test_dataset[class_name][0] )
		labels = test_dataset[class_name][1]		
		hypo_name = test_dataset[class_name][2]
				
		mapped_features = []

		for vec in hypo_vectors :

			sub = numpy.asarray(hyper_vec) - vec
			Y_pred = reg_model.predict(vec.reshape(1, -1))
			mapped_features.append(Y_pred[0].tolist())					
		

		y_pred = clf.predict(mapped_features)
		#y_pred_proba = clf.predict_proba(mapped_features).tolist()

		test_acc = sklearn.metrics.accuracy_score( test_dataset[class_name][1], y_pred)
		precision = sklearn.metrics.precision_score( test_dataset[class_name][1], y_pred)
		recall = sklearn.metrics.recall_score(test_dataset[class_name][1], y_pred)
		auc = sklearn.metrics.roc_auc_score( test_dataset[class_name][1], y_pred)

		return test_acc, len(hypo_vectors), precision, recall, auc

#usage : python hypernym_classification_2.py lex_function  /home/anupama/tensor/data/mydata/pos_train_40_classes_7531_pairs_allsenses.p #/home/anupama/tensor/data/mydata/neg_train_40_classes_22593_pairs_allsenses.p /home/anupama/tensor/models/word2vec/word2vec_ukwac_unigrams20_size300.pkl

def print_results( test_acc, train_acc, test_len , test_p, test_r, test_f, test_auc ):

	avg_test_p = 0
	avg_test_r = 0
 	avg_test_f = 0
	avg_test_auc = 0
	avg_test_acc = 0

	for key, value in sorted(test_acc.items(), key=operator.itemgetter(1), reverse=True )  :
		print key , " --> ", value, "(", test_len[key], ")", "train acc = ", train_acc[key], " p ", test_p[key], "r = ", test_r[key], "f = ", test_f[key]," auc = ", test_auc[key]

		avg_test_p += test_p[key]
		avg_test_r += test_r[key]
 		avg_test_f += test_f[key]
		avg_test_auc += test_auc[key]
		avg_test_acc += value

	print "avg acc = ", ((float)(avg_test_acc))/len(test_acc)
	print "avg precision = ", avg_test_p/len(test_acc)
	print "avg recall = ", avg_test_r/len(test_acc)
	print "avg fscore = ", avg_test_f/len(test_acc)
	print "avg auc = ", avg_test_auc/len(test_acc)
	

def lex_function_classwise ( ) :

	train_acc = {}
	test_acc = {}
	train_len = {}
	test_len = {}
	test_precision = {}
	test_recall = {}
	test_fscore = {}
	test_auc = {}	

	for class_name in train_dataset.keys():
		
		hyper_vec = numpy.asarray(model[class_name])
		# Normalise the hyernym vector 
		hyper_vec = hyper_vec / numpy.linalg.norm(hyper_vec)
		
		# Lexical Function Learning
		reg_model, acc, freq = lex_function_learning( class_name, hyper_vec )
	
		# Classifier Training   
		clf, acc =  lex_function_classifier_training( class_name, reg_model, hyper_vec )

		# Testing 
		testacc, tlen, precision, recall, auc = lex_function_test( class_name, reg_model, hyper_vec, clf) 
		
		train_acc[class_name] = acc
		test_acc[class_name] = testacc
		train_len[class_name] = len
		test_len[class_name] = tlen
		test_precision[class_name] = precision
		test_recall[class_name] = recall
		test_auc[class_name] = auc		
		test_fscore[class_name] = ( 2 * precision * recall ) / ( precision + recall )

	
	print_results( test_acc, train_acc, test_len , test_precision, test_recall, test_fscore, test_auc )	


def cosine_similarity(a, b) :
	
	c = dot(a,b.T)/linalg.norm(a)/linalg.norm(b)
	return c

def SVM_classfier_classwise (  ) :
    	
	test_p = {}
	test_r = {}
	test_f = {}
	test_auc = {}
	test_acc = {}
	test_len = {}
	train_acc = {}	

	for k in train_dataset.keys():

	 	#clf = sklearn.linear_model.SGDClassifier()
		clf = sklearn.svm.SVC(kernel="linear")
		print k , " train len = ", len(train_dataset[k][1]), " test len = ", len(test_dataset[k][1])
		clf.fit(train_dataset[k][0], train_dataset[k][1])
		 
    		trainacc = sklearn.metrics.accuracy_score(clf.predict(train_dataset[k][0]), train_dataset[k][1])		

		y_pred = clf.predict(test_dataset[k][0])
		#y_pred_proba = clf.predict_proba(test_dataset[k][0])

		acc = sklearn.metrics.accuracy_score(test_dataset[k][1], y_pred)
		test_p[k] = sklearn.metrics.precision_score(test_dataset[k][1], y_pred)
		test_r[k] = sklearn.metrics.recall_score(test_dataset[k][1], y_pred)
		test_auc[k] = sklearn.metrics.roc_auc_score( test_dataset[k][1], y_pred)
		test_f[k] = sklearn.metrics.f1_score( test_dataset[k][1], y_pred)
		test_len[k] = len(test_dataset[k][1])
		train_acc[k] = trainacc
		test_acc[k] = acc

	print_results( test_acc, train_acc, test_len , test_p, test_r, test_f, test_auc )
		


if __name__ == "__main__": 

	
	parser = argparse.ArgumentParser()
	parser.add_argument("function", help="Functions : 1) naive_svm 2) lex_function")
	parser.add_argument("pos_dict", help="Positive instances dictionary")
	parser.add_argument("neg_dict", help="Negative instances dictionary")
	parser.add_argument("model", help="Word vectors model")
	
	args = parser.parse_args()

	pos_dict = pickle.load( open(args.pos_dict, 'rb') )
	neg_dict = pickle.load( open(args.neg_dict, 'rb') )
	model = gensim.models.Word2Vec.load(args.model)
	
	train_dataset, test_dataset = split_train_val_test (  )

	if( args.function == "naive_svm" ) :
		SVM_classfier_classwise ( )

	elif( args.function == "lex_function" ) :
		lex_function_classwise ( )
	

	
