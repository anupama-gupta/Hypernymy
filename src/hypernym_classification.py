
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

# Return the list of vectors corresponding to the list of hypnonym names
# eg, train_words = ["cat", "dog", "goat"], class_name = "animal", features_list = [ [0.04,0.07...], [0.05,..], [0.04,...] ]
def extract_features ( train_words, class_name ) :

	#b = numpy.asarray(model[class_name])
	#b = b / numpy.linalg.norm(b)
	features_list = []
	for t in train_words  :
		a = numpy.asarray(model[t])
		# Normalize the vector
		a = a / numpy.linalg.norm(a)		
 		features_list.append(a)
		#features_list.append((b-a))

	return features_list


'''
Input :  
1. pos_dict : Key - Hypernym , Values - True hyponymns, eg: pos_dict['animal'] = ['cat', 'dog', 'goat']
2. neg_dict : Key - Hypernym , Values - False hyponyms, eg : neg_dict['animal'] = ['grass', 'prey', 'bone', 'ocean' ]

 The following method extracts :
 train_set => 75% of positive instances + 75% of negative instances 
 test_set =>  25% of positive instances + 25% of negative instances
 
 Output : 
 1. train_dataset ( dictionary where :  key - hypernym , value -  [ [train_set_vectors], [train_set_labels], [train_set_names] ] ) 
 2. test_dataset  ( dictionary where :  key - hypernym , value  - [ [test_set_vectors],  [test_set_labels] , [test_set_names] ] )
 
 Eg : train_dataset['animal'] = [ [ [0.04,..], [0.03,...], .. ] , [1,1,1,.,0,0], ['cat', 'dog',...] ] 
 
 '''

def split_train_test ( train=0.75 ) : 

	train_dataset = collections.defaultdict(list)
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
		
#Return the list of word vectors(as numpy arrays) of positive instances in the train_dataset
#This list will be used to train the regression model in 'lex_function_learning'
def extract_postive_features ( dataset, labels ) :
	
	pos_dataset = []
	for i , x in enumerate ( dataset ) :
		if( labels[i] == 1 ) :
			pos_dataset.append(x)
	return pos_dataset
	
#Learns a classifier using the features obtained by applying the function mapping 
def lex_function_classifier_training( class_name, reg_model, hyper_vec) :
	
	#labels = train_dataset[class_name][1]
	#hypo_name = train_dataset[class_name][2]

	mapped_features = []
		
	for i, vec in enumerate(train_dataset[class_name][0]) :

		sub = hyper_vec - vec				
		Y_pred = reg_model.predict(vec.reshape(1, -1))
		mapped_features.append(Y_pred[0])	
				

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
		pls2 = PLSRegression(n_components=50, max_iter=5000)

		X = extract_postive_features ( train_dataset[class_name][0], train_dataset[class_name][1] )			

		Y = []

		for hypo_vec in X :

			sub = hyper_vec-hypo_vec
			Y.append(sub) # Target = difference vector ( Hypernym_vector - Hyponym_vector )
			#Y.append(hyper_vec) # Target = Hypernym vector 

		pls2.fit( X, Y)	
		train_acc = pls2.score(X, Y)
		print "class = ", class_name, "train len = ", len(X)
		
		return pls2, train_acc, len(X)

#Evaluates the classifier model on the test_dataset using the lexically mapped features as input
def lex_function_test( class_name, reg_model, hyper_vec, clf) :
	      
		#labels = test_dataset[class_name][1]		
		#hypo_name = test_dataset[class_name][2]
				
		mapped_features = []

		for vec in test_dataset[class_name][0] :

			sub = hyper_vec - vec
			Y_pred = reg_model.predict(vec.reshape(1, -1))
			mapped_features.append(Y_pred[0])					
		

		y_pred = clf.predict(mapped_features)
		#y_pred_proba = clf.predict_proba(mapped_features)

		test_acc = sklearn.metrics.accuracy_score(test_dataset[class_name][1], y_pred)
		precision = sklearn.metrics.precision_score( test_dataset[class_name][1], y_pred)
		recall = sklearn.metrics.recall_score(test_dataset[class_name][1], y_pred)
		auc = sklearn.metrics.roc_auc_score( test_dataset[class_name][1], y_pred)

		return test_acc, len(test_dataset[class_name][0]), precision, recall, auc
		
#Displays the individual and average evalaution scores for all the classes/hypernyms
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
	
''' Following tasks executed :
1. lexical function (LF) learning using positive hyponym instances in the train_dataset
2. classification model evaluation using the features obtained by using the LF
''' 
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

#Perform hypernym relation classification using word2vec hyponym vectors as features
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
	
	train_dataset, test_dataset = split_train_test (  )

	if( args.function == "naive_svm" ) :
		SVM_classfier_classwise( )

	elif( args.function == "lex_function" ) :
		lex_function_classwise( )
