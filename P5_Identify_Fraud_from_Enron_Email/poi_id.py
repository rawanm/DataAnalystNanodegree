#!/usr/bin/python

import sys
import pickle
import copy
import math
from collections import defaultdict
sys.path.append("../tools/")


### local imports: 
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### sklearn imports: 
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# helper methods: 
def add_features(data_dict, features_list):
	features_list_updated = list(features_list)
	for name in data_dict:
	    data_point = data_dict[name]

	    from_poi_to_this_person = data_point["from_poi_to_this_person"]
	    to_messages = data_point["to_messages"]
	    fraction_from_poi = compute_fraction( from_poi_to_this_person, to_messages )
	    data_point["fraction_from_poi"] = fraction_from_poi

	    from_this_person_to_poi = data_point["from_this_person_to_poi"]
	    from_messages = data_point["from_messages"]
	    fraction_to_poi = compute_fraction( from_this_person_to_poi, from_messages )
	    data_point["fraction_to_poi"] = fraction_to_poi
	features_list_updated.append('fraction_from_poi')
	features_list_updated.append('fraction_to_poi')
	return data_dict, features_list_updated    

def get_missing_values_count (data_dict):
	missing_features_count = defaultdict(int)
	data = data_dict.values()
	for data_point in data: 
		for feature in data_point:
			if data_point[feature] == 'NaN':
				missing_features_count[feature] += 1
	missing_features_count = sorted(missing_features_count.items(), key=lambda(k,v): v, reverse=True)
	print missing_features_count

def compute_fraction(poi_messages, all_messages ):
    """ 
    given a number messages to/from POI (numerator) 
    and number of all messages to/from a person (denominator),
    return the fraction of messages to/from that person
    that are from/to a POI
   """
    poi_messages = float(poi_messages)
    all_messages = float(all_messages)
    ### compute: 
    ### the fraction of all messages to this person that come from POIs
    ### the fraction of all messages from this person that are sent to POIs
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    if not (math.isnan(poi_messages) &  math.isnan(all_messages)):
        fraction = poi_messages/all_messages
    return fraction

def get_test_data(features, labels, feature_list, folds = 10):
	cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
	features_train = []
	features_test  = []
	labels_train   = []
	labels_test    = []
	for train_idx, test_idx in cv: 
		for ii in train_idx:
			features_train.append( features[ii] )
			labels_train.append( labels[ii] )
		for jj in test_idx:
			features_test.append( features[jj] )
			labels_test.append( labels[jj] )    
	return features_train, features_test, labels_train, labels_test

def custom_grid_search (pipeline, params, features_list, features_train, labels_train, scoring_metric, cv):
	grid_searcher = GridSearchCV(pipeline, param_grid=params, cv=cv,
	                           n_jobs=-1, scoring = scoring_metric, verbose=0)

	grid_searcher.fit(features_train, labels_train)

	mask = grid_searcher.best_estimator_.named_steps['select'].get_support()
	top_features = [x for (x, boolean) in zip(features_list, mask) if boolean]
	n_pca_components = grid_searcher.best_estimator_.named_steps['reduce'].n_components_

	print "Cross-validated {0} score: {1}".format(scoring_metric, grid_searcher.best_score_)
	print "{0} features selected".format(len(top_features))
	print "Top Features: "
	print top_features
	print "Reduced to {0} PCA components".format(n_pca_components)
	print "Params: ", grid_searcher.best_params_ 

	clf = grid_searcher.best_estimator_
	print "Best Estimator:"
	print clf
	return clf

def get_dataset_overview(labels, features):
	poi = 0
	for label in labels:
		if(label):
	 		poi += 1

	print "Features count:"
	print len(features)
	
	print "Total dataset count:"
	print len(labels)
	
	print "POI count:"
	print poi

	print "Non-POI count:"
	print len(labels) - poi
	
def test_algorithms(algorithm, features_list ,features_train, labels_train, scoring_metric):

	# scoring_metric:  f1, recall, precision
	scoring_metric = scoring_metric
	
	### Pipeline and GridSearch:
	generic_estimator = estimators = [
	    ('scale', MinMaxScaler()),
	    ('select', SelectKBest(k = 10)),
	    ('reduce', PCA(n_components=0.75))
	]

	generic_params = {'select__k':[5, 7, 9, 13, 15, 17, 20, 'all'],
				  'reduce__n_components': [1, 2, 3, 4, 5, .25, .5, .75, .9, 'mle'], 
	              'reduce__whiten': [True]
	              }

	estimators =  list(generic_estimator)
	params = generic_params.copy()

	if algorithm == 'SVC':
		# SVC: 
		estimators.append(('classify', SVC(class_weight='balanced')))
		
		# SVC: 		
		SVC_custom_params = {'classify__C': [1,10.,100,1e3, 1e4, 1e5],
		              'classify__gamma': [0.0],
		              'classify__kernel': ['rbf'],
		              'classify__tol': [1e-3],
		              'classify__class_weight': ['balanced'],
		              'classify__random_state': [13, 20, 42],
		              }
		params.update(SVC_custom_params)

	elif algorithm == 'LinearSVC': 
		# SVC Linear: 
		estimators.append(('classify', LinearSVC()))
		
		# SVC Linear: 
		LinearSVC_custom_params = {'classify__C': [1,10.,100,1e3, 1e4, 1e5],
		              'classify__tol': [1e-3],
		              'classify__class_weight': ['balanced'],
		              'classify__random_state': [13, 20, 42],
		              }
		params.update(LinearSVC_custom_params)

	elif algorithm == 'DT': 
		# Decision Tree: 
		estimators.append(('classify', DecisionTreeClassifier()))
		
		# Decision Tree: 
		DT_custom_params = {'classify__min_samples_split': [10, 15, 30, 40],
					  'classify__criterion' : ['gini', 'entropy'],
		              'classify__splitter': ['best', 'random'],
		              'classify__class_weight': ['balanced'],
		              'classify__random_state': [13, 20, 42],
		              }
		params.update(DT_custom_params)

	elif algorithm == 'RF':  
		# Random Forest: 
		estimators.append(('classify', RandomForestClassifier()))

		# Random Forest: 
		RandomForest_custom_params = {'classify__min_samples_split': [10, 15, 30, 40],
					  'classify__n_estimators': [5, 7, 10, 20], 
					  'classify__criterion' : ['gini', 'entropy'],
		              'classify__class_weight': ['balanced'],
		              'classify__random_state': [13, 20, 42],
		              }
		params.update(RandomForest_custom_params)

	print "Pipeline:"
	print "GridSearch: " + algorithm
	folds = 1000
	cv = StratifiedShuffleSplit(labels, folds, random_state=random)
	pclf = Pipeline(estimators)	
	clf = custom_grid_search(pclf, params, features_list, features_train, labels_train, scoring_metric, cv)
	test_classifier(clf.steps[-1][1], data_dict_updated, features_list_updated)

	if algorithm in ['DT','RF']:
		features_importance = clf.steps[-1][1].feature_importances_
		max_importance = 0
		max_index = 0
		for index, importance in enumerate(features_importance):
			if importance > 0.2:
				if importance > max_importance:
					max_importance = importance
					max_index = index
		print "Highest importance: "
		print max_importance, " feature: " + features_list_updated[index+1]
	
	# return the classifier to be used in further steps: 
	return clf.steps[-1][1] #clf

### static variables
random = 13

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 
'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

# removeing 'Total' and Non peson record: 
data_dict.pop("TOTAL", 0 ) 
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

### Task 3: Create new feature(s)
data_dict_updated, features_list_updated = add_features(data_dict, features_list)


features_x_org = list(features_list)
features_x_org.remove('poi')

features_x = list(features_list_updated)
features_x.remove('poi')

### Store to my_dataset for easy export below.
# uncomment this line in order to accuratly test orignal features list for comparison:
my_dataset_org = data_dict
my_dataset = data_dict_updated

### Extract features and labels from dataset for local testing
data_org = featureFormat(my_dataset_org, features_list)
data = featureFormat(my_dataset, features_list_updated)

labels_org, features_org = targetFeatureSplit(data_org)
labels, features = targetFeatureSplit(data)

# get overview of the dataset: 

#get_dataset_overview(labels_org,features_list)
get_dataset_overview(labels,features_list_updated)
get_missing_values_count(my_dataset)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!

# uncomment this section to run classifier on new features and orignal features: 
'''
features_train_org, features_test_org, labels_train_org, labels_test_org = \
    train_test_split(features_org, labels_org, test_size=0.3, random_state=42)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print "Test without new features: "
test_algorithms('LinearSVC', features_x_org, features_train_org, labels_train_org, 'f1') 

print "Test with new features: "
test_algorithms('LinearSVC', features_x, features_train, labels_train, 'f1') 
'''
###################################################################

features_train, features_test, labels_train, labels_test = get_test_data(features, labels, features_list_updated)

# uncomment this section to run classifiers selections and tuning: 
test_algorithms('SVC', features_x, features_train, labels_train, 'f1') 
'''
#test_algorithms('SVC', features_x, features_train, labels_train, 'f1') 
#test_algorithms('LinearSVC', features_x, features_train, labels_train, 'f1') 
#test_algorithms('RF', features_x, features_train, labels_train, 'f1') 
#test_algorithms('DT', features_x, features_train, labels_train, 'f1') 
'''
###################################################################

clf = test_algorithms('RF', features_x, features_train, labels_train, 'f1') 

print "Test Final Classifier: "
clf.fit(features_train, labels_train)
ppred = clf.predict(features_test)

accuracy = accuracy_score(labels_test, ppred, normalize=True, sample_weight=None)
print "Accuracy:"
print accuracy

recall = recall_score(labels_test, ppred)
print "Recall:"
print recall

precision = precision_score(labels_test, ppred)
print "Precision:"
print precision

f1 = f1_score(labels_test, ppred)
print "F1:"
print f1

test_classifier(clf, data_dict_updated, features_list_updated)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)