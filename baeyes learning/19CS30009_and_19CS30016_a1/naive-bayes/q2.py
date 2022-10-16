from csv import reader
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi
import pandas as pd
import matplotlib
import re
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from math import sqrt,exp,pi

def get_X_y(path):
    """
    Loads the X and y from the given path.
    Assumes last columns of the x are the target values.
    :param path: the path to the x
    :return: the x as X and y numpy arrays
    """
    x = pd.read_csv(path)
    X = x.drop(x.columns[-1], axis=1)
    y = x[x.columns[-1]]
    return X, y
def outlier_removal(X_train,y_train):
    df = X_train.copy()
    df['y'] = y_train

    df['filter']=np.zeros(df.shape[0])
    columns = X_train.columns
    for col in columns:
        df['filter'] = df['filter']+[1 if abs((x - np.mean(df[col]))/np.std(df[col]))>3 else 0 for x in df[col]]
    df = df[df['filter']!=np.max(df['filter'])]
    X_train = df.iloc[:,:-2]
    y_train = df.iloc[:,-2]
    for column in X_train.columns:
        X_train[column] = (X_train[column] -X_train[column].mean()) / X_train[column].std()
    return X_train,y_train

def train_test_split(X, y, train_size, shuffle=True, seed=42):
    """
    Splits the x into training and test sets.
    :param X: the x
    :param y: the target values
    :param train_size: the size of the training set
    :param shuffle: whether to shuffle the x
    :param seed: the seed for the random generator
    :return: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = tts(X, y, stratify=y, test_size=1-train_size, random_state=seed)
    return X_train, X_test, y_train, y_test
 


def cross_validation_split(dataset, n_folds):
    """
    Splits the dataset into 10 n folds n-1 for training and 1 for validation.
    :param dataset: the training dataset
    :param n_folds: the number of folds
    :return: dataset_split
    """
    ds_split = list()
    cp = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            id = randrange(len(cp))
            fold.append(cp.pop(id))
        ds_split.append(fold)
    return ds_split
 

def accuracy_metric(actual, predicted):
    """
    Calculates the accuracy of baeyes classifier
    :param actual: the given output parameter
    :param predicted: the predicted hypothesis
    :return: accuracy
    """
    right = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            right += 1
    return right / float(len(actual)) * 100.0

def precision_metric(actual, predicted):
    """
    Calculates the precision of baeyes classifier
    :param actual: the given output parameter
    :param predicted: the predicted hypothesis
    :return: precision
    """
    true_pos=0
    total_pos=0
    for i in range(len(actual)):
        if predicted[i] == 1:
            total_pos+=1
        if predicted[i] == 1 and actual[i] == predicted[i]:
            true_pos+=1
    return true_pos / float(total_pos) * 100.0
def recall_metric(actual,predicted):
    """
    Calculates the recall of baeyes classifier
    :param actual: the given output parameter
    :param predicted: the predicted hypothesis
    :return: recall
    """
    true_pos=0
    false_neg=0
    for i in range(len(actual)):
        if predicted[i] == 0 and actual[i] != predicted[i]:
            false_neg+=1
        if predicted[i] == 1 and actual[i] == predicted[i]:
            true_pos+=1
    return true_pos / float(true_pos+false_neg) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    acc = list()
    pre = list()
    rec = list()
    scores = list()
    final_train_set = list()
    #i=0
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        #print("hello")
        accuracy = accuracy_metric(actual, predicted)
        acc.append(accuracy)
        if accuracy == max(acc):
            final_train_set=train_set
            #print(i)
        #i+=1
        precision = precision_metric(actual, predicted)
        pre.append(precision)
        recall = recall_metric(actual, predicted)
        rec.append(recall)
    
    scores.append(acc)
    scores.append(pre)
    scores.append(rec)
    return scores, final_train_set
 
def separate_by_class(dataset):
	separate = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separate):
			separate[class_value] = list()
		separate[class_value].append(vector)
	return separate
 
# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
 
# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries
 
# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separate = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separate.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries
 
# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
 
# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	all_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(all_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities
 
"""
Calculates the recall of baeyes classifier
:param actual: the given output parameter
:param predicted: the predicted hypothesis
:return: recall
"""
def predict(summaries, row):
	prob = calculate_class_probabilities(summaries, row)
	best_fit_label, best_fit_prob = None, -1
	for class_value, probability in prob.items():
		if best_fit_label is None or probability > best_fit_prob:
			best_fit_prob = probability
			best_fit_label = class_value
	return best_fit_label
 
"""
Naive Bayes algorithm
:param train: training dataset
:param test: the test datset
:return: predictions
"""
def naive_bayes(train, test):
	summary = summarize_by_class(train)
	predictions = list()
	for row in test:
		output = predict(summary, row)
		predictions.append(output)
	return(predictions)
 

seed(1012)
DATA_PATH = "./Dataset_E.csv"
X_full, y_full = get_X_y(DATA_PATH)
print("************** DATASET DESCRIPTION BEFORE REMOVAL OF OUTLIERS **************")
print((pd.concat([X_full, y_full], axis=1)))
print("*****************************************************************************")
X_full, y_full = outlier_removal(X_full, y_full)
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, 0.8)
print("\n************** DATASET DESCRIPTION AFTER REMOVAL OF OUTLIERS & NORMALIZATION **************")
print((pd.concat([X_full, y_full], axis=1)))
print("*********************************************************************************************")
(pd.concat([X_full, y_full], axis=1)).to_csv('normalized_dataset.csv')
# print(X_full)
# print(y_full)
# print(X_train)
# print(X_test)
dataset= (pd.concat([X_train, y_train], axis=1)).values.tolist()
# print(dataset)
n_folds = 10
scores, final_train_set = evaluate_algorithm(dataset, naive_bayes, n_folds)
print("\n****************** RESULTS OF 10-FOLD CROSS VALIDATION *****************")
print('Scores: ' )
print('Accuracy: %s' % scores[0])
print('Precision: %s' % scores[1])
print('Recall: %s' % scores[2])
print('Mean Accuracy: %.3f%%' % (sum(scores[0])/float(len(scores[0]))))
print('Mean Precision: %.3f%%' % (sum(scores[1])/float(len(scores[1]))))
print('Mean Recall: %.3f%%' % (sum(scores[2])/float(len(scores[2]))))
print("**********************************************************************")


#calculate testing accuracy using full dataset
pred1=naive_bayes((pd.concat([X_train, y_train], axis=1)).values.tolist(), (pd.concat([X_test, y_test], axis=1)).values.tolist())
pred2=naive_bayes(final_train_set, (pd.concat([X_test, y_test], axis=1)).values.tolist())
accuracy=accuracy_metric(pred2,y_test.values.tolist())
print('Final Test Accuracy : %.3f%%' %accuracy)
precision=precision_metric(pred2,y_test.values.tolist())
print('Final Test Precision : %.3f%%' %precision)
recall=recall_metric(pred2,y_test.values.tolist())
print('Final Test Recall : %.3f%%' %recall)


