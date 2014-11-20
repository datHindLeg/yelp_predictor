### Extract features from csv data, namely review_text. 
### Then, select important features
### lastly, perform regression

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import string

import os
import nltk
import re, math, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import stopwords

from sklearn import datasets, linear_model
from sklearn import ensemble
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation

from bokeh.plotting import *

# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# DATA MANIPULATION AND PRINTOUT 
# -----------------------------------------------------------------------------------------------------
# *****************************************************************************************************

def main():

    # Load data set (includes both training and validation)
    df_training = pd.read_csv("/home/datascience/FINAL_PROJECT/yelp_predictor/dataset_yelp.csv",sep='|')

    # Load data set (includes test set)
    df_test = pd.read_csv("/home/datascience/FINAL_PROJECT/yelp_predictor/dataset_unknown.csv",sep='|')

    # Load rejects (data we failed to get because yelp had no health scores)
    df_rejects = pd.read_csv("/home/datascience/FINAL_PROJECT/yelp_predictor/dataset_rejects.csv",sep='|')

    data_retained, total_reviews, total_data_pts, df_train_name, df_train_text,df_train_period_rating, df_train_score, df_validation_name, df_validation_text, df_validation_period_rating, df_validation_score = load_data(df_training,df_test, df_rejects)
    print_general_metrics(total_reviews,total_data_pts,data_retained)

    #regression to determine health score based on period yelp review rating
    # coefficient that is printed if the amount health score is expected to change based on a 1 unit change increase in period_rating
    values,residuals,rank,singular_values,rcond = np.polyfit(df_training['period_rating'], df_training['health_score'],1, full=True)
    print "For every 1 point increase in in yelp rating, health score changes by " + str(values[0]) + " points."
    print "Sum of squared residuals (large = bad) : " + str(residuals[0])
    #plot_basic_regr(values[0], values[1])

    # heavy lifting in these 2 lines
    model_train_count, model_validation_count, feature_names = get_features(df_train_text, df_train_score,df_validation_text,df_validation_score)
    values = ridge_regression(model_train_count,df_train_score,model_validation_count,df_validation_score, feature_names)

    # The value results for validation set that we got
    print '\nname', 'score\n'
    for name, value in zip(df_validation_name, values):
        print name + ' => ' + str(int(math.floor(value))) + '\n'

# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------------------------------
# *****************************************************************************************************

def load_data(train, test, rejects):
    len_rejects = 0
    try:
        len_rejects = len(rejects.index)
    except:
        pass

    data_retained = len(train.index) / float( len_rejects + len(train.index) )

    # Count number of yelp reviews we got total
    df_number_reviews = train['number_reviews']
    df_test_number_reviews = test['number_reviews']
    total_reviews = df_number_reviews.sum(axis=1) + df_test_number_reviews.sum(axis=1)

    # Count number of data points (rows we got)
    total_data_pts = len(train.index) + len(test.index)

    # Split the data into 80% training, 20% validation (includes random shuffling)
    # TODO: PERFORM K FOLDS
    rows_train = np.random.choice(train.index.values, int(math.floor(len(train.index)*.8)), replace=False)
    df_train = train.ix[rows_train]
    train = train.drop(rows_train)
    rows_validation = np.random.choice(train.index.values, len(train.index), replace=False)
    df_validation = train.ix[rows_validation]

    # Select data (review text) and name of restaurant (name) and target (health score)
    df_train_name = df_train['name']
    df_train_text = df_train['review_text']
    df_train_period_rating = df_train['period_rating']
    df_train_score = df_train['health_score']

    df_validation_name = df_validation['name']
    df_validation_text = df_validation['review_text']
    df_validation_period_rating = df_validation['period_rating']
    df_validation_score = df_validation['health_score']

    return data_retained, total_reviews, total_data_pts, df_train_name, df_train_text, df_train_period_rating, df_train_score, df_validation_name, df_validation_text, df_validation_period_rating, df_validation_score

# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# FEATURE EXTRACTION & SELECTION
# -----------------------------------------------------------------------------------------------------
# *****************************************************************************************************

def get_features(train_text,train_targets,validation_text,validation_targets):

    # TODO: NLP feature selection

    stop = stopwords.words('english')
    stop.remove('no')
    stop.remove('not')

    review_corpus_train = ""
    review_texts_train = train_text.tolist()
    review_corpus_train = ' '.join(review_texts_train)
    f = open("/home/datascience/text_pre_nlp.txt","w")
    f.write(review_corpus_train)
    f.close()

    feature_df = pd.read_csv('/home/datascience/FINAL_PROJECT/yelp_predictor/out_malt.conll', sep='\t')
    feature_df.columns = ['index','word','blank','basic_indi','extra_indi','blank2','numba','POS','type','blank3']
    df_pair = feature_df[['word','basic_indi']]
    df_pair2 = df_pair[(df_pair['basic_indi'] == 'JJ') | (df_pair['basic_indi'] == 'JJR') | (df_pair['basic_indi'] == 'JJS')]
    df_adj = df_pair2['word'].str.lower().drop_duplicates()

    # this is actually a set, as duplicates were removed at series level
    # unique set of adjectives from feature set
    adjs = df_adj.tolist()

    # filter out non-adjectives from text
    #train_text.map( lambda text: list(text.split(" ") & adjs) )
    
    train_text2 = train_text.str.lower()
    validation_text2 = validation_text.str.lower()

    #train_text3 = train_text2.map(lambda text: " ".join([x for x in text.split(" ") if x in adjs]))
    #validation_text3 = validation_text2.map(lambda text: " ".join([x for x in text.split(" ") if x in adjs]))

    train_text3 = train_text2.map(lambda text: " ".join([x for x in text.split(" ") if unicode(x,'utf-8') not in stop]))
    validation_text3 = validation_text2.map(lambda text: " ".join([x for x in text.split(" ") if unicode(x,'utf-8') not in stop]))

    vectorizer = CountVectorizer(min_df=1)

    # Pure bi-gram (only 2 word featuers)
    #vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b', min_df=1)

    # Bi-gram + single (all 2 word and single combinations)
    #vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)

    X_train_counts = vectorizer.fit_transform(train_text3)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # validation set
    X_validation_counts = vectorizer.transform(validation_text3)
    X_validation_tfidf = tfidf_transformer.transform(X_validation_counts)

    # TODO:tree-based feature selection
    #clf = ExtraTreesClassifier()
    #X_train_tree = clf.fit_transform(train_text, train_targets)
    #X_validation_tree = clf.transform(validation_text,validation_targets)

    
    """
    # print tfidf scores for each feature
    pairings = []
    feature_names = vectorizer.get_feature_names()
    for col in X_validation_tfidf.nonzero()[1]:
        pairings.append( (feature_names[col], X_validation_tfidf[0, col]) )

    sorted_by_tfidf = sorted(pairings, key=lambda tup: tup[1], reverse=True)
    print sorted_by_tfidf[0:20]
    """

    return X_train_tfidf, X_validation_tfidf, vectorizer.get_feature_names()

# ***************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# REGRESSIONS
# -----------------------------------------------------------------------------------------------------
# ****************************************************************************************************

# arguments to regr.fit must be nupmpy matrices (use as_matrix() to convert Pandas sequence into numpy matrix (or scores) before sending)
# minimizes (sum of (y_predicted - y_actual)^2)
def linear_regression(train_model, train_targets, validation_model, validation_targets, features):
    regr = linear_model.LinearRegression(normalize=True)
    regr.fit(train_model, train_targets)

    feature_to_coef = {}
    for pairing in zip(features,regr.coef_):
        feature_to_coef[pairing[0]] = pairing[1]

    print '\nmax coef value: ' + str(max(feature_to_coef.itervalues()))
    print 'min coef value: ' + str(min(feature_to_coef.itervalues()))
    print '\nFeatures with highest regression coefficient values (positive in order):\n ' + str(sorted(feature_to_coef,key=feature_to_coef.get,reverse=True)[:150]) + '\n'
    print 'Features with lowest regression coefficient values (negative in order):\n ' + str(sorted(feature_to_coef,key=feature_to_coef.get,reverse=False)[:150]) + '\n'
    print_regression_metrics(regr, validation_model,validation_targets)
    return regr.predict(validation_model)

# sets non-informative coefficients to 0
def lasso_regression(train_model, train_targets, validation_model, validation_targets, features):
    regr = linear_model.Lasso(alpha=.3)
    regr.fit(train_model, train_targets)

    feature_to_coef = {}
    for pairing in zip(features,regr.coef_):
        feature_to_coef[pairing[0]] = pairing[1]

    print '\nmax coef value: ' + str(max(feature_to_coef.itervalues()))
    print 'min coef value: ' + str(min(feature_to_coef.itervalues()))
    print '\nFeatures with highest regression coefficient values (positive):\n ' + str(sorted(feature_to_coef,key=feature_to_coef.get,reverse=True)[:150]) + '\n'
    print 'Features with lowest regression coefficient values (negative):\n ' + str(sorted(feature_to_coef,key=feature_to_coef.get,reverse=False)[:150]) + '\n'
    print_regression_metrics(regr, validation_model,validation_targets)
    return regr.predict(validation_model)

# regression regularization model, good at elminiating uselss features
# minimizes (y_predicted - y_actual)^2 + lambda * sum of (abs(coefficient))), where lambda is a parameter
def SGD_regression(train_model, train_targets, validation_model, validation_targets, features):
   regr = linear_model.SGDRegressor(penalty="l2",shuffle=True)
   regr.fit(train_model, train_targets)
   #print_regression_metrics(regr, validation_model,validation_targets)
   
   feature_to_coef = {}
   for pairing in zip(features,regr.coef_):
       feature_to_coef[pairing[0]] = pairing[1]
       
   print '\nmax coef value: ' + str(max(feature_to_coef.itervalues()))
   print 'min coef value: ' + str(min(feature_to_coef.itervalues()))
   print '\nFeatures with highest regression coefficient values (positive):\n ' + str(sorted(feature_to_coef,key=feature_to_coef.get,reverse=True)[:150]) + '\n'
   print 'Features with lowest regression coefficient values (negative):\n ' + str(sorted(feature_to_coef,key=feature_to_coef.get,reverse=False)[:150]) + '\n'
   print_regression_metrics(regr, validation_model,validation_targets)
   return regr.predict(validation_model)

# regularized regression model, reduces over-fitting in training the model (using restaurant names as top features, for example)
def ridge_regression(train_model, train_targets, validation_model, validation_targets, features):
    regr = linear_model.Ridge(alpha=1.0,normalize=True)
    regr.fit(train_model, train_targets)
    feature_to_coef = {}
    for pairing in zip(features,regr.coef_):
        feature_to_coef[pairing[0]] = pairing[1]

    print '\nmax coef value: ' + str(max(feature_to_coef.itervalues()))
    print 'min coef value: ' + str(min(feature_to_coef.itervalues()))
    print '\nFeatures with highest regression coefficient values (positive in order):\n ' + str(sorted(feature_to_coef,key=feature_to_coef.get,reverse=True)[:150]) + '\n'
    print 'Features with lowest regression coefficient values (negative in order):\n ' + str(sorted(feature_to_coef,key=feature_to_coef.get,reverse=False)[:150]) + '\n'
    print_regression_metrics(regr, validation_model,validation_targets)
    return regr.predict(validation_model)

# TODO: doesn't work now
def random_forest_regression(model_train, train_targets, model_validation, validation_targets, features):
    regr = ensemble.RandomForestRegressor()
    regr.fit(train_model,train_targets)
    regr.predict(validation_model)

# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------------------------------
# ****************************************************************************************************

# prints metrics about data
def print_general_metrics(total_reviews, total_data_pts,data_retained):
    
    print '\nWe collected a total of ' + str(total_reviews) + ' yelp reviews across ' + str(total_data_pts) + ' data points.'
    # Amount of yelp data that had health scores (which we only used)
    print '' + floored_percentage(data_retained, 1) + ' of our data set was usable.\n'

# prints regression metrics, pass in the regression model, model_validation, validation_targets)
def print_regression_metrics(regr, model_validation, validation_targets):
    
    rmse = np.mean( (regr.predict(model_validation) - validation_targets) ** 2 ) 
    variance = regr.score(model_validation,validation_targets) 
    accuracy = np.mean(np.floor(regr.predict(model_validation)) == np.floor(validation_targets)) 

    lower = np.floor(regr.predict(model_validation)) >= (np.floor( validation_targets - 3))
    upper = np.floor(regr.predict(model_validation)) <= (np.floor( 3 + validation_targets))
    accuracy3 = np.mean( np.logical_and(lower,upper))

    lower = np.floor(regr.predict(model_validation)) >= (np.floor( validation_targets - 5))
    upper = np.floor(regr.predict(model_validation)) <= (np.floor( 5 + validation_targets))
    accuracy5 = np.mean( np.logical_and(lower,upper))

    # evaluate mean square error using validation set
    print("Residual mean squared error (lower=better): %.2f" % rmse)

    # Explained variance score: 1 is perfect prediction
    print('Variance score (1 is perfect): %.2f' % math.fabs(variance))

    # Accuracy of model
    print 'Accuracy WITHIN +- 0 range, measured on integer scores: ' + floored_percentage(accuracy, 2)

    # Accuracy of model within deviation of 3 from correction (e.g. if score is 80, we would be accurate if we predicted in range of (77,83) - inclusive
    print 'Accuracy WITHIN +- 3 range, measured on integer scores: ' + floored_percentage(accuracy3, 2)
 
    # Accuracy of model within deviation of 5 from correction
    print 'Accuracy WITHIN +- 5 range, measured on integer scores: ' + floored_percentage(accuracy5, 2)
    

# Make scatter plot of regression results, pass in validation model and validation targets
def plot_regression(v_model, v_targets):
    plt.scatter(v_model, v_targets,  color='black')
    plt.plot(v_model, regr.predict(v_model), color='blue',
                     linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

# Converts decimal to percentage
def floored_percentage(val, digits):
    val *= 10 ** (digits + 2)
    return '{1:.{0}f}%'.format(digits, math.floor(val) / 10 ** digits)

def plot_basic_regr(coef, intercept):
    r_x, r_y = zip(*((i, i*coef + intercept) for i in range(6)))

    output_file("regression.html")
    line(r_x, r_y, color="red")
    # Specify that the two graphs should be on the same plot.
    hold(True)
    scatter(r_x, r_y, marker="square", color="blue",
            title = "Regress yelp review scores on health scores",x_axis_label = "Yelp Review Score",y_axis_label = "Health Score")
    show()


if __name__ == "__main__":
    main()

'''

    # TextBlob() takes in a string, so need to convert panads Series to list, then string corpus
    review_corpus_train = ""
    review_texts_train = train_text.tolist()
    review_corpus_train = ' '.join(review_texts_train)
    blob_train = TextBlob(unicode(review_corpus_train, 'utf-8'))
    blob_train.noun_phrases

    review_corpus_val = ""
    review_texts_val = validation_text.tolist()
    review_corpus_val = ' '.join(review_texts_val)
    blob_val = TextBlob(unicode(review_corpus_val, 'utf-8'))
    
    # attempt to spell check sentence, takes way too long
    #blob.correct()

    # remove very objective sentences (below 0.2)
    #corpus_subj = ""
    #for sentence.correct() in blob.sentences:
    #    if sentence.sentiment.subjectivity > 0.2:
    #        corpus_subj += sentence
    #blob_subj = TextBlob(unicode(corpus_subj, 'utf-8'))

    # use polarity as main feature. Polarity can range from -1 to 1 inclusive, -1 is very negative, 1 is very positive
    #polarity = blob_subj.sentiment.polarity
    #print polarity

'''






