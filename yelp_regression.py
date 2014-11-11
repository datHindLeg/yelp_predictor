### Extract features from csv data, namely review_text. 
### Then, select important features
### lastly, perform regression

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn import linear_model
from sklearn import ensemble
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# DATA MANIPULATION AND PRINTOUT 
# -----------------------------------------------------------------------------------------------------
# *****************************************************************************************************

def main():
    # Load data set (includes both training and validation)
    df = pd.read_csv("/home/datascience/FINAL_PROJECT/yelp_predictor/dataset_yelp.csv",sep='|')

    # Load data set (includes test set)
    df_test = pd.read_csv("/home/datascience/FINAL_PROJECT/yelp_predictor/dataset_unknown.csv",sep='|')

    # Load rejects (data we failed to get because yelp had no health scores)
    len_rejects = 0
    try:
        df_rejects = pd.read_csv("/home/datascience/FINAL_PROJECT/yelp_predictor/dataset_rejects.csv",sep='|')
        len_rejects = len(df_rejects.index)
    except:
        pass

    data_retained = len(df.index) / float( len_rejects + len(df.index) )

    # Count number of yelp reviews we got total
    df_number_reviews = df['number_reviews']
    df_test_number_reviews = df_test['number_reviews']
    total_reviews = df_number_reviews.sum(axis=1) + df_test_number_reviews.sum(axis=1)

    # Count number of data points (rows we got)
    total_data_pts = len(df.index) + len(df_test.index)

    # Split the data into 80% training, 20% validation (includes random shuffling)
    # TODO: PERFORM K FOLDS
    rows_train = np.random.choice(df.index.values, int(math.floor(len(df.index)*.8)), replace=False)
    df_train = df.ix[rows_train]
    df = df.drop(rows_train)
    rows_validation = np.random.choice(df.index.values, len(df.index), replace=False)
    df_validation = df.ix[rows_validation]

    # Select data (review text) and name of restaurant (name) and target (health score)
    df_train_name = df_train['name']
    df_train_text = df_train['review_text']
    df_train_score = df_train['health_score']
    df_validation_name = df_validation['name']
    df_validation_text = df_validation['review_text']
    df_validation_score = df_validation['health_score']

    # heavy lifting in these 2 lines
    model_train_tfidf, model_validation_tfidf = build_feature_extractors(df_train_text, df_validation_text)
    values, rms, variance = linear_regression(model_train_tfidf, df_train_score, model_validation_tfidf, df_validation_score)

    print '\nWe collected a total of ' + str(total_reviews) + ' yelp reviews across ' + str(total_data_pts) + ' data points.'
    # Amount of yelp data that had health scores (which we only used)
    print '' + floored_percentage(data_retained, 1) + ' of our data set was usable.\n'

    # The value results for validation set that we got
    print 'name', 'score\n'
    for name, value in zip(df_validation_name, values):
        print name + ' => ' + str(int(math.floor(value))) + '\n'

    # evaluate mean square error using validation set
    print("Residual mean error: %.2f" % rms)

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % math.fabs(variance))


# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# FEATURE EXTRACTION & SELECTION
# -----------------------------------------------------------------------------------------------------
# *****************************************************************************************************

# TODO: returns bag of words model, in tuple (train, validation)
def build_feature_extractors(train_text, validation_text):

    # 1. starting with a simple bag of words model
    # training set
    #vectorizer = CountVectorizer(min_df=1)

    # 2. bi-gram
    vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)

    # 3. tri-gram
    #vectorizer = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)

    # 4. TODO: dependency parsing with a fast DP like maltparser.

    X_train_counts = vectorizer.fit_transform(train_text)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # validation set
    Y_validation_counts = vectorizer.transform(validation_text)
    Y_validation_tfidf = tfidf_transformer.transform(Y_validation_counts)

    # DEBUG: find out what features are being selected
    #print_features(X_train_tfidf)
    #print_features(vectorizer)

    return X_train_tfidf, Y_validation_tfidf


# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# REGRESSIONS
# -----------------------------------------------------------------------------------------------------
# ****************************************************************************************************

# test linear regression first
# later test ridge, randomforestregressor, regularized regression
# returns (values, diff, residual sum of squares, variance)
def linear_regression(model_train, train_targets, model_validation, validation_targets):
    regr = linear_model.LinearRegression()

    # train the model
    regr.fit(model_train, train_targets)

    return (regr.predict(model_validation), np.mean((regr.predict(model_validation) - validation_targets) ** 2), 
            regr.score(model_validation, validation_targets))

def ridge_regression(model_train, train_targets, model_validation, validation_targets):
    regr = linear_model.Ridge(alpha=1.0)

    # train the model
    regr.fit(model_train, train_targets)

    return (regr.predict(model_validation), np.mean((regr.predict(model_validation) - validation_targets) ** 2), 
            regr.score(model_validation, validation_targets))

# TODO: doesn't work now
def random_forest_regression(model_train, train_targets, model_validation, validation_targets):
    regr = ensemble.RandomForestRegressor()

    # train the model
    regr.fit(model_train.toarray(), train_targets.toarray())

    return (regr.predict(model_validation), np.mean((regr.predict(model_validation) - validation_targets) ** 2), 
            regr.score(model_validation, validation_targets))



# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------------------------------
# ****************************************************************************************************

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

# Prints features for the model we used
def print_features(model):
    try:
        print model.get_feature_names()
    except:
        print 'ERROR: The model you passed in is not in the right format. Try passing a vectorizer'


if __name__ == "__main__":
    main()








