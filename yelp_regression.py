### Extract features from csv data, namely review_text. 
### Then, select important features

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# do all data frame manipulation here
def main():
    # Load data set (includes both training and validation)
    df = pd.read_csv("/home/datascience/FINAL_PROJECT/yelp_predictor/dataset_yelp.csv",sep='|')

    # Load data set (includes test set)
    df_test = pd.read_csv("/home/datascience/FINAL_PROJECT/yelp_predictor/dataset_unknown.csv",sep='|')

    # Load rejects (data we failed to get because yelp had no yelp reviews)

    len_rejects = 0
    try:
        df_rejects = pd.read_csv("/home/datascience/FINAL_PROJECT/yelp_predictor/dataset_rejects.csv",sep='|')
        len_rejects = len(df_rejects.index)
    except:
        pass

    data_retained = len(df.index) / float( len_rejects + len(df.index) )
    
    # Split the data into 80% training, 20% validation (includes random shuffling)
    rows_train = np.random.choice(df.index.values, int(math.floor(len(df.index)*.8)), replace=False)
    df_train = df.ix[rows_train]
    df.drop(rows_train)
    rows_validation = np.random.choice(df.index.values, len(df.index), replace=False)
    df_validation = df.ix[rows_validation]

    # Select data (review text) and name of restaurant (name) and target (health score)
    df_train_name = df_train['name']
    df_train_text = df_train['review_text']
    df_train_score = df_train['health_score']
    df_validation_name = df_validation['name']
    df_validation_text = df_validation['review_text']
    df_validation_score = df_validation['health_score']

    model_train_tfidf, model_validation_tfidf = build_feature_extractors(df_train_text, df_validation_text)
    coefs, rms, variance = linear_regression(model_train_tfidf, df_train_score, model_validation_tfidf, df_validation_score)

    # Amount of yelp data that had health scores (which we only used)
    print '' + floored_percentage(data_retained, 1) + ' of our data set was usable.'
    # The coefficients results for validation set that we got
    for name, coef in zip(df_validation_name, coefs):
        print('%r => %s' % (name, coef))

    # evaluate mean square error using validation set
    print("Residual sum of squares: %.2f" % rms)

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % variance)


# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# FEATURE EXTRACITON & SELECTION
# -----------------------------------------------------------------------------------------------------
# *****************************************************************************************************

# TODO: returns bag of words model, in tuple (train, validation)
def build_feature_extractors(train_text, validation_text):

    # 1. starting with a simple bag of words model
    # training set
    vectorizer_bag = CountVectorizer(min_df=1)
    X_train_counts = vectorizer_bag.fit_transform(train_text)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # validation set
    Y_validation_counts = vectorizer_bag.transform(validation_text)
    Y_validation_tfidf = tfidf_transformer.transform(Y_validation_counts)

    return X_train_tfidf, Y_validation_tfidf

    '''
    # 2. n-gram model, test which one is best by passing in diff values for n (most likely between 2-4)
    bigram_vectorizer = CountVectorizer(ngram_range(1,2),
                           token_pattern=r'\b\w+\b', min_df=1)
    X_2 = bigram_vectorizer.fit_transform(corpus)
    
    trigram_vectorizer = CountVectorizer(ngram_range(1,3),
                           token_pattern=r'\b\w+\b', min_df=1)
    X_3 = bigram_vectorizer.fit_transform(corpus)
    
    # 3. TODO: dependency parsing with a fast DP like maltparser.

    '''

# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# REGRESSIONS
# -----------------------------------------------------------------------------------------------------
# ****************************************************************************************************

# test linear regression first
# later test ridge, randomforestregressor, regularized regression
# returns (coefficients, residual sum of squares, variance)
def linear_regression(model_train, train_targets, model_validation, validation_targets):
    regr = linear_model.LinearRegression()

    # train the model
    regr.fit(model_train, train_targets)

    return (regr.coef_, np.mean((regr.predict(model_validation) - validation_targets) ** 2), 
            regr.score(model_validation, validation_targets))



# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# MISC
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

def floored_percentage(val, digits):
    val *= 10 ** (digits + 2)
    return '{1:.{0}f}%'.format(digits, math.floor(val) / 10 ** digits)


if __name__ == "__main__":
    main()








