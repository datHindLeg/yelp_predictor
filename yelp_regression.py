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
    # Load data set
    df = pd.read_csv("/home/datascience/FINAL_PROJECT/yelp_predictor/dataset_yelp.csv",sep='|')
    
    # Split the data into 80% training, 20% validation (includes random shuffling)
    rows_train = np.random.choice(df.index.values, int(math.floor(len(df.index)*.8)), replace=False)
    df_train = df.ix[rows_train]
    df.drop(rows_train)
    rows_validation = np.random.choice(df.index.values, len(df.index), replace=False)
    df_validation = df.ix[rows_validation]

    # Select data (review text) and target (health score)
    df_train_text = df_train['review_text']
    df_train_score = df_train['health_score']
    df_validation_text = df_validation['review_text']
    df_validation_score = df_validation['health_score']

    model_X = build_feature_extractors(df_train_text)
    linear_regression(model_x)

# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# DEFINE FEATURE EXTRACTION MODELS
# -----------------------------------------------------------------------------------------------------
# *****************************************************************************************************

# TODO: currently returns bag_of_worlds only, later: returns a tuple of 4 feature extraction models (X, X_2, X_3, X_4)
def build_feature_extractors(train_text, validation):

    # 1. starting with a simple bag of words model
    vectorizer_bag = CountVectorizer(min_df=1)
    X_train_counts = vectorizer_bag.fit_transform(train_text)
    X_validation_counts = vectorizer_bag.transform(scores)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    return X_train_tfidf

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
# PERFORM FEATURE SELECTION 
# -----------------------------------------------------------------------------------------------------
# *****************************************************************************************************

def linear_regression(model):
    regr = linear_model.LinearRegression()
    pass


if __name__ == "__main__":
    main()








