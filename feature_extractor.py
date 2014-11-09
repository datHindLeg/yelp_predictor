import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model

from sklearn.feature_extraction.text import CountVectorizer

# Load data set
df = pd.read_csv("/home/datascience/FINAL_PROJECT/yelp_predictor/dataset_yelp.csv")

# Select data (review text) and target (health score)
df_text = df['review_text', 'health_score']

# Split the data into 70% training, 30% testing
# Also randomly sample / shuffle rows

rows_traini_text = np.random.choice(df_text.index.values, len(df_text.index)*.7, replace=False)

df_train = df_text.ix[rows_train]
df_text.drop(rows_train)

rows_test = np.random.choice(df_text.index.values, len(df_text.index), replace=False)
df_test = df_test.ix[rows_test]


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)




