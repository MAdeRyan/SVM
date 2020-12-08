import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bankdata = pd.read_csv("accent-mfcc-data-1.csv")
bankdata.shape
bankdata.head()

X = bankdata.drop('language', axis=1)
y = bankdata['language']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)