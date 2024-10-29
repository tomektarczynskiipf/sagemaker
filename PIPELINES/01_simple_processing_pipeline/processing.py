
import os
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["class"] = pd.Series(iris.target)
    df = df[df['class'].isin([0, 1])] # Lets keep only class 0 and 1 to have binary classification
    df = df[[list(df.columns)[-1]] + list(df.columns)[:-1]] # Reorder target as the first column
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    
    train_df, test_df = train_test_split(df, test_size=0.33, random_state=42, stratify=df["class"])
    
    iris_train = train_df.to_numpy()
    np.savetxt('/opt/ml/processing/output/iris_train.csv', iris_train, delimiter=',', fmt='%1.1f, %1.3f, %1.3f, %1.3f, %1.3f')
    
    iris_test = test_df.to_numpy()
    np.savetxt('/opt/ml/processing/output/iris_test.csv', iris_test, delimiter=',', fmt='%1.1f, %1.3f, %1.3f, %1.3f, %1.3f')
    
    iris_inference = test_df.iloc[:, 1:].to_numpy()
    np.savetxt('/opt/ml/processing/output/iris_inference.csv', iris_inference, delimiter=',', fmt='%1.3f, %1.3f, %1.3f, %1.3f')
    
    column_names_list = ','.join(df.columns)
    with open('/opt/ml/processing/output//column_names.csv', 'w') as file:
        file.write(column_names_list)
