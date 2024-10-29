
from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd

from sklearn import tree

if __name__ == '__main__':
    model_dir = os.environ['SM_MODEL_DIR'] # Folder where model must be saved
    train_dir = os.environ['SM_CHANNEL_TRAIN'] # Folder where train data is stored

    # Lets assume there is only one training file
    train_file_name = os.listdir(train_dir)[0]
    train_file_path = os.path.join(train_dir, train_file_name)
    
    train_data = pd.read_csv(train_file_path, header=None, engine="python")

    # labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:]  

    # Now use scikit-learn's decision tree classifier to train the model.
    # Hyperparameters are hardcoded
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=30)
    clf = clf.fit(train_X, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))
