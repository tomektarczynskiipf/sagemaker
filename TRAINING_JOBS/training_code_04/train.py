
from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd
import xgboost as xgb

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

    # Train the model
    # Hyperparameters are hardcoded
    clf = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.1)
    clf = clf.fit(train_X, train_y)

    # Save model object
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))
