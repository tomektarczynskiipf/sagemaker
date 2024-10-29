
from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd

from sklearn.linear_model import LogisticRegression

# Function is overwritten to include header
def input_fn(request_body, content_type='text/csv'):
    if content_type == 'text/csv':
        data = pd.read_csv(request_body, header=True)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    return data

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
    clf = LogisticRegression(max_iter=100)
    clf = clf.fit(train_X, train_y)

    # Save model object
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))

    script_content = """
import os
import joblib
import numpy as np

def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    return model

def predict_fn(input_data, model):
    prediction = model.predict_proba(input_data)
    return prediction
"""
    script_path = os.path.join(model_dir, "inference.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    print(f"Inference script created at {script_path}")
