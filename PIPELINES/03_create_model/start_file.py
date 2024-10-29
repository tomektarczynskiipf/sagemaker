
from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd

from sklearn.linear_model import LogisticRegression

# There is no default function to load the model
# Without this function the job will fail!
def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

# There is a default function to calculate the predictions.
# It calculates the class 0/1 instead of probability
# That is why we should override it with a custom function
def predict_fn(input_data, model):
    pred_prob = model.predict_proba(input_data)
    return pred_prob
