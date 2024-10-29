
from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--objective', type=str, default='binary:logistic')

    args = parser.parse_args()

    model_dir = os.environ['SM_MODEL_DIR'] # Folder where model must be saved
    train_dir = os.environ['SM_CHANNEL_TRAIN'] # Folder where train data is stored
    output_dir = os.environ['SM_OUTPUT_DATA_DIR']
    
    # Lets assume there is only one training file
    train_file_name = os.listdir(train_dir)[0]
    train_file_path = os.path.join(train_dir, train_file_name)
    train_data = pd.read_csv(train_file_path, header=None, engine="python")
    
    # Separate features and labels
    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:]  

    # Track training results
    eval_results = {}

    # Train the model
    clf = xgb.XGBClassifier(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        objective=args.objective)
    clf.fit(train_X, train_y, eval_set=[(train_X, train_y)], eval_metric="logloss", early_stopping_rounds=10, verbose=True, callbacks=[xgb.callback.record_evaluation(eval_results)])


    # Save the model
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))

    # Create and save the chart
    epochs = len(eval_results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, eval_results['validation_0']['logloss'], label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(os.path.join(output_dir, "training_log_loss.png"))
    plt.close()
