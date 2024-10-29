import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    args = parser.parse_args()

    # Load the training data
    train_data = pd.read_csv(args.train)

    # Assume the last column is the target variable
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, '/opt/ml/model/model.joblib')