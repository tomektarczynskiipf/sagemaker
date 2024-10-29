from __future__ import print_function
import pandas as pd
import os

if __name__ == '__main__':

    for key, value in os.environ.items():
        print(f"{key}: {value}")

    input_path = "/opt/ml/processing/input/myinput/"
    output_path = '/opt/ml/processing/output/'

    input_file_path = os.path.join(input_path, "sample_data.csv")
    output_file_path = os.path.join(output_path, "output.csv")
    
    # Read the CSV file
    df = pd.read_csv(input_file_path)
    
    # Calculate the sum of all columns
    column_sums = df.sum()
    
    # Store the sums in a text file
    with open(output_file_path, 'w') as f:
        for column, sum_value in column_sums.items():
            f.write(f'{column}: {sum_value}\n')
