from __future__ import print_function
import boto3
from botocore.exceptions import NoCredentialsError

if __name__ == '__main__':

    def upload_to_s3(bucket_name, file_name, content):
        # Create an S3 client
        s3 = boto3.client('s3')
    
        try:
            # Upload the file
            s3.put_object(Bucket=bucket_name, Key=file_name, Body=content)
            print(f"File '{file_name}' uploaded to bucket '{bucket_name}'.")
        except NoCredentialsError:
            print("Credentials not available.")
    
    # Define your bucket name, file name, and content
    bucket_name = 'sagemaker-bucket-ds'
    file_name = '01_example.txt'
    content = 'Hello, this is a test file.'
    
    # Upload the file
    upload_to_s3(bucket_name, file_name, content)
