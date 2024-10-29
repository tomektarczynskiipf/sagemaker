from pipeline import create_pipeline

def lambda_handler(event, context):
    try:
        create_pipeline()
        return {
            'statusCode': 200,
            'body': 'Pipeline created successfully'
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': str(e)
        }