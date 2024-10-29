import boto3
import sagemaker
import sagemaker.session
import os
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import TrainingStep
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import TransformStep
from sagemaker.inputs import TransformInput
from sagemaker.workflow.pipeline import Pipeline

def create_pipeline():

    region = boto3.Session().region_name
    sagemaker_session = sagemaker.session.Session()
    role = sagemaker.get_execution_role()
    default_bucket = sagemaker_session.default_bucket()
    
    pipeline_session = PipelineSession()
    
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type='ml.t3.medium',
        instance_count=1,
        base_job_name="05-full-pipeline",
        sagemaker_session=pipeline_session,
        role=role,
    )
    
    processor_args = sklearn_processor.run(
        inputs=[],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination='s3://sagemaker-bucket-ds/PIPELINE/05-full-pipeline/train'),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination='s3://sagemaker-bucket-ds/PIPELINE/05-full-pipeline/test'),
            ProcessingOutput(
                output_name="inference",
                source="/opt/ml/processing/output/inference",
                destination='s3://sagemaker-bucket-ds/PIPELINE/05-full-pipeline/inference'),        
    
        ],
        code="CODES/processing.py",
    ) 
    
    step_process = ProcessingStep(
        name="Preprocessing",
        step_args=processor_args
    )
    
    sklearn = SKLearn(
        entry_point='train.py', # The file with the training code
        source_dir='CODES', # The folder with the training code
        framework_version='1.2-1', # Version of SKLearn which will be used
        instance_type='ml.m5.large', # Instance type that wil be used
        role=role, # Role that will be used during execution
        sagemaker_session=pipeline_session, 
        base_job_name='05_full_pipeline' # Name of the training job. Timestamp will be added as suffix
    )
    
    train_args = sklearn.fit({"train": step_process.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri})
    
    step_train = TrainingStep(
        name="SimpleTrain",
        step_args = train_args
    )
    
    # Create the SKLearnModel
    sklearn_model = SKLearnModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        entry_point='start_file.py', # The file with the training code
        source_dir="CODES", # The folder with the training code
        role=role,
        framework_version='1.2-1',  # Replace with the appropriate sklearn version
        sagemaker_session=pipeline_session
    )
    
    step_create_model = ModelStep(
       name="MyModelCreationStep",
       step_args=sklearn_model.create(instance_type="ml.m5.large"),
    )
    
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.large",
        instance_count=1,
        output_path=f"s3://sagemaker-bucket-ds/PIPELINE/05-full-pipeline/INFERENCE_OUTPUT/"
    )
    
    step_transform = TransformStep(
        name="MyTransformStep",
        transformer=transformer,
        inputs=TransformInput(
            data=step_process.properties.ProcessingOutputConfig.Outputs['inference'].S3Output.S3Uri,
            content_type='text/csv', # It is neccessary because csv is not default format
            split_type='Line' # Each line equals one observation)
    ))
    
    pipeline_name = f"06-full-pipeline"
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_process, step_train, step_create_model, step_transform],
    )
    
    pipeline.upsert(role_arn=role)
    
    #execution = pipeline.start()