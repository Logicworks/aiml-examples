import boto3
from sagemaker.utils import name_from_base

# Create an endpoint configuration (assuming it already exists)
endpoint_config_name = 'stable-diffusion-model-txt2img-stabilit-2023-05-24-21-35-07-375'
endpoint_name = name_from_base('stable-diffusion-model-demo')

sagemaker_client = boto3.client('sagemaker')
response = sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name,
)
print(endpoint_name)