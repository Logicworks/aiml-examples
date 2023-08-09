
ARG REGION=us-west-2

# Use the SageMaker PyTorch image as the base
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04

# Install necessary Python libraries
RUN pip install deepspeed diffusers==0.11.1 transformers==4.25.1 scipy==1.9.3 accelerate==0.15.0
