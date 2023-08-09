# Amazon Deep Learning Containers (DLC) - Extending base containers

This repository contains one notebook that assist in the creation of a Docker image and an ECR (Elastic Container Registry) image.

## 📓 extending-image notebook
In this notebook we will show you how to extend a base DLC image with libraries that fits your specific needs. We will begin by specifying a base image that we will retrieve from Amazon Elastic Container Registry (ECR). Then, we will proceed to install additional Python libraries necessary for running stable diffusion-2.1.

- deepspeed
- diffusers==0.11.1
- transformers==4.25.1
- scipy==1.9.3
- accelerate==0.15.0

The installation of deepspeed enables us to utilize its capabilities in our inference script. Once the Docker container is constructed, it can be pushed to ECR for future utilization.