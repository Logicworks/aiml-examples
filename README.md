

# Triumph Tech - Amazon SageMaker Examples

Example Jupyter notebooks that demonstrate how to use [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html) and the [Amazon SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/).

## 📓 Examples

| Notebook                                                                 | Type       | Description                                                                                                                                          |
|--------------------------------------------------------------------------|------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| [01 Extending AWS DLC image](./samples/01_extending_aws_dlc_images/extending-image.ipynb)                                                  | Containers | Learn how to extend the functionality of the AWS Deep Learning Container (DLC) image to meet your specific requirements.                             |
| [02 Deploy Stable Diffusion2.1 with Custom image and custom script](./samples/02_deploy_stable_diffusion2_1_with_custom_image_and_custom_script/deploy-stable-diffusion-2-1.ipynb)           | Inference  | End-to-end example of deploying Stable Diffusion2.1 by utilizing a custom image and custom script tailored to your needs.                           |
| [03 Deploy Stable Diffusion2.1 with DJL serving](./samples/03_deploy_stable_diffusion2_1_with_djl_serving/deploy-stable-diffusion-2-1_with_djl_serving.ipynb)                              | Inference  | End-to-end example of deploying Stable Diffusion2.1 using DJL serving, enabling efficient serving of your model for inference.                                   |
| [04 Deploy Stable Diffusion2.1 with Inferentia2](./samples//04_deploy_stable_diffusion2_1_with_inferentia_inf2//deploy-stable-diffusion-2-1_inf2.ipynb)                              | Inference  | End-to-end example on how to deploy Stable Diffusion2.1 using Inferentia2, taking advantage of AWS's machine learning accelerator.                       |
| [05 How to create an AI image generator application using vectorize database](./samples/05_how_to_create_an_ai_image_generator_application_using_vectorize_database) | POC        | Develop an AI application capable of generating images using a vectorized database, enabling flexible and dynamic image synthesis.                   |
| How to create a multimodal AI agent (txt2txt,txt2img)                    | POC        | Build a powerful multimodal AI agent capable of performing both text-to-text and text-to-image tasks, expanding its range of capabilities.           |
| Train BERT model with AWS Trainium                                       | Training   | End-to-end training and deploying of a BERT (Bidirectional Encoder Representations from Transformers) model using AWS Trainium, optimizing your training workflow.     |
| Deploy BERT model on AWS Inferentia                                      | Inference  | Explore the deployment process of a BERT model on AWS Inferentia, leveraging its high-performance capabilities for efficient inference.              |
| Deploy BERT model on AWS Inferentia2                                     | Inference  | Learn how to deploy a BERT model on AWS Inferentia2, harnessing the power of AWS's advanced machine learning accelerator for enhanced performance.   |
| Quantize and deploy LLM using Hugging Face LLM Inference Container       | Inference  | Discover how to quantize and deploy a Language Model (LLM) using the Hugging Face LLM Inference Container, enabling efficient and optimized serving. |