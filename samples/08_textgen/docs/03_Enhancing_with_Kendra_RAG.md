# Enhancing with Kendra RAG

Expand your model's capabilities by integrating AWS Kendra with Retrieval Augmented Generation (RAG) to improve its question-answering abilities.

## Activating and Configuring Kendra

1. Set up a Kendra index in the AWS console.
2. Connect Kendra to an S3 bucket containing the necessary documents.

## Direct Use with TextGen

1. Ensure the API extension for Kendra RAG is active.
2. Navigate to the 'Chat' tab in GenFlow.
3. Enter the Kendra Index ID and activate the AWS Kendra integration.

## Integration via SageMaker Inference Endpoint

1. Access the SageMaker inference tab in TextGen.
2. Confirm you have an active SageMaker endpoint and a configured Kendra index.
3. Enter details and activate the features needed for RAG.
4. Customize the LangChain template and refine prompt engineering for better performance.

By following these steps, your chatbot will have enhanced knowledge retrieval abilities, leveraging Kendra's capabilities.
