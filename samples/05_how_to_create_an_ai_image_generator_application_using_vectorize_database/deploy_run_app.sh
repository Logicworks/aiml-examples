#!/usr/bin/env bash

DOMAIN_ID=$1
REGION=$2
PORT=${4:-8080}

echo https://${DOMAIN_ID}.studio.${REGION}.sagemaker.aws/jupyter/default/proxy/${PORT}/

endpoint_name=$(python deploy.py)
streamlit run app_test.py "$endpoint_name" --server.port "$PORT"
