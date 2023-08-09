#!/bin/bash

curl --request POST \
     --url https://dream-cacher-xxxxx.svc.us-west4-gcp-free.pinecone.io/vectors/delete \
     --header 'Api-Key: xxxxx' \
     --header 'accept: application/json' \
     --header 'content-type: application/json' \
     --data '{"deleteAll":true}'
