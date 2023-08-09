from typing import Optional
import boto3
import json
import base64
import os
from io import BytesIO
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import uuid
import time
from itertools import cycle
from datetime import datetime
import copy
import numpy as np
import ast
import pinecone
from pinecone import PineconeProtocolError
import time
import sys

endpoint_name = sys.argv[1]

# Use the endpoint_name as needed in your second Python script
print(f"Endpoint name received: {endpoint_name}")

ENDPOINT_NAME = endpoint_name or "sagemaker-jumpstart-2023-05-23-19-25-54-548"

S3_BUCKET = "stable-diffusion-2-1-demo"

PINECONE_API_KEY = "xxxxxx-xxxxx-xxxx-xxxxx-xxxxx"
PINECONE_ENVIRONMENT = "us-west4-gcp-free"
PINECONE_INDEX_NAME = "dream-cacher"

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

search = None

# Set up Boto3 client for S3
s3_client = boto3.client('s3')
sm_runtime = boto3.client("runtime.sagemaker")

def get_image(key: str) -> Optional[Image.Image]:
    if key in st.session_state:
        return st.session_state[key]
    return None


def set_image(key: str, img: Image.Image):
    st.session_state[key] = img

def decode_base64_image(image_string):
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    return Image.open(buffer)

def invoke_sagemaker_endpoint(input_dict, endpoint_name):

    model_input_dict = json.dumps(input_dict, indent=2).encode("utf-8")

    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=model_input_dict,
        ContentType="application/json",
    )
    output = json.loads(response["Body"].read().decode())

    if input_dict['mode'] == 'text2vector':
        return output['generated_vector']

    # decode images
    decoded_output = [decode_base64_image(image) for image in output["generated_images"]]
    seeds = output["seeds"]

    return decoded_output, seeds

def search_tab(inference_json,s3_image_prefix,s3_inference_json_uri):
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown(
            """
            <style>
            span {
                text-align: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # st.title("Stable diffusion 2.1 search")
        search = st.text_input(
                "",
                value="",
                key=f"search",
                placeholder="Search"
            )

        if st.button("Search") and search > '':
            with st.spinner("Searching images..."):
                request_dict = {
                  "inputs": search,
                  "mode": 'text2vector',
                }


                output_vector = invoke_sagemaker_endpoint(request_dict, ENDPOINT_NAME)
                output_vector = ast.literal_eval(output_vector)
                if search:
                    pc_index = st.session_state['pinecone']
                    xc = pc_index.query(
                       output_vector, top_k=20, include_metadata=True
                    )

                    matches = [match['metadata'] for match in xc['matches'] if match['score'] > .1]
                    image_files = [image['image_s3_uri'].split(S3_BUCKET)[1][1:] for image in matches]
                    captions = [match['inputs'] for match in matches]

                        # Set grid width
                    grid_width_value = 3

                    images = []
                    for idx, path in enumerate(image_files):
                        image = get_image_from_s3(S3_BUCKET,path)
                        images.append(image)

                    caption = captions

                    # Display images
                    display_images(images,caption,grid_width_value)





def generate_tab(inference_json,s3_image_prefix,s3_inference_json_uri):

    col1, col2 = st.columns(2)
    search = None

    with col1:

        prompt = st.text_area(
            "Prompt",
            value="border collie puppy",
            key=f"prompt",
        )
        negative_prompt = st.text_area(
            "Negative prompt",
            value="",
            key=f"negative-prompt",
        )
        seed = st.text_input(
            "Seed",
            value="",
            key=f"seed",
        )

    with col2:

        number_of_output = st.slider(
            label='Output Count',
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Specify number of images to output. More images takes longers.")

        num_inference_steps = st.slider(
            label='Inference steps',
            min_value=25,
            max_value=100,
            value=25,
            step=25,
            help="Number of inference steps.")

        strength_value = st.slider(
            label='Strength',
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="Conceptually, indicates how much to transform the reference image. Must be between 0 and 1. image will be used as a starting point, adding more noise to it the larger the strength. The number of denoising steps depends on the amount of noise initially added. When strength is 1, added noise will be maximum and the denoising process will run for the full number of iterations specified in num_inference_steps. A value of 1, therefore, essentially ignores image.")

        guidance_value = st.slider(
            label='Guidance',
            min_value=5.0,
            max_value=15.0,
            value=7.5,
            step=0.5,
            help="Guidance scale as defined in Classifier-Free Diffusion Guidance. guidance_scale is defined as w of equation 2. of Imagen Paper. Guidance scale is enabled by setting guidance_scale > 1. Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.")

    # Set grid width
    grid_width_value = st.slider(
            "Columns",
            min_value=1,
            max_value=10,
            step=1,
            value=3,
            key="gen-width"
        )

    request_dict = {
      "inputs": prompt,
      "negative_prompt": negative_prompt,
      "num_images_per_prompt" : number_of_output,
      "num_inference_steps": num_inference_steps,
      "strength": strength_value,
      "guidance": guidance_value,
    }

    if seed != "":
        request_dict["seed"] = int(seed)


    image=image_uploader()
    generated_vector = None;

    if image:
        st.image(image)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image = base64.b64encode(buffered.getvalue()).decode()
        request_dict["image"]=image
        request_dict["mode"]="image2image"
    else:
        request_dict["mode"]="text2image"

    if st.button("Generate"):
        with st.spinner("Generating image..."):
            print("generate pinecone init")
            pc_index = st.session_state['pinecone']
            generated_vector = invoke_sagemaker_endpoint({**request_dict, 'mode': 'text2vector' }, ENDPOINT_NAME)
            generated_vector = ast.literal_eval(generated_vector)
            output_images, seeds = invoke_sagemaker_endpoint(request_dict, ENDPOINT_NAME)


        if request_dict["mode"] == "text2image":
            caption = [f"image_{i} {seeds[i-1]}" for i in range(1, len(output_images)+1)]
        if request_dict["mode"] == "image2image":
            caption = [f"image_{i}" for i in range(1, len(output_images)+1)]
        display_images(output_images,caption,grid_width_value)
        s3_paths=write_image_to_s3(output_images,S3_BUCKET,s3_image_prefix)
        pine_arr = []
        for idx, p in enumerate(s3_paths, start=0):
            _id = str(uuid.uuid4())  # creates format "xxxx-xxxx-xxxx-xxxx"


            timestamp = datetime.utcnow()
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            request_dict_copy = copy.deepcopy(request_dict)
            request_dict_copy["image_s3_uri"]=p
            request_dict_copy["generated_timestamp"]=timestamp_str
            request_dict_copy['seed'] = seeds[idx] if request_dict["mode"] == "text2image" else ''
            pine_arr.append((_id, generated_vector, request_dict_copy))
            # request_dict_copy['generated_vector']=generated_vector
            inference_json.append(request_dict_copy)
        pc_index.upsert(pine_arr)
        write_json_to_s3(inference_json,S3_BUCKET,s3_inference_json_uri)


def image_uploader():
    image = st.file_uploader("Image", ["jpg", "png", "jpeg"], key=f"uploader")
    if image:
        image = Image.open(image)
        print(f"loaded input image of size ({image.width}, {image.height})")
        image = image.resize((DEFAULT_WIDTH, DEFAULT_HEIGHT))
        return image


def get_image_files(bucket_name, prefix):
    # List all objects in the bucket with the specified prefix
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # Get the image file names
    image_files = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        if is_image_file(key):
            image_files.append(key)

    return image_files

def is_image_file(file_name):
    # Check if the file has an image extension
    valid_extensions = [".jpg", ".jpeg", ".png"]
    file_extension = file_name[file_name.rfind('.'):]
    return file_extension.lower() in valid_extensions

def display_images(images,caption,max_grid_width):

    cols = cycle(st.columns(max_grid_width))
    for idx, images in enumerate(images):
        next(cols).image(images, caption=caption[idx])


def history_tab(s3_image_prefix, inference_json):

    # Set grid width
    grid_width_value = st.slider(
            "Columns",
            min_value=1,
            max_value=10,
            step=1,
            value=3
        )
    # Get all image files from the bucket
    image_files_existing = get_image_files(S3_BUCKET,s3_image_prefix)

    image_files = [i['image_s3_uri'].split(S3_BUCKET)[1][1:] for i in inference_json][::-1]

    images = []
    for idx, path in enumerate(image_files):
        image = get_image_from_s3(S3_BUCKET,path)
        images.append(image)

    seeds = [i['seed'] if 'seed' in i else '' for i in inference_json]
    caption = [f"""{inference_json[i-1]['inputs']} -- {seeds[i-1]}""" for i in range(1, len(images)+1)][::-1]

    # Display images
    display_images(images,caption,grid_width_value)

def get_image_from_s3(bucket_name,image_file):
    # Download the image file from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=image_file)
    image_data = response['Body'].read()

    # Open the image using PIL
    image = Image.open(BytesIO(image_data))

    return image

def side_bar():
    with st.sidebar:

        st.header("Session Info")
        session_id = st.text_input(
                "Session ID",
                key=f"session-id",
            )
    return session_id

def read_inference_json(bucket_name, key):

    try:
        # Read inference.json file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        inference_json_data = response['Body'].read()

        # Parse the JSON data
        inference_json = json.loads(inference_json_data)

        return inference_json
    except s3_client.exceptions.NoSuchKey:
        # If inference.json doesn't exist, create a new empty file
        empty_inference_json = []
        s3_client.put_object(Body=json.dumps(empty_inference_json), Bucket=bucket_name, Key=key)

        return empty_inference_json

def write_image_to_s3(images, bucket_name, s3_image_prefix):
    s3_paths=[]
    for image in images:
        _id=str(uuid.uuid4())
        image_name=f"{_id}.jpeg"
        key=f"{s3_image_prefix}/{image_name}"
        # Create an in-memory byte stream for the JPEG image
        image_byte_stream = BytesIO()
        image.save(image_byte_stream, format='JPEG')
        image_byte_stream.seek(0)

        # Upload the image byte stream to S3
        s3_client.upload_fileobj(image_byte_stream, bucket_name, key)
        s3_uri=f"s3://{bucket_name}/{key}"
        s3_paths.append(s3_uri)
    return s3_paths

def write_json_to_s3(data, bucket_name, key):
    # Convert the JSON data to a string
    json_data = json.dumps(data)

    # Upload the JSON string to S3
    s3_client.put_object(Body=json_data, Bucket=bucket_name, Key=key)


def main():
    st.set_page_config(layout="wide")
    st.title("Stable Diffusion Playground")

    # Instantiate session ID
    session_id = side_bar()

    # Set s3 paths
    s3_inference_json_uri = f"{session_id}/inferences.json"
    s3_image_prefix = f"{session_id}/image"

    # read inference json
    inference_json = read_inference_json(S3_BUCKET,s3_inference_json_uri)

    tab1, tab2, tab3 = st.tabs(
        ["Search", "Generate", "History"]
    )


    if 'pinecone' not in st.session_state:
        pinecone.init(
            api_key=PINECONE_API_KEY,  # get for free from app.pinecone.io
            environment=PINECONE_ENVIRONMENT  # find next to API key
        )
        st.session_state['pinecone'] = pinecone.Index(PINECONE_INDEX_NAME)
        print('---- pinecone init -----')


    with tab1:
        search_tab(inference_json,s3_image_prefix,s3_inference_json_uri)

    with tab2:
        generate_tab(inference_json,s3_image_prefix,s3_inference_json_uri)

    with tab3:
        history_tab(s3_image_prefix, inference_json)





if __name__ == "__main__":
    main()
