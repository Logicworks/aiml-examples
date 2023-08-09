import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

import torch
import torch.nn as nn
import torch_neuronx
import base64
import torch
from io import BytesIO
import json
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple

class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        sample = self.unetwrap(sample, timestep.float().expand((sample.shape[0],)), encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]


# Optimized attention
def get_attention_scores(self, query, key, attn_mask):
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    # Check for square matmuls
    if(query.size() == key.size()):
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(dtype)

    return attention_probs

def custom_badbmm(a, b):
    bmm = torch.bmm(a, b)
    scaled = bmm * 0.125
    return scaled


#  -- Rest of sagemaker --


def model_fn(model_dir):
    model_id = "stabilityai/stable-diffusion-2-1-base"
    text_encoder_filename = os.path.join(model_dir, 'text_encoder/model.pt')
    decoder_filename = os.path.join(model_dir, 'vae_decoder/model.pt')
    unet_filename = os.path.join(model_dir, 'unet/model.pt')
    post_quant_conv_filename = os.path.join(model_dir, 'vae_post_quant_conv/model.pt')

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Load the compiled UNet onto two neuron cores.
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    device_ids = [0,1]
    pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

    # Load other compiled models onto a single neuron core.
    pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
    pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
    pipe.vae.decoder = torch.jit.load(decoder_filename)
    pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)

    return { "text2image": pipe }


def predict_fn(data, pipe):
    device = 'xla'
    # get prompt & parameters
    prompt = data.pop("inputs", data)
    # set valid HP for stable diffusion
    num_inference_steps = max(min(data.pop("num_inference_steps", 25), 100), 0)
    guidance_scale = data.pop("guidance_scale", 7.5)
    num_images_per_prompt = data.pop("num_images_per_prompt", 1)
    negative_prompt = data.pop("negative_prompt", None)

    width = max(min(data.pop("width", 512), 1024), 64)
    height = max(min(data.pop("height", 512), 1024), 64)
    width = (width // 8) * 8
    height = (height // 8) * 8

    # get mode (text2image, text2vector, image2image)
    mode = data.pop("mode", data)

    seed = data.pop("seed", None)
    latents = []
    seeds = []

    generator = torch.Generator()
    if mode == 'text2image':
        if seed:
            generator.manual_seed(seed)
            latents = torch.randn(
                (1, pipe[mode].unet.in_channels, height // 8, width // 8),
                generator = generator,
                device = device
            )
            latents = [latents]
            #we set the amount of images to 1, otherwise we're generating x times the same image.
            num_images_per_prompt = 1
        else:
            for _ in range(num_images_per_prompt):
                # Get a new random seed, store it and use it as the generator state
                _seed = generator.seed()
                seeds.append(_seed)
                generator = generator.manual_seed(_seed)

                image_latents = torch.randn(
                    (1, pipe[mode].unet.in_channels, height // 8, width // 8),
                    generator = generator,
                    device = device
                )
                latents.append(image_latents)
                # latents = image_latents if latents is None else torch.cat((latents, image_latents))

        # print(latents.shape)
        generated_images = []
        # run generation with parameters
        for counter, _ in enumerate(range(num_images_per_prompt)):
            generated_image = pipe['text2image'](
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                negative_prompt=negative_prompt,
                latents = latents[counter]
            )["images"]
            generated_images.append(generated_image[0])

        # create response
        encoded_images = []
        for image in generated_images:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

        # create response
        return {"generated_images": encoded_images, "seeds": seeds or [seed]}

    if mode == 'text2vector':
        # tokenize the prompt
        prompt_inputs = pipe['text2image'].tokenizer(
            prompt, return_tensors='pt',
            padding='max_length'
        )['input_ids']
        # create prompt encoding
        
        prompt_embeds = pipe['text2image'].text_encoder.neuron_text_encoder(prompt_inputs)
        # extract CLIP embedding
        prompt_embeds = prompt_embeds['pooler_output']

        prompt_embeds = prompt_embeds.cpu().detach().numpy()

        # Serialize the NumPy array to JSON
        prompt_embeds = json.dumps(prompt_embeds.tolist())

        return {"generated_vector": prompt_embeds}

    return {"error": "specify mode (text2image, or text2vector)"}
