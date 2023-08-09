import base64
import torch
from io import BytesIO
import json
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline


def model_fn(model_dir):

    device = "cuda"
    image2image_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
    ).to(device)

    # Load stable diffusion and move it to the GPU
    pipe = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)


    return { "text2image": pipe, "image2image": image2image_pipe }


def predict_fn(data, pipe):
    device = "cuda"
    # get prompt & parameters
    prompt = data.pop("inputs", data)
    # set valid HP for stable diffusion
    num_inference_steps = max(min(data.pop("num_inference_steps", 25), 100), 0)
    guidance_scale = data.pop("guidance_scale", 7.5)
    strength = data.pop("strength", 0.8)
    num_images_per_prompt = data.pop("num_images_per_prompt", 1)
    negative_prompt = data.pop("negative_prompt", None)

    width = max(min(data.pop("width", 512), 1024), 64)
    height = max(min(data.pop("height", 512), 1024), 64)
    width = (width // 8) * 8
    height = (height // 8) * 8

    # get mode (text2image, text2vector, image2image)
    mode = data.pop("mode", data)
    init_image = data.pop("image", None)


    seed = data.pop("seed", None)
    latents = None
    seeds = []

    generator = torch.Generator(device=device)
    if mode == 'text2image':
        if seed:
            generator.manual_seed(seed)
            latents = torch.randn(
                (1, pipe[mode].unet.in_channels, height // 8, width // 8),
                generator = generator,
                device = device
            )
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
                latents = image_latents if latents is None else torch.cat((latents, image_latents))

        # run generation with parameters
        with torch.autocast("cuda"):
            generated_images = pipe['text2image'](
                [prompt] * num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                # num_images_per_prompt=num_images_per_prompt,
                negative_prompt=[negative_prompt] * num_images_per_prompt if negative_prompt else None,
                latents = latents
            )["images"]

        # create response
        encoded_images = []
        for image in generated_images:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

        # create response
        return {"generated_images": encoded_images, "seeds": seeds or [seed]}

    if mode == 'image2image' and init_image:
        seed = seed or generator.seed()
        # generators = [generator.manual_seed(seed)]*num_images_per_prompt
        # run generation with parameters
        init_image = base64.b64decode(init_image)
        buffer = BytesIO(init_image)
        init_image = Image.open(buffer).convert("RGB")
        init_image = init_image.resize((width, height))


        generated_images = pipe['image2image'](
            num_images_per_prompt=num_images_per_prompt,
            prompt=prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            negative_prompt=negative_prompt,
            # negative_prompt=[negative_prompt]*num_images_per_prompt if negative_prompt else None,
            # generator=generators,
        )["images"]

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
        ).to("cuda")
        # create prompt encoding
        prompt_embeds = pipe['text2image'].text_encoder(**prompt_inputs)
        # extract CLIP embedding
        prompt_embeds = prompt_embeds['pooler_output']

        prompt_embeds = prompt_embeds.cpu().detach().numpy()

        # Serialize the NumPy array to JSON
        prompt_embeds = json.dumps(prompt_embeds.tolist())

        return {"generated_vector": prompt_embeds}

    return {"error": "specify mode (text2image, text2vector, or image2image)"}
