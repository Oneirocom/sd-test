import torch
from torch import autocast
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import base64
from io import BytesIO
import os

def dummy_checker(images, **kwargs): return images, False

def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    repo = 'spaceinvader/fb'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(repo, subfolder="scheduler")
    model = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16, revision="main", scheduler=scheduler, use_auth_token=HF_AUTH_TOKEN).to("cuda")    

def inference(model_inputs:dict):
    global model

    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 768)
    width = model_inputs.get('width', 768)
    steps = model_inputs.get('steps', 20)
    guidance_scale = model_inputs.get('guidance_scale', 9)
    seed = model_inputs.get('seed', None)

    if not prompt: return {'message': 'No prompt was provided'}
    
    generator = None
    if seed: generator = torch.Generator("cuda").manual_seed(seed)
    
    model.safety_checker = dummy_checker
    
    with autocast("cuda"):
        image = model(prompt, guidance_scale=guidance_scale, height=height, width=width, num_inference_steps=steps, generator=generator).images[0]
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {'image_base64': image_base64}
